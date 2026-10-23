from __future__ import annotations

import numpy as np

from .catalog import CLASSES, FEATURE_KEYS
from .query import validate_query
from .reasoning import check_constraints

METHODS = {
    "category": {"signals": ["category"], "constraints": False},
    "geometry": {"signals": ["geometry"], "constraints": False},
    "geometry_class": {"signals": ["geometry", "category", "class_reference"], "constraints": False},
    "flat_full": {"signals": ["geometry", "category", "class_reference", "semantic"], "constraints": True, "flat": True},
    "kg_no_geometry": {"signals": ["category", "class_reference", "semantic"], "constraints": True},
    "kg_no_class": {"signals": ["geometry"], "constraints": True},
    "kg_no_constraints": {"signals": ["geometry", "category", "class_reference", "semantic"], "constraints": False},
    "kg_full": {"signals": ["geometry", "category", "class_reference", "semantic"], "constraints": True},
}


LABELS = {"category": "Category probabilities", "category_from_intent": "Intent → category",
          "geometry": "Exemplar cosine", "geometry_class": "Cosine + classifier",
          "flat_full": "Matched flat table", "kg_no_geometry": "Remove exemplar cosine",
          "kg_no_class": "Remove classifier / function branch", "kg_no_constraints": "Remove dimension filter",
          "kg_full": "Full framework"}


REFERENCE_METHODS = ["category_from_intent", *[m for m in METHODS if m != "category"]]


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 1e-12:
        return None
    return float(np.clip(a @ b / norm, -1, 1))


class Retriever:
    def __init__(self, catalog, knowledge, graph, config):
        self.catalog, self.knowledge, self.graph, self.config = catalog, knowledge, graph, config
        self.parts = {p["part_id"]: p for p in catalog["parts"]}
        self.function_edges = {}
        for edge in graph["edges"]:
            allowed = {"curated"}
            if knowledge.get("admission_policy") == "include_source_supported":
                allowed.add("source_supported")
            if edge["predicate"] == "potential_function" and edge["status"] in allowed:
                self.function_edges.setdefault(edge["function_id"], []).append(edge)
        self.prediction_edges = {}
        for edge in graph["edges"]:
            if edge["predicate"] == "predicted_category":
                self.prediction_edges.setdefault(edge["subject"], {})[edge["object"].removeprefix("class:")] = edge
        self.vectors = {}
        kind = config.get("geometry_kind", "descriptor")
        if kind not in {"descriptor", "embedding"}:
            raise ValueError("geometry_kind must be descriptor or embedding")
        if kind == "embedding":
            for part in catalog["parts"]:
                if part["embedding"] is not None:
                    self.vectors[part["part_id"]] = np.asarray(part["embedding"], dtype=float)
            dims = {len(v) for v in self.vectors.values()}
            if len(dims) != 1:
                raise ValueError("Embedding backend needs one consistent nonempty representation")
            self.transform = {"kind": "embedding", "normalization": "cosine; no test-fitted transform"}
        else:
            train = [p for p in catalog["parts"] if p["split"] == "train"]
            if not train:
                raise ValueError("Descriptor normalization needs training parts")
            keys = [k for k in FEATURE_KEYS if any(p["features"].get(k) is not None for p in train)]
            if not keys:
                raise ValueError("No reusable descriptor features; import descriptors first")
            def raw(part):
                return [np.nan if part["features"].get(k) is None else part["features"][k] for k in keys]
            train_x = np.asarray([raw(p) for p in train])
            medians = np.nanmedian(train_x, axis=0)
            train_x = np.where(np.isfinite(train_x), train_x, medians)
            mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
            scale[scale < 1e-8] = 1.0
            for part in catalog["parts"]:
                if not any(part["features"].get(k) is not None for k in keys):
                    continue
                x = np.asarray(raw(part), dtype=float)
                x = np.where(np.isfinite(x), x, medians)
                self.vectors[part["part_id"]] = (x-mean)/scale
            self.transform = {"kind": "descriptor", "fit_split": "train", "keys": keys,
                              "medians": medians.tolist(), "mean": mean.tolist(), "scale": scale.tolist()}

    def candidate_ids(self):
        ids = [p["part_id"] for p in self.catalog["parts"] if p["split"] in self.config.get("gallery_splits", ["test"])]
        if self.config.get("matched_cohort", True):
            ids = [pid for pid in ids if pid in self.vectors and np.linalg.norm(self.vectors[pid]) > 1e-12
                   and self.parts[pid]["prediction"] is not None]
        return sorted(ids)

    def probabilities(self, pid, flat=False):
        if flat:
            prediction = self.parts[pid]["prediction"]
            return prediction["probabilities"] if prediction else None
        edges = self.prediction_edges.get(f"part:{pid}")
        return {c: edges[c]["probability"] if c in edges else 0.0 for c in CLASSES} if edges else None

    def semantic(self, pid, function_ids, flat=False):
        probs = self.probabilities(pid, flat)
        if probs is None or not function_ids:
            return None, []
        paths, scores = [], []
        for fid in function_ids:
            relations = ([r for r in self.knowledge["class_functions"] if r["function_id"] == fid]
                         if flat else self.function_edges.get(fid, []))
            if not relations:
                return None, []
            score = 0.0
            for relation in relations:
                category = relation["class_id"]
                weight = probs.get(category, 0.0) * relation["relevance"] * relation["confidence"]
                score += weight
                if weight > 0:
                    paths.append({"path": [f"part:{pid}", f"class:{category}", f"function:{fid}"],
                                  "contribution": weight, "category_status": "predicted", "relation_status": relation["status"],
                                  "relation_scope": relation["scope"], "caveat": relation["caveat"], "source": relation["source"],
                                  "interpretation": "retrieval prior, not verified part function"})
            scores.append(score)
        # Conjunction: each requested function must receive support; no arbitrary OR expansion.
        return min(scores), sorted(paths, key=lambda p: -p["contribution"])

    def rank(self, query, method="kg_full", k=10, candidate_ids=None):
        query = validate_query(query, self.knowledge)
        if method not in METHODS or k < 1:
            raise ValueError("Unknown method or invalid k")
        recipe = METHODS[method]
        reference_id = query["reference_part_id"]
        if reference_id and reference_id not in self.parts:
            raise ValueError("Unknown reference part")
        gallery = candidate_ids if candidate_ids is not None else self.candidate_ids()
        if any(pid not in self.parts for pid in gallery):
            raise ValueError("Unknown gallery part")
        rows, rejected, unknown = [], 0, 0
        for pid in gallery:
            part = self.parts[pid]
            if pid == reference_id:
                continue
            if reference_id and part["file_sha256"] and part["file_sha256"] == self.parts[reference_id]["file_sha256"]:
                continue
            signals, paths = {}, []
            flat = recipe.get("flat", False)
            probabilities = self.probabilities(pid, flat)
            if "geometry" in recipe["signals"] and reference_id in self.vectors and pid in self.vectors:
                value = cosine(self.vectors[pid], self.vectors[reference_id])
                if value is not None:
                    signals["geometry"] = (value+1)/2
            if "category" in recipe["signals"] and probabilities and query["category_ids"]:
                signals["category"] = sum(probabilities.get(c, 0) for c in query["category_ids"])
            if "class_reference" in recipe["signals"] and reference_id and probabilities:
                reference_probs = self.probabilities(reference_id, flat)
                if reference_probs:
                    a = np.asarray([probabilities.get(c, 0) for c in CLASSES])
                    b = np.asarray([reference_probs.get(c, 0) for c in CLASSES])
                    signals["class_reference"] = float(a @ b)
            if "semantic" in recipe["signals"]:
                value, paths = self.semantic(pid, query["function_ids"], flat)
                if value is not None:
                    signals["semantic"] = value
            reasoning = check_constraints(part, query["constraints"])
            if recipe["constraints"] and reasoning["status"] == "fail":
                rejected += 1
                continue
            if recipe["constraints"] and reasoning["status"] == "unknown":
                unknown += 1
                if self.config.get("unknown_policy", "exclude") == "exclude":
                    rejected += 1
                    continue
            if (not signals and recipe["constraints"] and query["constraints"] and not query["function_ids"]
                    and not query["category_ids"] and not reference_id):
                # A purely numeric discovery query can return feasible parts without semantic evidence.
                signals["constraint_only"] = float(reasoning["status"] == "pass")
            if not signals:
                continue
            weights = self.config.get("weights", {})
            denominator = sum(weights.get(name, 1.0) for name in signals)
            score = sum(weights.get(name, 1.0)*value for name, value in signals.items())/denominator
            rows.append({"part_id": pid, "score": score, "signals": signals, "reasoning": reasoning, "paths": paths,
                         "category_prediction": part["prediction"], "functional_suitability": "unverified",
                         "intent_support": "class_prior_only" if signals.get("semantic", 0) > 0 else "not_established",
                         "retrieval_status": "candidate", "relative_path": part["relative_path"]})
        rows.sort(key=lambda row: (-row["score"], row["part_id"]))
        return {"query": query, "method": method, "geometry_kind": self.transform["kind"],
                "status": "ranked" if rows else "abstained", "gallery_count": len(gallery), "ranked_count": len(rows),
                "constraint_rejected": rejected, "constraint_unknown": unknown,
                "constraints_enforced": recipe["constraints"], "results": [{"rank": i+1, **row} for i, row in enumerate(rows[:k])],
                "limitations": ["Function scores are class-level priors propagated through predictions.",
                                "No claim about strength, safety, load capacity or full assembly compatibility."]}
