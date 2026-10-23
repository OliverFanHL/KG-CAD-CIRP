"""Configuration, integrity-checked bundles and legacy blank annotation packs."""
from pathlib import Path
import math
import shutil
from .io import object_hash, read_json, write_json, read_csv, write_csv
from .graph import build_graph
from .knowledge import load_knowledge

ROOT = Path(__file__).resolve().parent.parent


def config_from(path):
    config = read_json(path) if path else read_json(ROOT / "configs" / "core.json")
    if config.get("unknown_policy", "exclude") not in {"exclude", "keep"}:
        raise ValueError("unknown_policy must be exclude/keep")
    if not config.get("gallery_splits") or any(s not in {"train", "val", "test"} for s in config["gallery_splits"]):
        raise ValueError("gallery_splits must contain valid part splits")
    weights = config.get("weights", {})
    if any(k not in {"geometry", "category", "class_reference", "semantic", "constraint_only"} for k in weights):
        raise ValueError("Unknown signal weight")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Signal weights must be finite positive numbers")
    if not isinstance(config.get("seed", 17), int) or not isinstance(config.get("bootstrap_samples", 1000), int) or config.get("bootstrap_samples", 1000) < 1:
        raise ValueError("Seed and positive bootstrap sample count must be integers")
    return config


def save_bundle(catalog, annotation_dir, output, knowledge_policy="curated_only"):
    for part in catalog["parts"]:
        part["interfaces"] = []
    knowledge = load_knowledge(annotation_dir, catalog, knowledge_policy)
    graph = build_graph(catalog, knowledge)
    write_json(Path(output) / "catalog.json", catalog)
    write_json(Path(output) / "knowledge.json", knowledge)
    write_json(Path(output) / "graph.json", graph)
    write_json(Path(output) / "bundle_manifest.json", {"catalog_sha256": object_hash(catalog), "knowledge_sha256": object_hash(knowledge),
                                                      "graph_sha256": object_hash(graph)})
    return knowledge, graph


def load_bundle(directory):
    directory = Path(directory)
    values = [read_json(directory / name) for name in ["catalog.json", "knowledge.json", "graph.json"]]
    manifest = directory / "bundle_manifest.json"
    if manifest.exists():
        expected = read_json(manifest)
        for name, value in zip(["catalog", "knowledge", "graph"], values):
            if object_hash(value) != expected[f"{name}_sha256"]:
                raise ValueError("Bundle was modified; rebuild rather than editing serving artifacts")
    elif values[0].get("dataset_type") != "synthetic_fixture":
        raise ValueError("Real bundle lacks integrity manifest")
    return values


def prepare_annotations(directory, catalog=None, queries=None, config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "data/annotations" / "templates").iterdir():
        target = directory / path.name
        if not target.exists():
            shutil.copyfile(path, target)
    if catalog:
        manifest = [{"part_id": p["part_id"], "split": p["split"], "relative_path": p["relative_path"],
                     "file_sha256": p["file_sha256"] or "", "cylinder_candidates": len(p.get("cylinder_candidates", []))} for p in catalog["parts"]]
        if not (directory / "part_manifest.csv").exists():
            write_csv(directory / "part_manifest.csv", manifest)
        if queries:
            config = config or config_from(None)
            gallery = [p for p in catalog["parts"] if p["split"] in config["gallery_splits"] and p["prediction"] is not None
                       and (p["embedding"] is not None if config["geometry_kind"] == "embedding" else bool(p["features"]))]
            rows = []
            for query in queries:
                ref = next((p for p in catalog["parts"] if p["part_id"] == query.get("reference_part_id")), None)
                for part in gallery:
                    if part["part_id"] == query.get("reference_part_id") or (ref and ref["file_sha256"] and part["file_sha256"] == ref["file_sha256"]):
                        continue
                    rows.append({"query_id": query["query_id"], "part_id": part["part_id"], "grade": "", "constraint_verdict": "",
                                 "reviewer": "", "evidence": "", "source": "independent_expert", "split": query.get("split", "test")})
            target = directory / "judgments_to_fill.csv"
            if target.exists():
                raise ValueError("Refusing to overwrite annotation work")
            write_csv(target, rows)


