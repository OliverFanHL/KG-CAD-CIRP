from __future__ import annotations

from .catalog import CLASSES


def build_graph(catalog, knowledge):
    nodes, edges = [], []
    for label in CLASSES:
        nodes.append({"id": f"class:{label}", "type": "class", "label": label})
    for function in knowledge["functions"]:
        nodes.append({"id": f"function:{function['function_id']}", "type": "function", **function})
    for relation in knowledge["class_functions"]:
        edges.append({"subject": f"class:{relation['class_id']}", "predicate": "potential_function",
                      "object": f"function:{relation['function_id']}", **relation,
                      "semantics": "class-level retrieval prior; does not verify a part's function"})
    for part in catalog["parts"]:
        pid = f"part:{part['part_id']}"
        nodes.append({"id": pid, "type": "part", "relative_path": part["relative_path"],
                      "split": part["split"], "file_sha256": part["file_sha256"]})
        edges.append({"subject": pid, "predicate": "dataset_label", "object": f"class:{part['label']}",
                      "status": "observed_dataset", "source": part["relative_path"].split("/")[0],
                      "serving_allowed": False})
        if part["prediction"]:
            for label, probability in part["prediction"]["probabilities"].items():
                edges.append({"subject": pid, "predicate": "predicted_category", "object": f"class:{label}",
                              "probability": probability, "status": "predicted", "source": part["prediction"]["source"],
                              "model": part["prediction"]["model"]})
        for attribute, measurement in part["measurements"].items():
            mid = f"measurement:{part['part_id']}:{attribute}"
            nodes.append({"id": mid, "type": "measurement", "attribute": attribute, **measurement})
            edges.append({"subject": pid, "predicate": "has_measurement", "object": mid, **measurement})
        for interface in part["interfaces"]:
            iid = f"interface:{part['part_id']}:{interface['interface_id']}"
            nodes.append({"id": iid, "type": "interface", **interface})
            edges.append({"subject": pid, "predicate": "has_reviewed_interface", "object": iid,
                          "status": "curated", "source": interface["source"]})
    return {"schema_version": 1, "nodes": nodes, "edges": edges, "knowledge_sha256": knowledge["sha256"],
            "reasoning_scope": "retrieval priors, numeric constraints, reviewed diameter clearance screening"}
