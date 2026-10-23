"""Prepare a larger, evidence-linked AI review cohort.

This utility deliberately writes to a new annotation directory.  It preserves
the original 674 source-bound records and never represents the expanded review
as an independent human assessment.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from cadkg.io import read_csv, read_json, write_csv, write_json


ROOT = Path("outputs/intent_batch2")
OUT = Path("outputs/intent_batch2_expanded_2x")
ANNOTATIONS = Path("annotations/intent_batch2_expanded_2x")
TARGET_PARTS = 1350


# Source categories are used only to balance sampling. New relevance grades
# require a separate source-view assessment and remain null until then.


def choose_parts(catalog, existing):
    """Keep all prior reviewed CADs, then fill evenly across source classes."""
    gallery = set(read_json(ROOT / 'protocol.json')['gallery'])
    if not set(existing) <= gallery:
        raise ValueError('A preserved annotation falls outside the frozen evaluation gallery')
    if not len(existing) <= TARGET_PARTS <= len(gallery):
        raise ValueError('Requested cohort size is incompatible with the eligible gallery')
    test = [p for p in catalog["parts"] if p["part_id"] in gallery]
    by_label = defaultdict(list)
    for part in test:
        if part["part_id"] not in existing:
            by_label[part["label"]].append(part)
    rng = np.random.default_rng(20261020)
    for rows in by_label.values():
        rng.shuffle(rows)
    chosen = set(existing)
    # Round-robin prevents the extra cohort from being dominated by one class.
    labels = sorted(by_label)
    position = defaultdict(int)
    while len(chosen) < TARGET_PARTS:
        advanced = False
        for label in labels:
            rows = by_label[label]
            while position[label] < len(rows) and rows[position[label]]["part_id"] in chosen:
                position[label] += 1
            if position[label] < len(rows) and len(chosen) < TARGET_PARTS:
                chosen.add(rows[position[label]]["part_id"])
                position[label] += 1
                advanced = True
        if not advanced:
            raise RuntimeError("Test gallery is too small for requested cohort")
    return chosen


def main():
    OUT.mkdir(exist_ok=True)
    ANNOTATIONS.mkdir(exist_ok=True)
    catalog = read_json(ROOT / "bundle/catalog.json")
    protocol = read_json(ROOT / 'protocol.json')
    family_ids = [f['family_id'] for f in protocol['families']]
    parts = {p["part_id"]: p for p in catalog["parts"]}
    original = read_json("annotations/intent_batch2/judgments.json")
    original_by_pair = {(r["family_id"], r["part_id"]): r for r in original["judgments"]}
    existing = {r["part_id"] for r in original["judgments"]}
    selected = choose_parts(catalog, existing)
    if len(selected) != TARGET_PARTS:
        raise RuntimeError("Cohort size mismatch")

    # Reuse the original neutral ID whenever it exists.  New IDs begin after
    # the current manifest so image provenance remains unambiguous.
    current = {r["part_id"]: r for r in read_csv(ROOT / "pool_manifest.csv")}
    manifest = []
    next_id = 1
    for pid in sorted(selected):
        if pid in current:
            item = current[pid]
            code = item["inspection_id"]
        else:
            while f"X{next_id:04d}" in {m["inspection_id"] for m in manifest}:
                next_id += 1
            code = f"X{next_id:04d}"
            next_id += 1
        part = parts[pid]
        manifest.append({"inspection_id": code, "part_id": pid,
                         "source_sha256": part["file_sha256"], "split": "test",
                         "family_ids": '|'.join(family_ids),
                         "source_label_for_sampling_only": part["label"]})
    write_csv(OUT / "pool_manifest.csv", manifest)
    write_json(OUT / 'family_pools.json', {fid: sorted(selected) for fid in family_ids})
    write_json(OUT / 'protocol.json', {**protocol,
        'annotation_policy': '1350 eligible held-out CADs against all 11 frozen task rubrics; category labels used for sampling only; new pairs remain unreviewed.'})
    if not (OUT / 'bundle').exists():
        shutil.copytree(ROOT / 'bundle', OUT / 'bundle')
    if not (OUT / 'rankings.csv').exists():
        shutil.copyfile(ROOT / 'rankings.csv', OUT / 'rankings.csv')

    families = read_json("annotations/intent_batch2/families.json")
    task_rows = []
    for family in families:
        for item in manifest:
            old = original_by_pair.get((family["family_id"], item["part_id"]))
            if old:
                task_rows.append({**old, "geometry_evidence": str(OUT / f"evidence/{item['inspection_id']}.png")})
            else:
                grade = None
                observation = ''
                rationale = ''
                task_rows.append({
                    "family_id": family["family_id"], "part_id": item["part_id"],
                    "inspection_id": item["inspection_id"], "source_sha256": item["source_sha256"],
                    "grade": grade, "decision": None, "review_status": "unreviewed",
                    "observation": observation,
                    "rationale": rationale,
                    "inspection_origin": "Source-view inspection pending; category-derived relevance forbidden",
                    "geometry_evidence": str(OUT / f"evidence/{item['inspection_id']}.png"),
                    "human_expert_reviewed": False,
                })
    write_json(ANNOTATIONS / "judgments.json", {
        "source": "review_queue_pending_visual_inspection",
        "human_expert_reviewed": False,
        "reviewer": "Unassigned for new task-part pairs",
        "target_unique_cads": TARGET_PARTS,
        "task_part_records": len(task_rows),
        "preserved_prior_ai_judgments": len(original_by_pair),
        "ungraded_records": sum(row["grade"] is None for row in task_rows),
        "disclosure": "Original 674 AI inspections preserved. Additional pairs require individual visual assessment; no class-derived grade is assigned. This is a review queue, not a complete labeled dataset.",
        "judgments": task_rows,
    })
    # Existing views are copied with their neutral IDs.
    evidence_dir = OUT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    previous_expansion = {r['part_id']: r for r in read_csv('outputs/intent_batch2_expanded/pool_manifest.csv')}
    for item in manifest:
        code = item["inspection_id"]
        previous = previous_expansion.get(item['part_id'])
        if previous and not (evidence_dir / f'{code}.json').exists():
            oldcode = previous['inspection_id']
            directory = Path('outputs/intent_batch2_expanded/evidence')
            binding = read_json(directory / f'{oldcode}.json')
            if binding['part_id'] != item['part_id'] or binding['source_sha256'] != item['source_sha256']:
                raise ValueError('Previous evidence binding mismatch')
            for suffix in ('npz', 'png'):
                shutil.copyfile(directory / f'{oldcode}.{suffix}', evidence_dir / f'{code}.{suffix}')
            binding.update(inspection_id=code, rendered_view_id=oldcode)
            write_json(evidence_dir / f'{code}.json', binding)
        for suffix in ("json", "npz", "png"):
            source = ROOT / "evidence" / f"{code}.{suffix}"
            target = evidence_dir / f"{code}.{suffix}"
            if source.exists() and not target.exists():
                shutil.copyfile(source, target)
    if '--render' in __import__('sys').argv:
        from cadkg.experiments.intent import evidence
        evidence(evidence_dir, manifest, catalog, reuse="outputs/full_scale", workers=4)
    print(json.dumps({"unique_cads": len(manifest), "task_part_records": len(task_rows),
                      "preserved": len(original_by_pair),
                      "pending": sum(row["grade"] is None for row in task_rows)}))


if __name__ == "__main__":
    main()
