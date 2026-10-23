from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .cad_features import _load_shape
from .dataset import TMCAD_CLASSES
from .provenance import class_label_mapping, display_label, with_source_labels
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def _set_equal_3d(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_step(step_path: str | Path, out_path: str | Path, relative_tolerance: float = 0.01) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    shape = _load_shape(step_path)
    bb = shape.BoundingBox()
    max_extent = max(float(bb.xlen), float(bb.ylen), float(bb.zlen), 1e-6)
    vertices, triangles = shape.tessellate(max(max_extent * relative_tolerance, 1e-5))
    xyz = np.asarray([v.toTuple() for v in vertices], dtype=float)
    tri = np.asarray(triangles, dtype=int)
    if len(tri) > 25000:
        idx = np.linspace(0, len(tri) - 1, 25000, dtype=int)
        tri = tri[idx]
    polys = xyz[tri]

    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(polys, linewidths=0.08, edgecolors="k", alpha=0.95)
    ax.add_collection3d(mesh)
    _set_equal_3d(ax, xyz)
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def _img_tag(path: Path, root: Path, width: int = 190) -> str:
    rel = path.relative_to(root).as_posix()
    return f'<img src="{html.escape(rel)}" width="{width}">'


def _label_text(label: str, mapping: dict[str, list[str]]) -> str:
    original = display_label(label, mapping)
    return original if original == label else f"{original} (model: {label})"


def _render_for_id(sample_id: str, manifest_idx: pd.DataFrame, render_dir: Path) -> Path | None:
    if sample_id not in manifest_idx.index:
        return None
    row = manifest_idx.loc[sample_id]
    out = render_dir / f"{sample_id}.png"
    try:
        return render_step(row["absolute_path"], out)
    except Exception as exc:
        LOGGER.warning("Could not render %s: %s", sample_id, exc)
        return None


def _relative_path_for_id(sample_id: str, manifest_idx: pd.DataFrame) -> str:
    return str(manifest_idx.loc[sample_id, "relative_path"]) if sample_id in manifest_idx.index else sample_id


def generate_qualitative_report(
    descriptor_csv: str | Path,
    classification_dir: str | Path,
    retrieval_dir: str | Path,
    output_dir: str | Path,
    max_queries: int = 12,
    top_k: int = 5,
) -> Path:
    out = ensure_dir(output_dir)
    render_dir = ensure_dir(out / "renders")
    desc = with_source_labels(pd.read_csv(descriptor_csv, low_memory=False,
                                          dtype={"sample_id": str}))
    mapping = class_label_mapping(desc, TMCAD_CLASSES)
    manifest_idx = desc.drop_duplicates("sample_id").set_index("sample_id")

    parts: list[str] = [
        "<html><head><meta charset='utf-8'><title>Jev-TMCAD qualitative report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;} table{border-collapse:collapse;margin-bottom:28px;} td,th{border:1px solid #bbb;padding:7px;vertical-align:top;} .ok{background:#eef8ee}.bad{background:#fdeeee}.small{font-size:12px;color:#444}</style>",
        "</head><body>",
        "<h1>Jev-TMCAD qualitative report</h1>",
        "<p>All predictions were made from label-free deterministic CAD descriptors. Folder names, filenames, paths, sample IDs, and ground-truth classes are not included in the Jev state.</p>",
    ]

    # Classification examples.
    pred_path = Path(classification_dir) / "predictions" / "jev_zero_shot.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, dtype={"sample_id": str})
        pred["correct"] = pred["true_label"] == pred["predicted_class"]
        correct_hi = pred[pred["correct"]].sort_values("max_probability", ascending=False).head(6)
        wrong_hi = pred[~pred["correct"]].sort_values("max_probability", ascending=False).head(10)
        uncertain = pred.sort_values("max_probability", ascending=True).head(6)
        examples = pd.concat(
            [
                correct_hi.assign(group="High-confidence correct"),
                wrong_hi.assign(group="High-confidence errors"),
                uncertain.assign(group="Lowest-confidence cases"),
            ],
            ignore_index=True,
        ).drop_duplicates("sample_id")
        parts.append("<h2>Classification examples</h2><table><tr><th>Group</th><th>Part</th><th>True</th><th>Predicted</th><th>Probability</th></tr>")
        for _, r in examples.iterrows():
            sid = str(r["sample_id"])
            img = _render_for_id(sid, manifest_idx, render_dir)
            cls = "ok" if bool(r["correct"]) else "bad"
            parts.append(
                f"<tr class='{cls}'><td>{html.escape(str(r['group']))}</td>"
                f"<td>{_img_tag(img, out) if img else sid}<div class='small'>{html.escape(_relative_path_for_id(sid, manifest_idx))}</div></td>"
                f"<td>{html.escape(_label_text(str(r['true_label']), mapping))}</td>"
                f"<td>{html.escape(_label_text(str(r['predicted_class']), mapping))}</td>"
                f"<td>{float(r['max_probability']):.3f}</td></tr>"
            )
        parts.append("</table>")

    # Retrieval examples: prefer class-balanced query sampling.
    top_path = Path(retrieval_dir) / "retrieval_topk.csv"
    if top_path.exists():
        top = pd.read_csv(top_path, dtype={"query_id": str, "candidate_id": str})
        methods = list(top["method"].drop_duplicates())
        query_meta = top[["query_id", "query_label"]].drop_duplicates()
        selected = []
        for label, grp in query_meta.groupby("query_label", sort=True):
            if len(selected) >= max_queries:
                break
            selected.append(grp.iloc[0].to_dict())
        if len(selected) < max_queries:
            used = {x["query_id"] for x in selected}
            for _, r in query_meta.iterrows():
                if r["query_id"] not in used:
                    selected.append(r.to_dict())
                    if len(selected) >= max_queries:
                        break

        parts.append("<h2>Retrieval examples</h2>")
        for q in selected:
            qid = str(q["query_id"])
            qimg = _render_for_id(qid, manifest_idx, render_dir)
            parts.append(f"<h3>Query: {html.escape(_label_text(str(q['query_label']), mapping))}</h3>")
            parts.append("<table><tr><th>Method</th><th>Query</th>" + "".join(f"<th>Rank {i}</th>" for i in range(1, top_k + 1)) + "</tr>")
            for method in methods:
                subset = top[(top["method"] == method) & (top["query_id"].astype(str) == qid)].sort_values("rank").head(top_k)
                row_html = f"<tr><td>{html.escape(str(method))}</td><td>{_img_tag(qimg, out) if qimg else qid}<div class='small'>{html.escape(_label_text(str(q['query_label']), mapping))}<br>{html.escape(_relative_path_for_id(qid, manifest_idx))}</div></td>"
                for _, cand in subset.iterrows():
                    cid = str(cand["candidate_id"])
                    cimg = _render_for_id(cid, manifest_idx, render_dir)
                    mark = "✓" if int(cand["relevant"]) else "✗"
                    row_html += (
                        f"<td class='{'ok' if int(cand['relevant']) else 'bad'}'>"
                        f"{_img_tag(cimg, out) if cimg else cid}"
                        f"<div class='small'>{mark} {html.escape(_label_text(str(cand['candidate_label']), mapping))}<br>{html.escape(_relative_path_for_id(cid, manifest_idx))}<br>score={float(cand['score']):.3f}</div></td>"
                    )
                row_html += "</tr>"
                parts.append(row_html)
            parts.append("</table>")

    parts.append("</body></html>")
    report = out / "index.html"
    report.write_text("\n".join(parts), encoding="utf-8")
    LOGGER.info("Qualitative report written to %s", report)
    return report
