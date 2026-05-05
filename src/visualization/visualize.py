#!/usr/bin/env python3
"""
H5 Structure Visualizer

Scans a folder recursively for .h5/.hdf5/.nc files, inspects the internal
structure, and writes an HTML report showing groups, datasets, shapes, dtypes,
attributes, and lightweight previews for numeric arrays.

Designed for repo layouts like:
manual-data/
  parity/
    mpr_A1_Cq.h5
    mpr_A2_Cq.h5
    mpr_B1_Cq.h5
  x_loop/
    xmpr_Cq.h5
  z_loop/
    zmpr_Cq.h5
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_EXTS = {".h5", ".hdf5", ".nc"}
MAX_SAMPLE_ELEMENTS = 2_000_000
MAX_STATS_ELEMENTS = 2_000_000
MAX_PREVIEW_POINTS_1D = 4000
MAX_TEXT_PREVIEW = 12


@dataclass
class DatasetInfo:
    path: str
    name: str
    shape: Tuple[int, ...]
    dtype: str
    ndim: int
    size: int
    attrs: Dict[str, Any] = field(default_factory=dict)
    kind: str = "unknown"
    preview_path: Optional[str] = None
    sample_text: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None


@dataclass
class GroupInfo:
    path: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileInfo:
    file_path: str
    rel_path: str
    groups: List[GroupInfo] = field(default_factory=list)
    datasets: List[DatasetInfo] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def clean_attr_value(value: Any) -> Any:
    """Convert HDF5 attrs into JSON/HTML friendly values."""
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"S", "O", "U"}:
                return [clean_attr_value(v) for v in value.tolist()]
            if value.size <= 20:
                return value.tolist()
            return f"array(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    except Exception as exc:
        return f"<unreadable attr: {exc}>"


def attrs_to_dict(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        for key, value in obj.attrs.items():
            out[str(key)] = clean_attr_value(value)
    except Exception as exc:
        out["<attr_error>"] = str(exc)
    return out


def format_shape(shape: Tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    return " x ".join(str(x) for x in shape)


def dataset_kind(ds: h5py.Dataset) -> str:
    kind = ds.dtype.kind
    if kind in {"i", "u", "f"}:
        return "numeric"
    if kind == "b":
        return "boolean"
    if kind in {"S", "O", "U"}:
        return "text"
    if kind == "c":
        return "complex"
    return f"dtype:{kind}"


def safe_read_sample(ds: h5py.Dataset) -> np.ndarray:
    """Read a bounded sample for plotting/statistics."""
    if ds.size == 0:
        return np.array([])

    if ds.size <= MAX_SAMPLE_ELEMENTS:
        return np.asarray(ds[()])

    # Sample a slice that preserves structure where possible.
    shape = ds.shape
    slicer = []
    remaining_budget = MAX_SAMPLE_ELEMENTS
    for dim in shape:
        if dim <= 1:
            slicer.append(slice(None))
            continue
        if remaining_budget <= 1:
            slicer.append(0)
            continue
        take = min(dim, max(1, int(round(remaining_budget ** (1 / max(1, len(shape)))))))
        slicer.append(slice(0, take))
    sample = np.asarray(ds[tuple(slicer)])
    return sample


def numeric_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(arr)
    if flat.size == 0:
        return {"size": 0}
    if np.iscomplexobj(flat):
        mag = np.abs(flat)
        base = mag
        prefix = "abs_"
    else:
        base = flat
        prefix = ""
    base = base[np.isfinite(base)] if np.issubdtype(base.dtype, np.number) else base
    if base.size == 0:
        return {"size": int(flat.size), "note": "no finite numeric values"}
    return {
        "size": int(flat.size),
        f"{prefix}min": float(np.min(base)),
        f"{prefix}max": float(np.max(base)),
        f"{prefix}mean": float(np.mean(base)),
        f"{prefix}std": float(np.std(base)),
    }


def text_preview(ds: h5py.Dataset) -> str:
    arr = ds[()]
    try:
        if isinstance(arr, bytes):
            return arr.decode("utf-8", errors="replace")
        if np.isscalar(arr):
            return str(clean_attr_value(arr))
        flat = np.ravel(arr)
        items = [str(clean_attr_value(x)) for x in flat[:MAX_TEXT_PREVIEW]]
        suffix = " ..." if flat.size > MAX_TEXT_PREVIEW else ""
        return "[" + ", ".join(items) + "]" + suffix
    except Exception as exc:
        return f"<text preview unavailable: {exc}>"


def preview_plot(arr: np.ndarray, output_png: Path, title: str) -> Optional[str]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return None

    plt.figure(figsize=(7, 4.5))
    try:
        if arr.ndim == 0:
            plt.text(0.5, 0.5, str(arr.item()), ha="center", va="center")
            plt.axis("off")
        elif arr.ndim == 1:
            y = arr
            if y.size > MAX_PREVIEW_POINTS_1D:
                idx = np.linspace(0, y.size - 1, MAX_PREVIEW_POINTS_1D).astype(int)
                y = y[idx]
            if np.iscomplexobj(y):
                y = np.abs(y)
            plt.plot(y)
            plt.xlabel("index")
            plt.ylabel("value")
        else:
            view = arr
            while view.ndim > 2:
                mid = view.shape[0] // 2
                view = view[mid]
            if view.ndim == 1:
                y = np.abs(view) if np.iscomplexobj(view) else view
                plt.plot(y)
                plt.xlabel("index")
                plt.ylabel("value")
            else:
                img = np.abs(view) if np.iscomplexobj(view) else view
                if img.shape[0] > 512 or img.shape[1] > 512:
                    step0 = max(1, math.ceil(img.shape[0] / 512))
                    step1 = max(1, math.ceil(img.shape[1] / 512))
                    img = img[::step0, ::step1]
                plt.imshow(img, aspect="auto")
                plt.colorbar()
                plt.xlabel("axis -1")
                plt.ylabel("axis -2")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_png, dpi=120)
        return output_png.name
    except Exception:
        return None
    finally:
        plt.close()


def inspect_file(file_path: Path, root: Path, assets_dir: Path) -> FileInfo:
    info = FileInfo(file_path=str(file_path), rel_path=str(file_path.relative_to(root)))
    try:
        with h5py.File(file_path, "r") as f:
            info.attrs = attrs_to_dict(f)

            def visitor(name: str, obj: Any) -> None:
                try:
                    if isinstance(obj, h5py.Group):
                        info.groups.append(
                            GroupInfo(
                                path="/" + name if name else "/",
                                name=name.split("/")[-1] if name else "/",
                                attrs=attrs_to_dict(obj),
                            )
                        )
                    elif isinstance(obj, h5py.Dataset):
                        ds_info = DatasetInfo(
                            path="/" + name,
                            name=name.split("/")[-1],
                            shape=tuple(int(x) for x in obj.shape),
                            dtype=str(obj.dtype),
                            ndim=int(obj.ndim),
                            size=int(obj.size),
                            attrs=attrs_to_dict(obj),
                            kind=dataset_kind(obj),
                        )

                        if ds_info.kind in {"numeric", "complex", "boolean"}:
                            sample = safe_read_sample(obj)
                            ds_info.stats = numeric_stats(sample)
                            preview_name = f"{file_path.stem}__{name.replace('/', '__')}.png"
                            preview_name = preview_name.replace(":", "_")
                            preview_path = assets_dir / preview_name
                            title = f"{file_path.name}: /{name}"
                            generated = preview_plot(sample, preview_path, title)
                            if generated:
                                ds_info.preview_path = f"assets/{generated}"
                            if obj.size > MAX_SAMPLE_ELEMENTS:
                                ds_info.note = "Preview/statistics use a bounded sample, not the full dataset."
                        elif ds_info.kind == "text":
                            ds_info.sample_text = text_preview(obj)
                        else:
                            ds_info.note = f"No preview for dataset kind {ds_info.kind}."

                        info.datasets.append(ds_info)
                except Exception as exc:
                    info.errors.append(f"Error visiting {name}: {exc}")

            f.visititems(visitor)
    except Exception as exc:
        info.errors.append(f"Failed to read file: {exc}")
    return info


def html_escape(x: Any) -> str:
    return html.escape(str(x))


def render_attrs(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return "<em>None</em>"
    items = []
    for k, v in attrs.items():
        items.append(f"<li><code>{html_escape(k)}</code>: {html_escape(v)}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def render_stats(stats: Dict[str, Any]) -> str:
    if not stats:
        return "<em>n/a</em>"
    parts = []
    for k, v in stats.items():
        parts.append(f"<li><code>{html_escape(k)}</code>: {html_escape(v)}</li>")
    return "<ul>" + "".join(parts) + "</ul>"


def render_tree(groups: List[GroupInfo], datasets: List[DatasetInfo]) -> str:
    lines: List[str] = []
    group_paths = {g.path for g in groups}
    if "/" not in group_paths:
        group_paths.add("/")

    all_nodes: Dict[str, Dict[str, List[str]]] = {}
    for path in sorted(group_paths):
        all_nodes.setdefault(path, {"groups": [], "datasets": []})
    for g in groups:
        if g.path == "/":
            continue
        parent = str(Path(g.path).parent).replace("\\", "/")
        if parent == ".":
            parent = "/"
        all_nodes.setdefault(parent, {"groups": [], "datasets": []})["groups"].append(g.path)
    for d in datasets:
        parent = str(Path(d.path).parent).replace("\\", "/")
        if parent == ".":
            parent = "/"
        all_nodes.setdefault(parent, {"groups": [], "datasets": []})["datasets"].append(d.path)

    ds_map = {d.path: d for d in datasets}

    def node_name(path: str) -> str:
        return "/" if path == "/" else path.rstrip("/").split("/")[-1]

    def render_node(path: str) -> str:
        children = all_nodes.get(path, {"groups": [], "datasets": []})
        grp_html = "".join(render_node(child) for child in sorted(children["groups"]))
        ds_html = "".join(
            f"<li><code>{html_escape(node_name(dp))}</code> <span class='muted'>shape={html_escape(format_shape(ds_map[dp].shape))}, dtype={html_escape(ds_map[dp].dtype)}</span></li>"
            for dp in sorted(children["datasets"])
        )
        title = html_escape(node_name(path))
        return f"<li><span class='group'>{title}</span><ul>{grp_html}{ds_html}</ul></li>"

    return "<ul class='tree'>" + render_node("/") + "</ul>"


def generate_report(files: List[FileInfo], out_dir: Path, root: Path) -> Path:
    rows = []
    for fi in files:
        rows.append(
            f"<tr><td><code>{html_escape(fi.rel_path)}</code></td>"
            f"<td>{len(fi.groups)}</td><td>{len(fi.datasets)}</td><td>{len(fi.errors)}</td></tr>"
        )

    sections = []
    for fi in files:
        dataset_blocks = []
        for ds in fi.datasets:
            preview_html = (
                f"<div class='preview'><img src='{html_escape(ds.preview_path)}' alt='preview'></div>"
                if ds.preview_path else ""
            )
            text_html = (
                f"<div><strong>Sample:</strong> <code>{html_escape(ds.sample_text)}</code></div>"
                if ds.sample_text else ""
            )
            note_html = f"<div class='note'>{html_escape(ds.note)}</div>" if ds.note else ""
            dataset_blocks.append(
                "<details class='dataset' open>"
                f"<summary><code>{html_escape(ds.path)}</code>"
                f" <span class='muted'>shape={html_escape(format_shape(ds.shape))}, dtype={html_escape(ds.dtype)}, kind={html_escape(ds.kind)}</span></summary>"
                f"<div class='dataset-body'><div><strong>Attributes</strong>{render_attrs(ds.attrs)}</div>"
                f"<div><strong>Statistics</strong>{render_stats(ds.stats)}</div>{text_html}{note_html}{preview_html}</div>"
                "</details>"
            )

        error_html = "".join(f"<li>{html_escape(err)}</li>" for err in fi.errors)
        sections.append(
            "<section class='file-section'>"
            f"<h2 id='{html_escape(fi.rel_path).replace('/', '_')}'>{html_escape(fi.rel_path)}</h2>"
            f"<p><strong>Full path:</strong> <code>{html_escape(fi.file_path)}</code></p>"
            f"<p><strong>Root attributes:</strong>{render_attrs(fi.attrs)}</p>"
            f"<div class='two-col'><div><h3>Tree</h3>{render_tree(fi.groups, fi.datasets)}</div>"
            f"<div><h3>Datasets</h3>{''.join(dataset_blocks) or '<em>No datasets found.</em>'}</div></div>"
            f"<div><h3>Errors</h3><ul>{error_html or '<li>None</li>'}</ul></div>"
            "</section>"
        )

    html_doc = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H5 Structure Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; line-height: 1.4; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f1f1; }}
    .muted {{ color: #666; font-size: 0.95em; }}
    .group {{ font-weight: 700; }}
    .tree ul {{ list-style: none; padding-left: 18px; }}
    .tree li {{ margin: 4px 0; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 2fr; gap: 24px; align-items: start; }}
    .dataset {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px 10px; margin-bottom: 12px; }}
    .dataset-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; }}
    .preview img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
    .file-section {{ margin: 40px 0; padding-top: 12px; border-top: 2px solid #eee; }}
    .note {{ margin-top: 8px; color: #7a5800; }}
    @media (max-width: 1100px) {{
      .two-col, .dataset-body {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>H5 Structure Report</h1>
  <p><strong>Scanned root:</strong> <code>{html_escape(root)}</code></p>
  <p>This report lists every discovered H5/NetCDF file, the groups and datasets inside it, shapes, dtypes, attributes, and lightweight previews for numeric arrays.</p>

  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>File</th><th>Groups</th><th>Datasets</th><th>Errors</th></tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="4">No H5 files found.</td></tr>'}
    </tbody>
  </table>

  {''.join(sections)}
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def find_h5_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


DEFAULT_ROOT = Path(r"E:\Software Engineering Stuff\Quantum\Majorana-MLQT-Project-EthanVu-CMPE188\manual-data")
DEFAULT_OUTPUT = DEFAULT_ROOT / "h5_visualizer_report"


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize H5 structures and dataset shapes.")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_ROOT),
        help=f"Root directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory where the HTML report and preview images will be written. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    files = find_h5_files(root)
    file_infos = [inspect_file(fp, root, assets_dir) for fp in files]
    report_path = generate_report(file_infos, out_dir, root)

    summary = {
        "root": str(root),
        "num_files": len(file_infos),
        "report": str(report_path),
        "files": [
            {
                "rel_path": fi.rel_path,
                "datasets": len(fi.datasets),
                "groups": len(fi.groups),
                "errors": fi.errors,
            }
            for fi in file_infos
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Scanned {len(file_infos)} files")
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
