import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FAMILIES = ["parity", "x_loop", "z_loop"]
MODELS = ["cthmm", "cnn_gru", "hsmm", "dmm"]
FAMILY_FILE_TOKENS = {
    "parity": ["parity"],
    "x_loop": ["x_loop", "x-loop", "xloop"],
    "z_loop": ["z_loop", "z-loop", "zloop"],
}


def discover_prepared(ready_dir: str | Path) -> list[str]:
    root = Path(ready_dir)
    if not root.exists():
        raise FileNotFoundError(f"Ready directory not found: {root}")
    files = sorted(root.glob("*.pt")) or sorted(root.glob("*.h5")) or sorted(root.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .pt/.h5/.hdf5 prepared bundles found in: {root}")
    return [str(p) for p in files]


def select_prepared_for_family(prepared: list[str], family: str, strict: bool) -> list[str]:
    if family == "all":
        return prepared
    tokens = FAMILY_FILE_TOKENS.get(family, [family])
    selected = []
    for fp in prepared:
        name = Path(fp).name.lower()
        stem = Path(fp).stem.lower()
        if any(tok in name or tok in stem for tok in tokens):
            selected.append(fp)
    if selected:
        return selected
    if strict:
        available = "\n".join(f"  - {p}" for p in prepared)
        raise FileNotFoundError(f"No prepared file matched family={family!r} using tokens={tokens}.\nAvailable files:\n{available}")
    # Fallback supports combined bundles that contain all families and rely on --family filtering.
    return prepared


def child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
    env.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.cpu_threads))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.cpu_threads))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env


def run(cmd: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def base_cmd(args: argparse.Namespace, prepared: list[str], stage_runs_dir: Path) -> list[str]:
    cmd = [args.python, args.trainer, "--prepared", *prepared, "--runs-dir", str(stage_runs_dir)]
    cmd += ["--num-workers", str(args.num_workers)]
    cmd += ["--prefetch-to-gpu"]
    if args.amp:
        cmd += ["--amp"]
    else:
        cmd += ["--no-amp"]
    if args.compile:
        cmd += ["--compile"]
    cmd += ["--target", args.target]
    cmd += ["--val-split", str(args.val_split), "--test-split", str(args.test_split)]
    cmd += ["--split-strategy", args.split_strategy]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    return cmd


def add_training_limits(cmd: list[str], args: argparse.Namespace, *, batch_size: int | None = None) -> list[str]:
    cmd += ["--epochs", str(args.epochs), "--batch-size", str(batch_size or args.batch_size)]
    if args.steps_per_epoch is not None:
        cmd += ["--steps-per-epoch", str(args.steps_per_epoch)]
    if args.full_epoch:
        cmd += ["--full-epoch"]
    return cmd


def build_command(args: argparse.Namespace, family: str, model: str, prepared: list[str], stage_runs_dir: Path) -> list[str]:
    cmd = base_cmd(args, prepared, stage_runs_dir) + [
        "--mode", model,
        "--family", family,
        "--training-scope", "together",
    ]
    if model == "cthmm":
        cmd += ["--cthmm-max-em-iters", str(args.cthmm_em_iters)]
        return cmd
    if model == "cnn_gru":
        cmd += ["--learning-task", args.learning_task, "--force-num-states", str(args.force_num_states)]
        return add_training_limits(cmd, args)
    if model == "hsmm":
        cmd += ["--hsmm-source", "handcrafted", "--hsmm-states", str(args.force_num_states)]
        return cmd
    if model == "dmm":
        cmd += ["--learning-task", args.learning_task, "--force-num-states", str(args.force_num_states)]
        return add_training_limits(cmd, args, batch_size=max(8, args.batch_size // 2))
    raise ValueError(f"Unsupported model: {model}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CT-HMM, CNN+GRU, HSMM, and DMM independently on their respective datasets.")
    ap.add_argument("--prepared", nargs="*", default=None, help="Prepared .pt/.h5 bundles. If omitted, --ready-dir is scanned.")
    ap.add_argument("--ready-dir", default="manual-data/~ready_torch")
    ap.add_argument("--runs-dir", default="training/runs_individual")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--trainer", default=str(Path(__file__).with_name("train_core.py")))
    ap.add_argument("--target", choices=["family", "run"], default="run")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--full-epoch", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--split-strategy", choices=["grouped", "random"], default="grouped")
    ap.add_argument("--cthmm-em-iters", type=int, default=25)
    ap.add_argument("--learning-task", choices=["self_supervised", "auto", "supervised", "multitask"], default="self_supervised")
    ap.add_argument("--force-num-states", type=int, default=2)
    ap.add_argument("--strict-family-files", action="store_true", help="Require separate prepared files whose names match each family.")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_prepared = list(args.prepared or [])
    if not all_prepared:
        all_prepared = discover_prepared(args.ready_dir)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    env = child_env(args)

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "individual_independent_training_no_teacher_sharing",
        "families": args.families,
        "models": args.models,
        "prepared": all_prepared,
        "runs_dir": str(runs_dir),
        "jobs": [],
    }
    jobs: list[dict[str, object]] = manifest["jobs"]  # type: ignore[assignment]

    for family in args.families:
        prepared = select_prepared_for_family(all_prepared, family, strict=args.strict_family_files)
        for model in args.models:
            stage_runs_dir = runs_dir / family / model
            stage_runs_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(args, family, model, prepared, stage_runs_dir)
            print("\n" + "=" * 100, flush=True)
            print(f"INDIVIDUAL TRAINING JOB: family={family} model={model}", flush=True)
            print("=" * 100, flush=True)
            status = "planned" if args.dry_run else "completed"
            error: str | None = None
            try:
                run(cmd, dry_run=args.dry_run, env=env)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                jobs.append({"family": family, "model": model, "status": status, "error": error, "command": cmd})
                (runs_dir / "individual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                if not args.continue_on_error:
                    raise
                continue
            jobs.append({"family": family, "model": model, "status": status, "error": error, "command": cmd})
            (runs_dir / "individual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    out = runs_dir / "individual_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote individual-training manifest: {out}", flush=True)


if __name__ == "__main__":
    main()
