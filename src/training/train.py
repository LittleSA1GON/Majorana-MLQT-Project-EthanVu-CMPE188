import argparse
import os
import subprocess
import sys
from pathlib import Path


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


def maybe_add(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run(cmd: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def common_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.prepared:
        out += ["--prepared", *args.prepared]
    else:
        out += ["--ready-dir", args.ready_dir]
    out += ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)]
    out += ["--num-workers", str(args.num_workers)]
    maybe_add(out, "--max-samples", args.max_samples)
    maybe_add(out, "--steps-per-epoch", args.steps_per_epoch)
    if args.full_epoch:
        out.append("--full-epoch")
    if args.amp:
        out.append("--amp")
    if args.compile:
        out.append("--compile")
    out += ["--cthmm-em-iters", str(args.cthmm_em_iters)]
    if args.dry_run:
        out.append("--dry-run")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run individual QCVV baselines and/or the combined QCVV pipeline.")
    ap.add_argument("--only", choices=["both", "individual", "pipeline"], default="both")
    ap.add_argument("--prepared", nargs="*", default=None)
    ap.add_argument("--ready-dir", default="manual-data/~ready_torch")
    ap.add_argument("--runs-dir", default="training/runs")
    ap.add_argument("--families", nargs="+", default=["parity", "x_loop", "z_loop"], choices=["parity", "x_loop", "z_loop"])
    ap.add_argument("--models", nargs="+", default=["cthmm", "cnn_gru", "hsmm", "dmm"], choices=["cthmm", "cnn_gru", "hsmm", "dmm"], help="Models for the individual phase.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--individual-target", choices=["family", "run"], default="run")
    ap.add_argument("--pipeline-target", choices=["family", "run"], default="family")
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
    ap.add_argument("--cthmm-em-iters", type=int, default=25)
    ap.add_argument("--teacher-policy", choices=["auto", "uniform", "weak", "off"], default="auto", help="Pipeline-only. auto tunes teacher use per family so x_loop/z_loop are not over-constrained by CT-HMM.")
    ap.add_argument("--teacher-weight", type=float, default=0.05, help="Pipeline-only global teacher weight used by --teacher-policy uniform/weak.")
    ap.add_argument("--teacher-min-confidence", type=float, default=0.80, help="Pipeline-only global teacher threshold used by --teacher-policy uniform/weak.")
    ap.add_argument("--x-loop-teacher-weight", dest="x_loop_teacher_weight", type=float, default=None)
    ap.add_argument("--x-loop-teacher-min-confidence", dest="x_loop_teacher_min_confidence", type=float, default=None)
    ap.add_argument("--z-loop-teacher-weight", dest="z_loop_teacher_weight", type=float, default=None)
    ap.add_argument("--z-loop-teacher-min-confidence", dest="z_loop_teacher_min_confidence", type=float, default=None)
    ap.add_argument("--hsmm-cnn-source", choices=["auto", "independent", "teacher"], default="auto")
    ap.add_argument("--dmm-preset", choices=["auto", "standard", "small"], default="auto")
    ap.add_argument("--skip-individual", action="store_true", help="Deprecated alias for --only pipeline.")
    ap.add_argument("--skip-pipeline", action="store_true", help="Deprecated alias for --only individual.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.skip_individual:
        args.only = "pipeline"
    if args.skip_pipeline:
        args.only = "individual"

    root = Path(__file__).resolve().parent
    runs_root = Path(args.runs_dir)
    env = child_env(args)

    if args.only in {"both", "individual"}:
        cmd = [args.python, str(root / "train_individual.py"), *common_args(args)]
        cmd += ["--runs-dir", str(runs_root / "individual")]
        cmd += ["--families", *args.families]
        cmd += ["--models", *args.models]
        cmd += ["--target", args.individual_target]
        run(cmd, dry_run=args.dry_run, env=env)

    if args.only in {"both", "pipeline"}:
        cmd = [args.python, str(root / "train_pipeline.py"), *common_args(args)]
        cmd += ["--runs-dir", str(runs_root / "pipeline")]
        cmd += ["--families", *args.families]
        cmd += ["--target", args.pipeline_target]
        cmd += ["--teacher-policy", args.teacher_policy]
        cmd += ["--teacher-weight", str(args.teacher_weight), "--teacher-min-confidence", str(args.teacher_min_confidence)]
        cmd += ["--hsmm-cnn-source", args.hsmm_cnn_source, "--dmm-preset", args.dmm_preset]
        for flag, value in (
            ("--x-loop-teacher-weight", args.x_loop_teacher_weight),
            ("--x-loop-teacher-min-confidence", args.x_loop_teacher_min_confidence),
            ("--z-loop-teacher-weight", args.z_loop_teacher_weight),
            ("--z-loop-teacher-min-confidence", args.z_loop_teacher_min_confidence),
        ):
            if value is not None:
                cmd += [flag, str(value)]
        run(cmd, dry_run=args.dry_run, env=env)


if __name__ == "__main__":
    main()
