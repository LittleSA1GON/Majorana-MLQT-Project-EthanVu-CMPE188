import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

FAMILIES = ["x_loop", "z_loop"]
FAMILY_FILE_TOKENS = {
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



def select_prepared_for_family(prepared: list[str], family: str, strict: bool = False) -> list[str]:
    tokens = FAMILY_FILE_TOKENS.get(family, [family])
    selected = [p for p in prepared if any(tok in Path(p).name.lower() or tok in Path(p).stem.lower() for tok in tokens)]
    if selected:
        return selected
    if strict:
        available = "\n".join(f"  - {p}" for p in prepared)
        raise FileNotFoundError(f"No prepared file matched family={family!r} using tokens={tokens}.\nAvailable files:\n{available}")
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


def newest_run(root: Path, contains: str) -> Path | None:
    if not root.exists():
        return None
    matches = [p for p in root.rglob(f"*{contains}*") if p.is_dir()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def maybe_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    for name in ("best.pt", "last.pt", "neural_cthmm_best.pt", "neural_cthmm_last.pt"):
        p = run_dir / name
        if p.exists():
            return p
    return None


AUTO_FAMILY_POLICY = {
    "parity": {
        "teacher_weight": 0.05,
        "teacher_min_confidence": 0.80,
        "hsmm_cnn_source": "teacher",
        "dmm_latent_dim": None,
        "dmm_conv_channels": None,
        "dmm_encoder_hidden": None,
        "dmm_teacher": True,
    },
    "x_loop": {
        "teacher_weight": 0.005,
        "teacher_min_confidence": 0.55,
        "hsmm_cnn_source": "independent",
        "dmm_latent_dim": 12,
        "dmm_conv_channels": 24,
        "dmm_encoder_hidden": 32,
        "dmm_teacher": False,
    },
    "z_loop": {
        "teacher_weight": 0.01,
        "teacher_min_confidence": 0.70,
        "hsmm_cnn_source": "independent",
        "dmm_latent_dim": 12,
        "dmm_conv_channels": 24,
        "dmm_encoder_hidden": 32,
        "dmm_teacher": True,
    },
}


def family_policy(args: argparse.Namespace, family: str) -> dict[str, object]:
    base = dict(AUTO_FAMILY_POLICY.get(family, AUTO_FAMILY_POLICY["parity"]))
    if args.teacher_policy == "uniform":
        base["teacher_weight"] = args.teacher_weight
        base["teacher_min_confidence"] = args.teacher_min_confidence
        base["dmm_teacher"] = True
    elif args.teacher_policy == "off":
        base["teacher_weight"] = 0.0
        base["teacher_min_confidence"] = 1.01
        base["dmm_teacher"] = False
    elif args.teacher_policy == "weak":
        base["teacher_weight"] = min(float(args.teacher_weight), 0.01)
        base["teacher_min_confidence"] = min(float(args.teacher_min_confidence), 0.70)
        base["dmm_teacher"] = family != "x_loop"
    elif args.teacher_policy != "auto":
        raise ValueError(f"Unsupported teacher policy: {args.teacher_policy}")

    # Explicit per-family CLI overrides win over policy defaults.
    override_prefix = family.replace("_", "_")
    weight_override = getattr(args, f"{override_prefix}_teacher_weight", None)
    conf_override = getattr(args, f"{override_prefix}_teacher_min_confidence", None)
    if weight_override is not None:
        base["teacher_weight"] = float(weight_override)
    if conf_override is not None:
        base["teacher_min_confidence"] = float(conf_override)

    if args.hsmm_cnn_source != "auto":
        base["hsmm_cnn_source"] = args.hsmm_cnn_source

    if args.dmm_preset == "standard":
        base["dmm_latent_dim"] = None
        base["dmm_conv_channels"] = None
        base["dmm_encoder_hidden"] = None
    elif args.dmm_preset == "small":
        base["dmm_latent_dim"] = 12
        base["dmm_conv_channels"] = 24
        base["dmm_encoder_hidden"] = 32
    elif args.dmm_preset != "auto":
        raise ValueError(f"Unsupported DMM preset: {args.dmm_preset}")
    return base


def add_dmm_policy_args(cmd: list[str], policy: dict[str, object]) -> list[str]:
    for flag, key in (
        ("--dmm-latent-dim", "dmm_latent_dim"),
        ("--dmm-conv-channels", "dmm_conv_channels"),
        ("--dmm-encoder-hidden", "dmm_encoder_hidden"),
    ):
        value = policy.get(key)
        if value is not None:
            cmd += [flag, str(value)]
    return cmd


def add_common_limits(cmd: list[str], args: argparse.Namespace, *, batch_size: int | None = None) -> list[str]:
    cmd += ["--epochs", str(args.epochs), "--batch-size", str(batch_size or args.batch_size)]
    if args.steps_per_epoch is not None:
        cmd += ["--steps-per-epoch", str(args.steps_per_epoch)]
    if args.full_epoch:
        cmd += ["--full-epoch"]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]
    return cmd


def base_cmd(args: argparse.Namespace, stage_runs_dir: Path) -> list[str]:
    cmd = [args.python, args.trainer, "--prepared", *args.prepared, "--runs-dir", str(stage_runs_dir)]
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
    return cmd


def export_prediction(
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    family: str,
    stage_runs_dir: Path,
    source_run: Path | None,
    stage_name: str,
    checkpoint: Path | None,
    split_manifest: Path | None,
) -> Path | None:
    if source_run is None:
        return None
    if args.dry_run and checkpoint is None:
        checkpoint = source_run / "DRYRUN_best.pt"
    if checkpoint is None or (not args.dry_run and not checkpoint.exists()):
        return None
    out_path = source_run / f"predictions_{stage_name}.npz" if not args.dry_run else stage_runs_dir / f"DRYRUN_{family}_{stage_name}.npz"
    cmd = base_cmd(args, stage_runs_dir) + [
        "--mode", "export",
        "--family", family,
        "--training-scope", "together",
        "--checkpoint", str(checkpoint),
        "--export-out", str(out_path),
    ]
    if split_manifest is not None and (args.dry_run or split_manifest.exists()):
        cmd += ["--split-manifest", str(split_manifest)]
    run(cmd, dry_run=args.dry_run, env=env)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the combined CT-HMM -> CNN -> teacher CNN -> HSMM -> DMM -> export QCVV pipeline.")
    ap.add_argument("--prepared", nargs="*", default=None, help="Prepared .pt/.h5 bundles. If omitted, --ready-dir is scanned.")
    ap.add_argument("--ready-dir", default="manual-data/~ready_torch")
    ap.add_argument("--runs-dir", default="training/runs_pipeline")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--trainer", default=str(Path(__file__).with_name("train_core.py")))
    ap.add_argument("--target", choices=["family", "run"], default="family")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--full-epoch", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--cuda-visible-devices", default=None)
    ap.add_argument("--strict-family-files", action="store_true", help="Require a prepared filename match for each family rather than falling back to combined bundles.")
    ap.add_argument("--amp", action="store_true", help="Enable AMP. Leave off for GTX 1080 Ti unless tested stable.")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.10)
    ap.add_argument("--split-strategy", choices=["grouped", "random"], default="grouped")
    ap.add_argument("--cthmm-em-iters", type=int, default=25)
    ap.add_argument("--teacher-weight", type=float, default=0.05)
    ap.add_argument("--teacher-min-confidence", type=float, default=0.80)
    ap.add_argument("--teacher-temperature", type=float, default=1.0)
    ap.add_argument("--teacher-policy", choices=["auto", "uniform", "weak", "off"], default="auto", help="auto applies safer per-family teacher settings for x_loop/z_loop; uniform uses the same global teacher settings for every family; off disables distillation.")
    ap.add_argument("--parity-teacher-weight", type=float, default=None)
    ap.add_argument("--parity-teacher-min-confidence", type=float, default=None)
    ap.add_argument("--x-loop-teacher-weight", dest="x_loop_teacher_weight", type=float, default=None)
    ap.add_argument("--x-loop-teacher-min-confidence", dest="x_loop_teacher_min_confidence", type=float, default=None)
    ap.add_argument("--z-loop-teacher-weight", dest="z_loop_teacher_weight", type=float, default=None)
    ap.add_argument("--z-loop-teacher-min-confidence", dest="z_loop_teacher_min_confidence", type=float, default=None)
    ap.add_argument("--hsmm-cnn-source", choices=["auto", "independent", "teacher"], default="auto", help="auto uses teacher CNN for parity and independent CNN for x_loop/z_loop.")
    ap.add_argument("--dmm-preset", choices=["auto", "standard", "small"], default="auto", help="auto uses smaller DMM settings for x_loop/z_loop to improve stability.")
    ap.add_argument("--skip-hsmm", action="store_true")
    ap.add_argument("--skip-dmm", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--compare", action="store_true", help="Optional: run compare mode after export. Not part of the default requested 6-stage pipeline.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.prepared = list(args.prepared or [])
    if not args.prepared:
        args.prepared = discover_prepared(args.ready_dir)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    env = child_env(args)

    manifest: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "combined_qcvv_pipeline",
        "pipeline_order": [
            "cthmm",
            "cnn_gru_independent",
            "cnn_gru_teacher_assisted",
            "hsmm_using_cnn_embeddings",
            "dmm",
            "export_predictions",
        ],
        "prepared": args.prepared,
        "families": args.families,
        "runs_dir": str(runs_dir),
        "stages": [],
    }
    stages: list[dict[str, object]] = manifest["stages"]  # type: ignore[assignment]

    for family in args.families:
        print("\n" + "=" * 100, flush=True)
        print(f"COMBINED PIPELINE FAMILY: {family}", flush=True)
        print("=" * 100, flush=True)

        family_root = runs_dir / family
        family_root.mkdir(parents=True, exist_ok=True)
        policy = family_policy(args, family)
        family_prepared = select_prepared_for_family(args.prepared, family, strict=args.strict_family_files)
        family_args = argparse.Namespace(**vars(args))
        family_args.prepared = family_prepared
        print(f"Family policy for {family}: {policy}", flush=True)
        print("Prepared files for this family:", *family_prepared, sep="\n  - ", flush=True)
        stages.append({"family": family, "stage": "family_policy", "policy": policy, "prepared": family_prepared})

        # 1. CT-HMM teacher/baseline.
        stage_root = family_root / "01_cthmm"
        cmd = base_cmd(family_args, stage_root) + [
            "--mode", "cthmm",
            "--family", family,
            "--training-scope", "together",
            "--cthmm-max-em-iters", str(args.cthmm_em_iters),
        ]
        run(cmd, dry_run=args.dry_run, env=env)
        cthmm_run = newest_run(stage_root, f"cthmm_together_{family}")
        if args.dry_run and cthmm_run is None:
            cthmm_run = stage_root / f"DRYRUN_cthmm_together_{family}_family"
        teacher_npz = cthmm_run / "teacher_predictions.npz" if cthmm_run else None
        split_manifest = cthmm_run / "split_manifest.json" if cthmm_run else None
        stages.append({"family": family, "stage": "1_cthmm", "run_dir": str(cthmm_run), "teacher_npz": str(teacher_npz) if teacher_npz else None})

        # 2. Independent CNN+GRU baseline.
        stage_root = family_root / "02_cnn_gru_independent"
        cmd = base_cmd(family_args, stage_root) + [
            "--mode", "cnn_gru",
            "--family", family,
            "--training-scope", "together",
            "--learning-task", "self_supervised",
            "--force-num-states", "2",
        ]
        if split_manifest and (args.dry_run or split_manifest.exists()):
            cmd += ["--split-manifest", str(split_manifest)]
        cmd = add_common_limits(cmd, args)
        run(cmd, dry_run=args.dry_run, env=env)
        cnn_ind_run = newest_run(stage_root, f"cnn_gru_together_{family}")
        if args.dry_run and cnn_ind_run is None:
            cnn_ind_run = stage_root / f"DRYRUN_cnn_gru_together_{family}_family"
        stages.append({"family": family, "stage": "2_cnn_gru_independent", "run_dir": str(cnn_ind_run)})

        # Avoid timestamp collisions in the core trainer on very fast dry/small runs.
        if not args.dry_run:
            time.sleep(1.05)

        # 3. Teacher-assisted CNN+GRU.
        stage_root = family_root / "03_cnn_gru_teacher_assisted"
        cmd = base_cmd(family_args, stage_root) + [
            "--mode", "cnn_gru",
            "--family", family,
            "--training-scope", "together",
            "--learning-task", "self_supervised",
            "--force-num-states", "2",
            "--teacher-weight", str(policy["teacher_weight"]),
            "--teacher-min-confidence", str(policy["teacher_min_confidence"]),
            "--teacher-temperature", str(args.teacher_temperature),
        ]
        if teacher_npz and (args.dry_run or teacher_npz.exists()):
            cmd += ["--teacher-npz", str(teacher_npz)]
        if split_manifest and (args.dry_run or split_manifest.exists()):
            cmd += ["--split-manifest", str(split_manifest)]
        cmd = add_common_limits(cmd, args)
        run(cmd, dry_run=args.dry_run, env=env)
        cnn_teacher_run = newest_run(stage_root, f"cnn_gru_together_{family}")
        if args.dry_run and cnn_teacher_run is None:
            cnn_teacher_run = stage_root / f"DRYRUN_cnn_gru_teacher_together_{family}_family"
        stages.append({"family": family, "stage": "3_cnn_gru_teacher_assisted", "run_dir": str(cnn_teacher_run)})

        # 4. HSMM using CNN embeddings.
        hsmm_run = None
        if not args.skip_hsmm:
            stage_root = family_root / "04_hsmm_using_cnn_embeddings"
            if policy.get("hsmm_cnn_source") == "independent":
                cnn_ckpt = maybe_checkpoint(cnn_ind_run) or maybe_checkpoint(cnn_teacher_run)
            else:
                cnn_ckpt = maybe_checkpoint(cnn_teacher_run) or maybe_checkpoint(cnn_ind_run)
            cmd = base_cmd(family_args, stage_root) + [
                "--mode", "hsmm",
                "--family", family,
                "--training-scope", "together",
                "--hsmm-states", "2",
            ]
            if cnn_ckpt or args.dry_run:
                cmd += ["--hsmm-source", "cnn", "--cnn-checkpoint", str(cnn_ckpt or f"DRYRUN_{family}_cnn_best.pt")]
            else:
                cmd += ["--hsmm-source", "handcrafted"]
            if split_manifest and (args.dry_run or split_manifest.exists()):
                cmd += ["--split-manifest", str(split_manifest)]
            run(cmd, dry_run=args.dry_run, env=env)
            hsmm_run = newest_run(stage_root, f"hsmm_together_{family}")
            if args.dry_run and hsmm_run is None:
                hsmm_run = stage_root / f"DRYRUN_hsmm_together_{family}_family"
            stages.append({"family": family, "stage": "4_hsmm_using_cnn_embeddings", "run_dir": str(hsmm_run)})

        # 5. DMM.
        dmm_run = None
        if not args.skip_dmm:
            stage_root = family_root / "05_dmm"
            cmd = base_cmd(family_args, stage_root) + [
                "--mode", "dmm",
                "--family", family,
                "--training-scope", "together",
                "--learning-task", "self_supervised",
                "--force-num-states", "2",
            ]
            if bool(policy.get("dmm_teacher", True)) and teacher_npz and (args.dry_run or teacher_npz.exists()):
                cmd += [
                    "--teacher-npz", str(teacher_npz),
                    "--teacher-weight", str(policy["teacher_weight"]),
                    "--teacher-min-confidence", str(policy["teacher_min_confidence"]),
                    "--teacher-temperature", str(args.teacher_temperature),
                ]
            if split_manifest and (args.dry_run or split_manifest.exists()):
                cmd += ["--split-manifest", str(split_manifest)]
            cmd = add_dmm_policy_args(cmd, policy)
            cmd = add_common_limits(cmd, args, batch_size=max(8, args.batch_size // 2))
            run(cmd, dry_run=args.dry_run, env=env)
            dmm_run = newest_run(stage_root, f"dmm_together_{family}")
            if args.dry_run and dmm_run is None:
                dmm_run = stage_root / f"DRYRUN_dmm_together_{family}_family"
            stages.append({"family": family, "stage": "5_dmm", "run_dir": str(dmm_run)})

        # 6. Export predictions.
        exported: list[str] = []
        if teacher_npz and (args.dry_run or teacher_npz.exists()):
            exported.append(str(teacher_npz))
        if not args.skip_export:
            stage_root = family_root / "06_export_predictions"
            for source_run, name in ((cnn_ind_run, "cnn_independent"), (cnn_teacher_run, "cnn_teacher_assisted"), (dmm_run, "dmm")):
                pred = export_prediction(
                    args=family_args,
                    env=env,
                    family=family,
                    stage_runs_dir=stage_root,
                    source_run=source_run,
                    stage_name=name,
                    checkpoint=maybe_checkpoint(source_run),
                    split_manifest=split_manifest,
                )
                if pred:
                    exported.append(str(pred))
            stages.append({"family": family, "stage": "6_export_predictions", "prediction_npz": exported})

        if args.compare and len(exported) >= 2:
            stage_root = family_root / "07_optional_compare"
            cmd = base_cmd(family_args, stage_root) + [
                "--mode", "compare",
                "--family", family,
                "--training-scope", "together",
                "--prediction-npz", *exported,
            ]
            run(cmd, dry_run=args.dry_run, env=env)
            cmp_run = newest_run(stage_root, f"compare_together_{family}")
            stages.append({"family": family, "stage": "optional_compare", "run_dir": str(cmp_run)})

    out = runs_dir / "pipeline_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote pipeline manifest: {out}", flush=True)


if __name__ == "__main__":
    main()
