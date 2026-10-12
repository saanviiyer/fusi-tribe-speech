"""Command-line entry point: `fusi-tribe <step>`. Run `fusi-tribe -h` for the steps."""

from __future__ import annotations

import argparse
import json

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fusi-tribe", description=__doc__)
    ap.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("fusi-glm", help="QC, registration, GLM and z maps for the fUSI run")
    sub.add_parser("fusi-timing", help="Data-driven response timing for the fUSI run")
    p = sub.add_parser("make-stimuli", help="Synthesise the in-silico runs (macOS `say`)")
    p.add_argument("--only", nargs="+", help="Only these conditions")
    p.add_argument("--force", action="store_true", help="Re-synthesise cached passage audio")
    p = sub.add_parser("run-tribe", help="TRIBE v2 predictions for each simulated run")
    p.add_argument("--mode", choices=["audio_only", "full"], default="audio_only")
    p.add_argument("--conditions", nargs="+")
    p = sub.add_parser("analyze-tribe", help="Retention, paired contrasts and timing from predictions")
    p.add_argument("--mode", choices=["audio_only", "full"], default="audio_only")
    sub.add_parser("report", help="Cross-modality figures")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.step == "fusi-glm":
        from .fusi.glm import run_glm
        print(json.dumps(run_glm(cfg), indent=2))
    elif args.step == "fusi-timing":
        from .fusi.timing import run_timing
        print(json.dumps(run_timing(cfg), indent=2))
    elif args.step == "make-stimuli":
        from .insilico.stimuli import make_stimuli
        make_stimuli(cfg, args.only, args.force)
    elif args.step == "run-tribe":
        from .insilico.tribe import run_tribe
        run_tribe(cfg, args.mode, args.conditions)
    elif args.step == "analyze-tribe":
        from .insilico.analysis import analyse_mode
        out = analyse_mode(cfg, args.mode)
        r = out["retention"]
        print(r[r.reliable].pivot(index="roi", columns="condition", values="retention").dropna().round(2))
    elif args.step == "report":
        from .report import mode_comparison, stimulus_spectrograms, timing_comparison
        stimulus_spectrograms(cfg)
        timing_comparison(cfg)
        if not mode_comparison(cfg):
            print("Full-mode results not found; skipped the audio-only vs full comparison")


if __name__ == "__main__":
    main()
