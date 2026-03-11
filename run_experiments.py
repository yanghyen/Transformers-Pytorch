#!/usr/bin/env python3
"""
실험 그리드 자동 실행 스크립트.

Base: N=6, dmodel=512, dff=2048, h=8, dk=dv=64, Pdrop=0.1, εls=0.1, 100K steps
그리드:
  - 그룹1 (h, dk, dv): (1,512,512), (4,128,128), (16,32,32), (32,16,16)
  - 그룹2 (dk, dv, h=8): (16,64), (32,64)
  - 각 설정당 seed 3개

실행: python run_experiments.py [--trainer tensorboard|wandb] [--config BASE_CONFIG]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent


def get_base_config(config_path=None):
    """기본 설정 로드 (config.yaml과 병합)."""
    sys.path.insert(0, str(ROOT))
    from config.config import get_config
    path = config_path or ROOT / "config" / "config.yaml"
    if not path.exists():
        path = None
    return get_config(yaml_path=str(path) if path else None)


def run_id_from_config(run_config):
    """run_config에서 run_id 문자열 생성 (d_k, d_v 포함 가능)."""
    d = run_config.get("d_model", 512)
    h = run_config.get("n_head", 8)
    f = run_config.get("d_ff", 2048)
    l = run_config.get("layers", 6)
    s = run_config.get("seed", 42)
    base = f"d{d}_h{h}_f{f}_l{l}_s{s}"
    if "d_k" in run_config and "d_v" in run_config:
        return f"d{d}_h{h}_dk{run_config['d_k']}_dv{run_config['d_v']}_f{f}_l{l}_s{s}"
    return base


def main():
    parser = argparse.ArgumentParser(description="실험 그리드 자동 실행 (Base + h/dk/dv 변형)")
    parser.add_argument(
        "--trainer",
        type=str,
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="학습 스크립트: tensorboard (src.train) 또는 wandb (train_wb)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="베이스 설정 YAML 경로 (없으면 config/config.yaml 사용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 실행할 명령만 출력",
    )
    args = parser.parse_args()

    # Base: N=6, dmodel=512, dff=2048, h=8, dk=dv=64, Pdrop=0.1, εls=0.1, 100K steps
    # 그룹1: (h, dk, dv)
    group1 = [
        {"n_head": 8, "d_k": 64, "d_v": 64},
        {"n_head": 1, "d_k": 512, "d_v": 512},
        {"n_head": 4, "d_k": 128, "d_v": 128},
        {"n_head": 16, "d_k": 32, "d_v": 32},
        {"n_head": 32, "d_k": 16, "d_v": 16},
    ]
    # 그룹2: (dk, dv), h=8 유지
    group2 = [
        {"d_k": 16, "d_v": 64},
        {"d_k": 32, "d_v": 64},
    ]
    group3 = [
        {"layers": 2},
        {"layers": 4},
        {"layers": 8},
        {"d_model": 256, "d_k": 32, "d_v": 32},
        {"d_ff": 1024}, 
        {"d_ff": 4096}, 
        {"dropout": 0.0},
        {"dropout": 0.2},
        {"label_smoothing": 0.0},
        {"label_smoothing": 0.2}
    ]
    seeds = [42, 123, 456]  # 각 설정당 3개 seed


    out_dir = ROOT / "config" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.config) if args.config else ROOT / "config" / "config.yaml"
    base_config = get_base_config(base_path)

    # 표시되지 않은 값은 base와 동일 (이미 base_config에 포함)
    runs = []
    for override in group3:
        for seed in seeds:
            run_config = {
                **base_config,
                "layers": 6,
                "d_model": 512,
                "d_ff": 2048,
                "n_head": base_config.get("n_head", 8),
                "dropout": base_config.get("dropout", 0.1),
                "label_smoothing": base_config.get("label_smoothing", 0.1),
                "max_steps": base_config.get("max_steps", 100_000),
                "seed": seed,
                **override,
            }
            run_id = run_id_from_config(run_config)
            yaml_path = out_dir / f"run_{run_id}.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            runs.append((run_id, str(yaml_path)))

    if args.trainer == "wandb":
        module = "train_wb"
    else:
        module = "src.train"

    total = len(runs)
    for i, (run_id, yaml_path) in enumerate(runs, 1):
        cmd = [sys.executable, "-m", module, "--config", yaml_path]
        print(f"[{i}/{total}] {run_id}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            print(f"  실패 (exit code {ret.returncode})", file=sys.stderr)
            sys.exit(ret.returncode)

    if args.dry_run:
        print(f"\n총 {total}개 실험 (--dry-run 이므로 실행하지 않음)")
    else:
        print(f"\n총 {total}개 실험 완료.")


if __name__ == "__main__":
    main()
