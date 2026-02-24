from pathlib import Path
import random
import torch
import numpy as np

# YAML 로드 (선택)
def _load_yaml(path):
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("YAML 설정을 쓰려면 PyYAML이 필요합니다: pip install pyyaml")
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")


def get_config(yaml_path=None):
    """기본 설정을 반환합니다. yaml_path가 있으면 해당 YAML과 병합합니다."""
    # "Attention Is All You Need" Table 3 - Base model
    base = {
        "batch_size": 64,  # used only when tokens_per_batch is None (sentence-based batching) 128도 해보기
        "tokens_per_batch_src": 25000,   # ~25k source tokens per batch (paper)
        "tokens_per_batch_tgt": 25000,   # ~25k target tokens per batch (paper); None = use tokens_per_batch_src
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 100,
        "d_model": 512,
        "n_head": 8,
        "d_ff": 2048,
        "layers": 6,
        "dropout": 0.1,       # P_drop
        "label_smoothing": 0.1,  # ε_ls
        "max_steps": 100_000,     # 100K steps (paper base)
        "datasource": "wmt/wmt14",  # 이전: opus_books
        "lang_src": "de",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "data/tokenizer_{0}.json",
        "tokenizer_vocab_size": 37000,  # BPE vocab size (Attention Is All You Need)
        "dataset_cache_dir": "data/tokenized",  # BPE 토큰화 결과 저장; 학습 시 여기서 token id 로드
        "val_metrics_subset_size": 2000,  # 1000스텝마다 검증 메트릭 계산 시 사용할 샘플 수 (None이면 전체)
        "experiment_name": "runs/tmodel",
        "seed": 42,
    }
    if yaml_path:
        overrides = _load_yaml(yaml_path)
        base.update(overrides)  # null도 반영 (동적 배치 끄기 등)
    return base


def get_run_id(config):
    """d_model, n_head, d_ff, layers, seed로 실행 식별자 문자열을 만듭니다."""
    d = config.get("d_model", 512)
    h = config.get("n_head", 8)
    f = config.get("d_ff", 2048)
    l = config.get("layers", 6)
    s = config.get("seed", 42)
    return f"d{d}_h{h}_f{f}_l{l}_s{s}"


def get_run_dir(config):
    """runs/run_id 루트 경로를 반환합니다. tmodel, metrics, checkpoints가 이 아래에 생성됩니다."""
    return Path("runs") / get_run_id(config)


def ensure_run_dirs(config):
    """runs/run_id/tmodel, runs/run_id/metrics, runs/run_id/checkpoints 디렉터리를 생성합니다."""
    run_dir = get_run_dir(config)
    (run_dir / "tmodel").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


def get_weights_file_path(config, epoch: str):
    run_dir = get_run_dir(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(run_dir / "checkpoints" / model_filename)


def latest_weights_file_path(config):
    run_dir = get_run_dir(config)
    checkpoints_dir = run_dir / "checkpoints"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(checkpoints_dir.glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def get_metrics_path(config):
    """runs/run_id/metrics/metrics.csv 경로를 반환합니다."""
    return get_run_dir(config) / "metrics" / "metrics.csv"


def get_metrics_10k_path(config):
    """runs/run_id/metrics/metrics_10k.csv 경로 (10k 스텝마다 전체 val set 메트릭)."""
    return get_run_dir(config) / "metrics" / "metrics_10k.csv"


def get_tensorboard_dir(config):
    """runs/run_id/tmodel 경로를 반환합니다."""
    return get_run_dir(config) / "tmodel"


def seed_everything(seed: int):
    """재현을 위해 random, numpy, torch 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = True
