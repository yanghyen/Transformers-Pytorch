# PyTorch Transformer

**"Attention Is All You Need"** (Vaswani et al., 2017) 논문의 Transformer 모델을 PyTorch로 재구현한 프로젝트입니다.  
기본 설정은 논문의 Base 모델(d_model=512, 8 heads, 6 layers, 100K steps)을 따르며, WMT14 de-en 데이터로 기계 번역을 학습합니다.

## 환경 설정

- Python 3.9+
- PyTorch, `datasets`, `tokenizers`, `sacrebleu`, `wandb`(선택), `tensorboard`(선택) 등

```bash
pip install -r requirements.txt
```

## 데이터셋

- **데이터 소스**: [Hugging Face `wmt/wmt14`](https://huggingface.co/datasets/wmt/wmt14), **de-en** (독일어→영어) 구성 사용.
- **전처리**: BPE 토큰화 후 `data/tokenized/`에 캐시됩니다. 학습 시 이 캐시에서 token id만 로드합니다.

### 데이터셋만 미리 다운로드 (선택)

학습 없이 raw 데이터만 받아두려면:

```bash
python src/dataset/download_dataset.py
```

설정은 `config/config.py`의 기본값(`datasource`, `lang_src`, `lang_tgt`)을 따릅니다.  
**학습을 처음 실행하면** 데이터가 없을 경우 자동으로 다운로드 후 BPE 토큰화·캐시까지 한 번에 진행됩니다.

## 학습

설정은 `config/config.yaml`(및 `config/config.py` 기본값)에서 읽습니다.  
YAML에서 `d_model`, `n_head`, `layers`, `max_steps`, `batch_size` 등을 바꿀 수 있습니다.

### TensorBoard 로깅

```bash
python src/train.py --config config/config.yaml
```

로그는 `runs/<run_id>/tmodel`에 쌓이며, `tensorboard --logdir runs`로 확인할 수 있습니다.

### 그리드 실험 일괄 실행

여러 설정(헤드 수, d_k/d_v 등) × 시드에 대해 일괄 학습하려면:

```bash
python run_experiments.py
```

`--config`로 베이스 YAML을 지정할 수 있고, `--dry-run`으로 실제 실행 없이 명령만 볼 수 있습니다.

## 평가

학습 결과는 `runs/<run_id>/` 아래에 `checkpoints/`, `metrics/metrics.csv` 등으로 저장됩니다.

- **검증 PPL**: 학습 중 `metrics.csv`에 기록됩니다.
- **요약 및 BLEU**: `summarize_experiment.py`로 여러 run을 한 번에 요약·평가할 수 있습니다.

```bash
# 특정 실험 디렉터리 하위 run들에 대해 PPL / BLEU / 파라미터 수 요약
python summarize_experiment.py --runs-dir runs/experiment1 --out runs/experiment1_summary.csv

# BLEU 생략 (빠르게 PPL만)
python summarize_experiment.py --runs-dir runs/experiment1 --skip-bleu

# 디코딩: beam search (기본), beam size 등
python summarize_experiment.py --runs-dir runs/experiment1 --decode beam --beam-size 4
```

기본적으로 체크포인트를 사용하며, `--checkpoint-name`으로 파일명을 바꿀 수 있습니다.  
BLEU는 sacrebleu 기준으로 test 세트에 대해 계산됩니다.

## 디렉터리 구조 요약

- `config/` — 설정 (`config.yaml`, `config.py`, 실험용 YAML들)
- `src/` — 모델(`model`), 데이터셋·토크나이저(`dataset`), 학습 스크립트(`train.py`) 등
- `run_experiments.py` — 그리드 실험 실행
- `summarize_experiment.py` — run 요약 및 PPL/BLEU 평가
- `runs/` — 실험별 run 디렉터리 (체크포인트, 메트릭, TensorBoard 로그)
- `data/` — 토크나이저 JSON, 토큰화 캐시(`data/tokenized/`)

## 참고

- 논문: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
