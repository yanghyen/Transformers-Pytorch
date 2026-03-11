#!/usr/bin/env python3
"""
Summarize runs under a directory (e.g. runs/experiment1):
  - PPL: from metrics/metrics.csv (best or last val_ppl)
  - (optional) Recomputed PPL: run checkpoint on cached *val* split and compute token-level NLL/PPL
  - Params: from checkpoint by building model and counting parameters (in millions)
  - BLEU: run decoding on cached test split and compute sacrebleu corpus BLEU

Examples:
  python summarize_experiment.py --runs-dir runs/experiment1 --out runs/experiment1_summary.csv
  python summarize_experiment.py --runs-dir runs/experiment1 --bleu-max-samples 2000 --decode beam --beam-size 4
  python summarize_experiment.py --runs-dir runs/experiment1 --skip-bleu
  python summarize_experiment.py --runs-dir runs/experiment1 --recompute-ppl
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# PyTorch 2.0.x does not support expandable_segments (added in 2.1+)
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import torch


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        s = str(x).strip()
        if s == "":
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _read_metrics_val_ppl(metrics_csv: Path) -> list[dict[str, Any]]:
    if not metrics_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            rows.append(r)
    return rows


def _pick_val_ppl(rows: list[dict[str, Any]], mode: str) -> tuple[float | None, int | None]:
    # returns (val_ppl, step)
    if not rows:
        return None, None
    parsed: list[tuple[int, float]] = []
    for r in rows:
        step = r.get("step")
        val_ppl = r.get("val_ppl")
        s = None
        try:
            s = int(str(step).strip())
        except Exception:
            s = None
        v = _safe_float(val_ppl)
        if s is None or v is None:
            continue
        parsed.append((s, v))
    if not parsed:
        return None, None
    if mode == "best":
        s, v = min(parsed, key=lambda t: t[1])
        return v, s
    if mode == "last":
        s, v = max(parsed, key=lambda t: t[0])
        return v, s
    raise ValueError(f"Unknown ppl mode: {mode}")


def _pick_checkpoint(run_dir: Path, preferred: str | None = "tmodel_100k.pt") -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    if preferred:
        p = ckpt_dir / preferred
        if p.exists():
            return p
    pts = sorted(ckpt_dir.glob("*.pt"))
    if not pts:
        return None

    # Prefer numeric/epoch-like suffixes, then newest mtime.
    def _score(path: Path) -> tuple[int, int]:
        name = path.stem  # e.g. tmodel_02
        num = -1
        if "_" in name:
            tail = name.split("_")[-1]
            if tail.isdigit():
                num = int(tail)
            elif tail.lower().endswith("k") and tail[:-1].isdigit():
                num = int(tail[:-1]) * 1000
        return (num, int(path.stat().st_mtime))

    pts.sort(key=_score)
    return pts[-1]


def _device_from_arg(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _strip_special_ids(ids: list[int], tokenizer_tgt) -> list[int]:
    special = set()
    for tok in ("[SOS]", "[EOS]", "[PAD]"):
        try:
            tid = tokenizer_tgt.token_to_id(tok)
            if tid is not None:
                special.add(int(tid))
        except Exception:
            pass
    return [i for i in ids if int(i) not in special]


def beam_search_decode(
    model,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    max_len: int,
    device: torch.device,
    beam_size: int = 4,
    length_penalty_alpha: float = 0.6,
) -> torch.Tensor:
    import torch.nn.functional as F
    from src.dataset.dataset import causal_mask

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    if source.dim() == 1:
        source = source.unsqueeze(0)

    encoder_output = model.encode(source, source_mask)
    decoder_initial = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    candidates: list[tuple[torch.Tensor, float, int]] = [(decoder_initial, 0.0, 1)]

    while True:
        if any(cand.size(1) >= max_len for cand, _, _ in candidates):
            break
        new_candidates: list[tuple[torch.Tensor, float, int]] = []
        for candidate, sum_log_prob, length in candidates:
            if candidate[0, -1].item() == eos_idx:
                new_candidates.append((candidate, sum_log_prob, length))
                continue
            cand_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device).int()
            out = model.decode(encoder_output, source_mask, candidate, cand_mask)
            logits = model.project(out[:, -1])
            log_prob = F.log_softmax(logits, dim=-1)
            topk_log_prob, topk_idx = torch.topk(log_prob, beam_size, dim=1)
            for i in range(beam_size):
                token = topk_idx[0, i].view(1, 1)
                token_log_prob = topk_log_prob[0, i].item()
                new_cand = torch.cat([candidate, token], dim=1)
                new_len = length + 1
                new_candidates.append((new_cand, sum_log_prob + token_log_prob, new_len))

        def normalized_score(item: tuple[torch.Tensor, float, int]) -> float:
            _, sum_lp, length = item
            lp = ((5.0 + length) / 6.0) ** length_penalty_alpha if length > 0 else 1.0
            return sum_lp / lp

        candidates = sorted(new_candidates, key=normalized_score, reverse=True)[:beam_size]
        if all(cand[0, -1].item() == eos_idx for cand, _, _ in candidates):
            break
    return candidates[0][0].squeeze(0)


def greedy_decode(
    model,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    from src.dataset.dataset import causal_mask

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    if source.dim() == 1:
        source = source.unsqueeze(0)
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        next_word = prob.argmax(dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


@dataclass
class RunSummary:
    run: str
    seed: int | None
    d_model: int | None
    n_head: int | None
    d_ff: int | None
    layers: int | None
    d_k: int | None
    d_v: int | None
    ppl: float | None
    ppl_step: int | None
    bleu: float | None
    flops_g: float | None
    params_m: float | None
    ckpt: str | None
    metrics_csv: str | None
    error: str | None


def _int_or_none(x: Any) -> int | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(s)
    except Exception:
        return None


def _load_ckpt_and_build_model(ckpt_path: Path, device: torch.device):
    from src.model import build_transformer

    state = torch.load(str(ckpt_path), map_location="cpu")
    cfg = state.get("config", {}) or {}
    sd = state.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint missing model_state_dict: {ckpt_path}")
    if "src_embed.embedding.weight" not in sd or "tgt_embed.embedding.weight" not in sd:
        raise ValueError(f"Checkpoint missing embedding weights: {ckpt_path}")
    vocab_src = int(sd["src_embed.embedding.weight"].shape[0])
    vocab_tgt = int(sd["tgt_embed.embedding.weight"].shape[0])
    seq_len = int(cfg.get("seq_len", 100))
    model = build_transformer(
        vocab_src,
        vocab_tgt,
        seq_len,
        seq_len,
        d_model=int(cfg.get("d_model", 512)),
        N=int(cfg.get("layers", 6)),
        h=int(cfg.get("n_head", 8)),
        d_ff=int(cfg.get("d_ff", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        d_k=cfg.get("d_k"),
        d_v=cfg.get("d_v"),
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return state, cfg, model, vocab_src, vocab_tgt


def _count_params_m(model) -> float:
    n = 0
    for p in model.parameters():
        n += p.numel()
    return n / 1e6


def _compute_bleu_for_run(
    model,
    device: torch.device,
    test_ds,
    tokenizer_tgt,
    max_len: int,
    bleu_max_samples: int | None,
    decode: str,
    beam_size: int,
    length_penalty_alpha: float,
    progress_every: int | None = None,
) -> float:
    import sacrebleu
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    preds: list[str] = []
    refs: list[str] = []

    n = 0
    with torch.no_grad():
        for batch in test_loader:
            if bleu_max_samples is not None and n >= bleu_max_samples:
                break
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            tgt_text = batch["tgt_text"][0] if isinstance(batch["tgt_text"], (list, tuple)) else batch["tgt_text"]

            if decode == "beam":
                out_ids = beam_search_decode(
                    model,
                    encoder_input,
                    encoder_mask,
                    tokenizer_tgt,
                    max_len=max_len,
                    device=device,
                    beam_size=beam_size,
                    length_penalty_alpha=length_penalty_alpha,
                ).tolist()
            elif decode == "greedy":
                out_ids = greedy_decode(
                    model,
                    encoder_input,
                    encoder_mask,
                    tokenizer_tgt,
                    max_len=max_len,
                    device=device,
                ).tolist()
            else:
                raise ValueError(f"Unknown decode: {decode}")

            out_ids = _strip_special_ids(out_ids, tokenizer_tgt)
            pred_text = tokenizer_tgt.decode(out_ids)
            preds.append(pred_text)
            refs.append(tgt_text)
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"  BLEU decoding: {n} samples...", flush=True)

    return float(sacrebleu.corpus_bleu(preds, [refs]).score)


def _get_test_ds_and_tokenizer_tgt(cfg: dict[str, Any]):
    """
    Load only the cached *test* split + tokenizer(s), avoiding loading massive train/val.
    Assumes cache exists under cfg['dataset_cache_dir'].
    """
    from datasets import load_from_disk
    from tokenizers import Tokenizer
    from src.dataset.dataset import BilingualDatasetFromIds

    cache_dir = Path(cfg.get("dataset_cache_dir", "data/tokenized"))
    cache_name = f"{str(cfg['datasource']).replace('/', '_')}_{cfg['lang_src']}_{cfg['lang_tgt']}"
    cache_path = cache_dir / cache_name
    test_path = cache_path / "test"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Tokenized test split not found: {test_path}. "
            "Run training once (or build the cache) to create data/tokenized/*."
        )

    use_joint_bpe = bool(cfg.get("use_joint_bpe", False))
    if use_joint_bpe:
        joint_path = Path(cfg.get("tokenizer_joint_file", "data/tokenizer_joint.json"))
        if not joint_path.exists():
            raise FileNotFoundError(f"Joint tokenizer not found: {joint_path}")
        tokenizer_src = Tokenizer.from_file(str(joint_path))
        tokenizer_tgt = tokenizer_src
    else:
        def _tok_path(lang: str) -> Path:
            p = Path(str(cfg["tokenizer_file"]).format(lang))
            if p.exists():
                return p
            fallback = Path(f"tokenizer_{lang}.json")
            if fallback.exists():
                return fallback
            raise FileNotFoundError(f"Tokenizer not found: {p} (nor {fallback})")

        tokenizer_src = Tokenizer.from_file(str(_tok_path(cfg["lang_src"])))
        tokenizer_tgt = Tokenizer.from_file(str(_tok_path(cfg["lang_tgt"])))

    test_hf = load_from_disk(str(test_path))
    test_ds = BilingualDatasetFromIds(test_hf, tokenizer_src, tokenizer_tgt, int(cfg["seq_len"]), return_variable_length=False)
    return test_ds, tokenizer_tgt


def _get_val_ds_and_tokenizer_tgt(cfg: dict[str, Any]):
    """
    Load only the cached *val* split + tokenizer(s), avoiding loading massive train/test.
    Assumes cache exists under cfg['dataset_cache_dir'].
    """
    from datasets import load_from_disk
    from tokenizers import Tokenizer
    from src.dataset.dataset import BilingualDatasetFromIds

    cache_dir = Path(cfg.get("dataset_cache_dir", "data/tokenized"))
    cache_name = f"{str(cfg['datasource']).replace('/', '_')}_{cfg['lang_src']}_{cfg['lang_tgt']}"
    cache_path = cache_dir / cache_name
    val_path = cache_path / "val"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Tokenized val split not found: {val_path}. "
            "Run training once (or build the cache) to create data/tokenized/*."
        )

    use_joint_bpe = bool(cfg.get("use_joint_bpe", False))
    if use_joint_bpe:
        joint_path = Path(cfg.get("tokenizer_joint_file", "data/tokenizer_joint.json"))
        if not joint_path.exists():
            raise FileNotFoundError(f"Joint tokenizer not found: {joint_path}")
        tokenizer_src = Tokenizer.from_file(str(joint_path))
        tokenizer_tgt = tokenizer_src
    else:
        def _tok_path(lang: str) -> Path:
            p = Path(str(cfg["tokenizer_file"]).format(lang))
            if p.exists():
                return p
            fallback = Path(f"tokenizer_{lang}.json")
            if fallback.exists():
                return fallback
            raise FileNotFoundError(f"Tokenizer not found: {p} (nor {fallback})")

        tokenizer_src = Tokenizer.from_file(str(_tok_path(cfg["lang_src"])))
        tokenizer_tgt = Tokenizer.from_file(str(_tok_path(cfg["lang_tgt"])))

    val_hf = load_from_disk(str(val_path))
    val_ds = BilingualDatasetFromIds(val_hf, tokenizer_src, tokenizer_tgt, int(cfg["seq_len"]), return_variable_length=False)
    return val_ds, tokenizer_tgt


def _compute_ppl_from_ckpt_on_val(
    *,
    model,
    device: torch.device,
    val_ds,
    pad_id: int,
    max_samples: int | None,
    batch_size: int,
) -> tuple[float, float, int]:
    """
    Compute token-level NLL and PPL on val split (teacher forcing).
    Returns: (ppl, nll, n_tokens) where nll is average negative log-likelihood per non-pad token.

    Notes:
    - This intentionally excludes label smoothing (pure NLL) to be comparable to typical paper reporting.
    - We use reduction='sum' then divide by total non-pad tokens (token-weighted average).
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model.eval()

    total_nll_sum = 0.0
    total_tokens = 0
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and seen >= max_samples:
                break
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            b = encoder_input.size(0)
            if max_samples is not None and seen + b > max_samples:
                b = max_samples - seen
                encoder_input = encoder_input[:b]
                decoder_input = decoder_input[:b]
                encoder_mask = encoder_mask[:b]
                decoder_mask = decoder_mask[:b]
                label = label[:b]

            enc = model.encode(encoder_input, encoder_mask)
            dec = model.decode(enc, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(dec)  # (B, T, V)

            # Sum NLL over all non-pad target tokens.
            vocab = logits.size(-1)
            nll_sum = F.cross_entropy(
                logits.view(-1, vocab),
                label.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            ).item()
            n_tokens = int((label != pad_id).sum().item())
            total_nll_sum += float(nll_sum)
            total_tokens += n_tokens
            seen += b

    if total_tokens <= 0:
        raise RuntimeError("No non-pad tokens found while computing val PPL.")
    nll = total_nll_sum / total_tokens
    ppl = float(math.exp(min(nll, 50)))
    return ppl, float(nll), int(total_tokens)


def _estimate_flops_g_train_forward(
    *,
    vocab_src: int,
    vocab_tgt: int,
    seq_len: int,
    d_model: int,
    n_head: int,
    d_ff: int,
    layers: int,
    dropout: float,
    d_k: int | None,
    d_v: int | None,
) -> float:
    """
    Approx FLOPs (GFLOPs) for a single teacher-forcing forward used in training:
      encode(encoder_input, encoder_mask) + decode(..., decoder_input, decoder_mask) + project(dec_out)
    Measured with fvcore's FlopCountAnalysis on CPU, batch_size=1, seq_len=seq_len.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as e:
        raise RuntimeError("fvcore is required for FLOPs. Install: pip install fvcore") from e

    import torch.nn as nn
    from src.model import build_transformer
    from src.dataset.dataset import causal_mask

    device = torch.device("cpu")
    model = build_transformer(
        vocab_src,
        vocab_tgt,
        seq_len,
        seq_len,
        d_model=d_model,
        N=layers,
        h=n_head,
        d_ff=d_ff,
        dropout=dropout,
        d_k=d_k,
        d_v=d_v,
    ).to(device)
    model.eval()

    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
            enc = self.m.encode(encoder_input, encoder_mask)
            dec = self.m.decode(enc, encoder_mask, decoder_input, decoder_mask)
            out = self.m.project(dec)
            return out

    wrapper = _Wrapper(model)

    b = 1
    encoder_input = torch.zeros((b, seq_len), dtype=torch.long, device=device)
    decoder_input = torch.zeros((b, seq_len), dtype=torch.long, device=device)
    encoder_mask = torch.ones((b, 1, 1, seq_len), dtype=torch.int, device=device)
    decoder_mask = (torch.ones((b, 1, seq_len), dtype=torch.int, device=device) & causal_mask(seq_len).int()).to(device)

    flops = FlopCountAnalysis(wrapper, (encoder_input, encoder_mask, decoder_input, decoder_mask)).total()
    return float(flops) / 1e9


def summarize_runs(
    runs_dir: Path,
    out_path: Path,
    ppl_mode: str,
    skip_bleu: bool,
    bleu_max_samples: int | None,
    decode: str,
    beam_size: int,
    length_penalty_alpha: float,
    checkpoint_name: str | None,
    device: torch.device,
    dataset_seed: int,
    limit_runs: int | None,
    verbose: bool = False,
    strict: bool = False,
    bleu_progress_every: int | None = None,
    skip_flops: bool = False,
    recompute_ppl: bool = False,
    ppl_max_samples: int | None = 2000,
    ppl_batch_size: int = 32,
) -> list[RunSummary]:
    run_dirs = [p for p in sorted(runs_dir.iterdir()) if p.is_dir()]
    if limit_runs is not None:
        run_dirs = run_dirs[:limit_runs]

    summaries: list[RunSummary] = []
    bleu_data_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
    flops_cache: dict[tuple[Any, ...], float] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "seed",
                "d_model",
                "n_head",
                "d_k",
                "d_v",
                "d_ff",
                "layers",
                f"{ppl_mode}_val_ppl",
                f"{ppl_mode}_val_ppl_step",
                "recomputed_val_ppl",
                "recomputed_val_nll",
                "recomputed_val_tokens",
                "bleu",
                "flops_g",
                "params_m",
                "ckpt",
                "metrics_csv",
                "error",
            ],
        )
        w.writeheader()
        f.flush()

        for i, run_dir in enumerate(run_dirs, 1):
            if verbose:
                print(f"[{i}/{len(run_dirs)}] {run_dir.name}", flush=True)
            metrics_csv = run_dir / "metrics" / "metrics.csv"
            rows = _read_metrics_val_ppl(metrics_csv)
            ppl, ppl_step = _pick_val_ppl(rows, ppl_mode)

            ckpt_path = _pick_checkpoint(run_dir, preferred=checkpoint_name)
            cfg: dict[str, Any] = {}
            seed = d_model = n_head = d_ff = layers = d_k = d_v = None
            params_m = None
            bleu = None
            flops_g = None
            recomputed_val_ppl: float | None = None
            recomputed_val_nll: float | None = None
            recomputed_val_tokens: int | None = None
            err: str | None = None

            if ckpt_path is not None and ckpt_path.exists():
                try:
                    _, cfg, model, vocab_src, vocab_tgt = _load_ckpt_and_build_model(ckpt_path, device=device)
                    seed = _int_or_none(cfg.get("seed"))
                    d_model = _int_or_none(cfg.get("d_model"))
                    n_head = _int_or_none(cfg.get("n_head"))
                    d_ff = _int_or_none(cfg.get("d_ff"))
                    layers = _int_or_none(cfg.get("layers"))
                    d_k = _int_or_none(cfg.get("d_k"))
                    d_v = _int_or_none(cfg.get("d_v"))
                    params_m = _count_params_m(model)
                    if not skip_flops and d_model and n_head and d_ff and layers:
                        flops_key = (
                            vocab_src,
                            vocab_tgt,
                            int(cfg.get("seq_len", 100)),
                            d_model,
                            n_head,
                            d_ff,
                            layers,
                            float(cfg.get("dropout", 0.1)),
                            d_k,
                            d_v,
                        )
                        if flops_key not in flops_cache:
                            flops_cache[flops_key] = _estimate_flops_g_train_forward(
                                vocab_src=vocab_src,
                                vocab_tgt=vocab_tgt,
                                seq_len=int(cfg.get("seq_len", 100)),
                                d_model=d_model,
                                n_head=n_head,
                                d_ff=d_ff,
                                layers=layers,
                                dropout=float(cfg.get("dropout", 0.1)),
                                d_k=d_k,
                                d_v=d_v,
                            )
                        flops_g = flops_cache[flops_key]
                    if not skip_bleu:
                        cfg_eval = dict(cfg)
                        # dataset cache does not depend on random seed; keep deterministic key
                        key = (
                            cfg_eval.get("dataset_cache_dir"),
                            cfg_eval.get("datasource"),
                            cfg_eval.get("lang_src"),
                            cfg_eval.get("lang_tgt"),
                            cfg_eval.get("seq_len"),
                            cfg_eval.get("use_joint_bpe"),
                            cfg_eval.get("tokenizer_joint_file"),
                            cfg_eval.get("tokenizer_file"),
                            cfg_eval.get("tokenizer_vocab_size"),
                        )
                        if key not in bleu_data_cache:
                            bleu_data_cache[key] = _get_test_ds_and_tokenizer_tgt(cfg_eval)
                        test_ds, tokenizer_tgt = bleu_data_cache[key]
                        bleu = _compute_bleu_for_run(
                            model=model,
                            device=device,
                            test_ds=test_ds,
                            tokenizer_tgt=tokenizer_tgt,
                            max_len=int(cfg_eval.get("seq_len", 100)),
                            bleu_max_samples=bleu_max_samples,
                            decode=decode,
                            beam_size=beam_size,
                            length_penalty_alpha=length_penalty_alpha,
                            progress_every=bleu_progress_every if verbose else None,
                        )

                    if recompute_ppl:
                        cfg_eval = dict(cfg)
                        val_ds, _tokenizer_tgt = _get_val_ds_and_tokenizer_tgt(cfg_eval)
                        # pad id from tokenizer (works for joint/non-joint)
                        try:
                            pad_id = int(_tokenizer_tgt.token_to_id("[PAD]"))
                        except Exception:
                            raise RuntimeError("Failed to get [PAD] token id from tokenizer for PPL recomputation.")
                        recomputed_val_ppl, recomputed_val_nll, recomputed_val_tokens = _compute_ppl_from_ckpt_on_val(
                            model=model,
                            device=device,
                            val_ds=val_ds,
                            pad_id=pad_id,
                            max_samples=ppl_max_samples,
                            batch_size=ppl_batch_size,
                        )
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    if verbose:
                        print(f"[warn] {run_dir.name}: {err}", flush=True)
                    if strict:
                        raise
                finally:
                    if device.type == "cuda":
                        try:
                            del model  # type: ignore[name-defined]
                        except Exception:
                            pass
                        torch.cuda.empty_cache()

            summary = RunSummary(
                run=run_dir.name,
                seed=seed,
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                layers=layers,
                d_k=d_k,
                d_v=d_v,
                ppl=ppl,
                ppl_step=ppl_step,
                bleu=bleu,
                flops_g=flops_g,
                params_m=params_m,
                ckpt=str(ckpt_path) if ckpt_path else None,
                metrics_csv=str(metrics_csv) if metrics_csv.exists() else None,
                error=err,
            )
            summaries.append(summary)

            # Write one row immediately and flush, so partial progress is saved.
            w.writerow(
                {
                    "run": summary.run,
                    "seed": summary.seed,
                    "d_model": summary.d_model,
                    "n_head": summary.n_head,
                    "d_k": summary.d_k,
                    "d_v": summary.d_v,
                    "d_ff": summary.d_ff,
                    "layers": summary.layers,
                    f"{ppl_mode}_val_ppl": summary.ppl,
                    f"{ppl_mode}_val_ppl_step": summary.ppl_step,
                    "recomputed_val_ppl": recomputed_val_ppl,
                    "recomputed_val_nll": recomputed_val_nll,
                    "recomputed_val_tokens": recomputed_val_tokens,
                    "bleu": summary.bleu,
                    "flops_g": summary.flops_g,
                    "params_m": summary.params_m,
                    "ckpt": summary.ckpt,
                    "metrics_csv": summary.metrics_csv,
                    "error": summary.error,
                }
            )
            f.flush()

            # Always print one-line summary when a run finishes.
            ppl_str = f"{summary.ppl:.4f}" if summary.ppl is not None else "NA"
            bleu_str = f"{summary.bleu:.4f}" if summary.bleu is not None else "NA"
            flops_str = f"{summary.flops_g:.3f}" if summary.flops_g is not None else "NA"
            params_str = f"{summary.params_m:.3f}" if summary.params_m is not None else "NA"
            step_str = str(summary.ppl_step) if summary.ppl_step is not None else "NA"
            msg = (
                f"[done] {summary.run} | {ppl_mode}_val_ppl={ppl_str} (step={step_str}) | "
                f"BLEU={bleu_str} | FLOPs(G)={flops_str} | Params(M)={params_str}"
            )
            if summary.error:
                msg += f" | error={summary.error}"
            print(msg, flush=True)

    return summaries


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize runs: ppl/bleu/params(M).")
    p.add_argument("--runs-dir", type=str, default="runs/experiment1", help="Directory containing run subfolders.")
    p.add_argument("--out", type=str, default=None, help="Output CSV path (default: <runs-dir>_summary.csv).")
    p.add_argument("--ppl-mode", type=str, choices=["best", "last"], default="best", help="Pick best(min) or last step val_ppl.")
    p.add_argument("--skip-bleu", action="store_true", help="Skip BLEU computation (faster).")
    p.add_argument(
        "--bleu-max-samples",
        type=int,
        default=2000,
        help="Max test samples for BLEU (default: 2000). Use 0 to evaluate on all (can be huge).",
    )
    p.add_argument("--decode", type=str, choices=["beam", "greedy"], default="beam", help="Decoding method for BLEU.")
    p.add_argument("--beam-size", type=int, default=4, help="Beam size if decode=beam.")
    p.add_argument("--length-penalty-alpha", type=float, default=0.6, help="Length penalty alpha for beam search.")
    p.add_argument("--checkpoint-name", type=str, default="tmodel_100k.pt", help="Preferred checkpoint filename under checkpoints/.")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0 ...")
    p.add_argument(
        "--recompute-ppl",
        action="store_true",
        help="Recompute val PPL from checkpoint on cached val split (token-level NLL, no label smoothing). Adds columns.",
    )
    p.add_argument(
        "--ppl-max-samples",
        type=int,
        default=2000,
        help="Max val samples for recomputed PPL (default: 2000). Use 0 for full val.",
    )
    p.add_argument(
        "--ppl-batch-size",
        type=int,
        default=32,
        help="Batch size for recomputed PPL (default: 32).",
    )
    p.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="(deprecated) kept for backward-compat; cache-based BLEU ignores this.",
    )
    p.add_argument("--limit-runs", type=int, default=None, help="Only process first N runs (debug).")
    p.add_argument("--verbose", action="store_true", help="Print per-run warnings (e.g., missing deps).")
    p.add_argument("--strict", action="store_true", help="Fail immediately on first error (debug).")
    p.add_argument("--bleu-progress-every", type=int, default=200, help="(verbose) print BLEU decoding progress every N samples.")
    p.add_argument("--skip-flops", action="store_true", help="Skip FLOPs estimation (faster).")
    args = p.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")

    out = Path(args.out) if args.out else runs_dir.with_name(f"{runs_dir.name}_summary.csv")
    device = _device_from_arg(args.device)
    bleu_max_samples = None if args.bleu_max_samples == 0 else args.bleu_max_samples
    ppl_max_samples = None if args.ppl_max_samples == 0 else args.ppl_max_samples

    summarize_runs(
        runs_dir=runs_dir,
        out_path=out,
        ppl_mode=args.ppl_mode,
        skip_bleu=bool(args.skip_bleu),
        bleu_max_samples=bleu_max_samples,
        decode=args.decode,
        beam_size=int(args.beam_size),
        length_penalty_alpha=float(args.length_penalty_alpha),
        checkpoint_name=args.checkpoint_name,
        device=device,
        dataset_seed=int(args.dataset_seed),
        limit_runs=args.limit_runs,
        verbose=bool(args.verbose),
        strict=bool(args.strict),
        bleu_progress_every=args.bleu_progress_every,
        skip_flops=bool(args.skip_flops),
        recompute_ppl=bool(args.recompute_ppl),
        ppl_max_samples=ppl_max_samples,
        ppl_batch_size=int(args.ppl_batch_size),
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

