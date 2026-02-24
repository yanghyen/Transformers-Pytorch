import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler


def get_all_sentences(ds, lang):
    for i in range(len(ds)):
        yield ds[i]["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        vocab_size = config.get("tokenizer_vocab_size", 37000)
        special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        print(f"  Training BPE tokenizer for '{lang}' (vocab_size={vocab_size}, min_frequency=2)...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        print(f"  -> Saved to {tokenizer_path} (vocab size: {tokenizer.get_vocab_size()})")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"  Loaded tokenizer for '{lang}' from {tokenizer_path} (vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len, return_variable_length=False):
        super().__init__()
        self.seq_len = seq_len
        self.return_variable_length = return_variable_length

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        # For token-based batching: precompute (src_content_len, tgt_content_len) per index
        self.lengths = None
        if return_variable_length:
            self._build_length_cache()

    def _build_length_cache(self):
        """Precompute content token lengths (no SOS/EOS) for each sample for batching by approximate length."""
        max_enc = self.seq_len - 2
        max_dec = self.seq_len - 1
        self.lengths = []
        for i in range(len(self.ds)):
            src_target_pair = self.ds[i]
            enc_ids = self.tokenizer_src.encode(src_target_pair['translation'][self.src_lang]).ids
            dec_ids = self.tokenizer_tgt.encode(src_target_pair['translation'][self.tgt_lang]).ids
            src_len = min(len(enc_ids), max_enc)
            tgt_len = min(len(dec_ids), max_dec)
            self.lengths.append((src_len, tgt_len))
        self.lengths = self.lengths

    def __len__(self):
        return len(self.ds)

    def _get_tokens_and_lengths(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        max_enc = self.seq_len - 2
        max_dec = self.seq_len - 1
        if len(enc_input_tokens) > max_enc:
            enc_input_tokens = enc_input_tokens[:max_enc]
        if len(dec_input_tokens) > max_dec:
            dec_input_tokens = dec_input_tokens[:max_dec]

        return enc_input_tokens, dec_input_tokens, src_text, tgt_text

    def __getitem__(self, idx):
        enc_input_tokens, dec_input_tokens, src_text, tgt_text = self._get_tokens_and_lengths(idx)

        if self.return_variable_length:
            # Variable-length: no padding, for token-based batching (padding in collate_fn)
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                ],
                dim=0,
            )
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                ],
                dim=0,
            )
            label = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                ],
                dim=0,
            )
            return {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "label": label,
                "src_len": len(enc_input_tokens),
                "tgt_len": len(dec_input_tokens),
                "src_text": src_text,
                "tgt_text": tgt_text,
            }

        # Fixed-length (original): pad to seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class TokenBasedBatchSampler(Sampler):
    """Batch sampler: group by approximate sequence length, each batch ~tokens_per_batch_src/tgt tokens (paper)."""

    def __init__(self, lengths, tokens_per_batch_src, tokens_per_batch_tgt, shuffle=True):
        self.lengths = lengths  # list of (src_len, tgt_len)
        self.tokens_per_batch_src = tokens_per_batch_src
        self.tokens_per_batch_tgt = tokens_per_batch_tgt
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.lengths)
        indices = list(range(n))
        if self.shuffle:
            # Sort by source length so similar-length sentences are batched together (paper)
            indices.sort(key=lambda i: self.lengths[i][0])
            # Shuffle within similar-length groups to avoid always same order
            import random
            # Shuffle in chunks of ~same length (e.g. bucket by length then shuffle buckets)
            chunk_size = max(1, n // 50)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = indices[start:end]
                random.shuffle(chunk)
                indices[start:end] = chunk

        batch = []
        batch_src = 0
        batch_tgt = 0
        for i in indices:
            src_len, tgt_len = self.lengths[i]
            # Start new batch if adding this would exceed budget (and we already have at least one)
            if batch and (batch_src + src_len > self.tokens_per_batch_src or batch_tgt + tgt_len > self.tokens_per_batch_tgt):
                yield batch
                batch = []
                batch_src = 0
                batch_tgt = 0
            batch.append(i)
            batch_src += src_len
            batch_tgt += tgt_len
        if batch:
            yield batch

    def __len__(self):
        # Approximate number of batches (exact would require full iteration)
        total_src = sum(l[0] for l in self.lengths)
        total_tgt = sum(l[1] for l in self.lengths)
        n = len(self.lengths)
        avg_src = total_src / n if n else 0
        avg_tgt = total_tgt / n if n else 0
        if avg_src <= 0 or avg_tgt <= 0:
            return max(1, n)
        return max(1, int(min(total_src / self.tokens_per_batch_src, total_tgt / self.tokens_per_batch_tgt)))


def collate_fn_variable_length(batch, pad_token_id, seq_len_max, causal_mask_fn):
    """Pad variable-length samples to max length in batch (capped at seq_len_max)."""
    pad_id = pad_token_id
    encoder_inputs = [b["encoder_input"] for b in batch]
    decoder_inputs = [b["decoder_input"] for b in batch]
    labels = [b["label"] for b in batch]

    max_enc = min(max(e.size(0) for e in encoder_inputs), seq_len_max)
    max_dec = min(max(d.size(0) for d in decoder_inputs), seq_len_max)

    def pad_to(tensor, length, pad_id):
        if tensor.size(0) >= length:
            return tensor[:length]
        return torch.cat([tensor, torch.full((length - tensor.size(0),), pad_id, dtype=torch.int64)])

    encoder_input = torch.stack([pad_to(e, max_enc, pad_id) for e in encoder_inputs])
    decoder_input = torch.stack([pad_to(d, max_dec, pad_id) for d in decoder_inputs])
    label = torch.stack([pad_to(l, max_dec, pad_id) for l in labels])

    encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
    decoder_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2) & causal_mask_fn(max_dec)  # (B, 1, seq_len, seq_len)

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": encoder_mask.int(),
        "decoder_mask": decoder_mask.int(),
        "label": label,
        "src_text": [b["src_text"] for b in batch],
        "tgt_text": [b["tgt_text"] for b in batch],
    }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


# ---------------------------------------------------------------------------
# 전처리: BPE 토큰화 후 저장 → 학습 시 token id만 로드 (dataset.map 방식)
# ---------------------------------------------------------------------------

class BilingualDatasetFromIds(Dataset):
    """이미 토큰화된 데이터셋(src_ids, tgt_ids)을 로드해 학습용 텐서만 반환. 토크나이저는 pad/sos/eos용으로만 사용."""

    def __init__(self, hf_ds, tokenizer_src, tokenizer_tgt, seq_len, return_variable_length=False):
        super().__init__()
        self.hf_ds = hf_ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.return_variable_length = return_variable_length

        self.pad_id = tokenizer_tgt.token_to_id("[PAD]")
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.pad_id], dtype=torch.int64)

        self.lengths = None
        if return_variable_length:
            self._build_length_cache()

    def _build_length_cache(self):
        if "src_len" in self.hf_ds.column_names and "tgt_len" in self.hf_ds.column_names:
            self.lengths = [(r["src_len"], r["tgt_len"]) for r in self.hf_ds]
        else:
            self.lengths = [
                (len(self.hf_ds[i]["src_ids"]), len(self.hf_ds[i]["tgt_ids"]))
                for i in range(len(self.hf_ds))
            ]

    def __len__(self):
        return len(self.hf_ds)

    def _ids_to_tensors(self, enc_ids, dec_ids, src_text, tgt_text):
        max_enc = self.seq_len - 2
        max_dec = self.seq_len - 1
        enc_ids = enc_ids[:max_enc]
        dec_ids = dec_ids[:max_dec]

        if self.return_variable_length:
            encoder_input = torch.cat(
                [self.sos_token, torch.tensor(enc_ids, dtype=torch.int64), self.eos_token], dim=0
            )
            decoder_input = torch.cat(
                [self.sos_token, torch.tensor(dec_ids, dtype=torch.int64)], dim=0
            )
            label = torch.cat(
                [torch.tensor(dec_ids, dtype=torch.int64), self.eos_token], dim=0
            )
            return {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "label": label,
                "src_len": len(enc_ids),
                "tgt_len": len(dec_ids),
                "src_text": src_text,
                "tgt_text": tgt_text,
            }

        enc_pad = self.seq_len - len(enc_ids) - 2
        dec_pad = self.seq_len - len(dec_ids) - 1
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_ids, dtype=torch.int64),
                self.eos_token,
                torch.full((enc_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_ids, dtype=torch.int64),
                torch.full((dec_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )
        label = torch.cat(
            [
                torch.tensor(dec_ids, dtype=torch.int64),
                self.eos_token,
                torch.full((dec_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        return self._ids_to_tensors(
            row["src_ids"], row["tgt_ids"], row["src_text"], row["tgt_text"]
        )


def _make_tokenize_map_fn(tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, max_enc, max_dec):
    """dataset.map(batched=True)용: raw translation → src_ids, tgt_ids, src_text, tgt_text, src_len, tgt_len."""

    def fn(examples):
        trans = examples["translation"]
        src_texts = [t[src_lang] for t in trans]
        tgt_texts = [t[tgt_lang] for t in trans]
        src_ids = [tokenizer_src.encode(s).ids[:max_enc] for s in src_texts]
        tgt_ids = [tokenizer_tgt.encode(t).ids[:max_dec] for t in tgt_texts]
        src_len = [len(ids) for ids in src_ids]
        tgt_len = [len(ids) for ids in tgt_ids]
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_text": src_texts,
            "tgt_text": tgt_texts,
            "src_len": src_len,
            "tgt_len": tgt_len,
        }

    return fn


def build_tokenized_dataset(config, map_batch_size=2000):
    """
    raw 데이터셋을 한 번 BPE 토큰화해 캐시에 저장하고, 학습 시에는 token id만 로드.
    반환: (train_ds, val_ds, tokenizer_src, tokenizer_tgt)
    train_ds, val_ds는 BilingualDatasetFromIds (또는 기존 BilingualDataset 호환 인터페이스).
    """
    from datasets import load_dataset

    cache_dir = Path(config.get("dataset_cache_dir", "data/tokenized"))
    cache_name = f"{config['datasource'].replace('/', '_')}_{config['lang_src']}_{config['lang_tgt']}"
    cache_path = cache_dir / cache_name
    train_path = cache_path / "train"
    val_path = cache_path / "val"

    seq_len = config["seq_len"]
    max_enc = seq_len - 2
    max_dec = seq_len - 1
    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]

    if train_path.exists() and val_path.exists():
        from tokenizers import Tokenizer
        from datasets import load_from_disk

        def _tokenizer_path(lang):
            p = Path(config["tokenizer_file"].format(lang))
            if p.exists():
                return str(p)
            fallback = Path(f"tokenizer_{lang}.json")
            if fallback.exists():
                return str(fallback)
            raise FileNotFoundError(
                f"Tokenizer not found: {p} (nor {fallback}). "
                "Delete cache under data/tokenized/ and re-run to build tokenizers in data/."
            )

        print(f"Loading pre-tokenized dataset from {cache_path}")
        train_hf = load_from_disk(str(train_path))
        val_hf = load_from_disk(str(val_path))
        tokenizer_src = Tokenizer.from_file(_tokenizer_path(src_lang))
        tokenizer_tgt = Tokenizer.from_file(_tokenizer_path(tgt_lang))
        use_token_batching = config.get("tokens_per_batch_src") is not None
        train_ds = BilingualDatasetFromIds(
            train_hf, tokenizer_src, tokenizer_tgt, seq_len, return_variable_length=use_token_batching
        )
        val_ds = BilingualDatasetFromIds(val_hf, tokenizer_src, tokenizer_tgt, seq_len, return_variable_length=False)
        return train_ds, val_ds, tokenizer_src, tokenizer_tgt

    # [1/3] Raw 다운로드
    print("[1/3] Downloading raw dataset...")
    print(f"  datasource={config['datasource']}, split={src_lang}-{tgt_lang}")
    raw = load_dataset(
        config["datasource"],
        f"{src_lang}-{tgt_lang}",
        split="train",
        trust_remote_code=True,
    )
    n_raw = len(raw)
    print(f"  -> Loaded {n_raw} examples.")

    # [2/3] BPE 토크나이저 학습 또는 로드
    print("[2/3] Building/loading BPE tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, raw, tgt_lang)

    # [3/3] dataset.map으로 토큰화
    print(f"[3/3] Tokenizing with dataset.map (batch_size={map_batch_size})...")
    tokenize_fn = _make_tokenize_map_fn(
        tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, max_enc, max_dec
    )
    ds_tokenized = raw.map(
        tokenize_fn,
        batched=True,
        batch_size=map_batch_size,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )
    n = len(ds_tokenized)
    print(f"  -> Tokenized {n} examples.")

    # Train/Val 분리 및 저장
    train_size = int(0.9 * n)
    val_size = n - train_size
    indices = list(range(n))
    random.seed(config.get("seed", 42))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_hf = ds_tokenized.select(train_indices)
    val_hf = ds_tokenized.select(val_indices)

    print(f"Saving to disk: {cache_path}")
    cache_path.mkdir(parents=True, exist_ok=True)
    train_hf.save_to_disk(str(train_path))
    val_hf.save_to_disk(str(val_path))
    print(f"  -> Saved (train={train_size}, val={val_size})")

    use_token_batching = config.get("tokens_per_batch_src") is not None
    train_ds = BilingualDatasetFromIds(
        train_hf, tokenizer_src, tokenizer_tgt, seq_len, return_variable_length=use_token_batching
    )
    val_ds = BilingualDatasetFromIds(val_hf, tokenizer_src, tokenizer_tgt, seq_len, return_variable_length=False)
    return train_ds, val_ds, tokenizer_src, tokenizer_tgt