import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.config import get_config, latest_weights_file_path
from src.model import build_transformer
from src.dataset.dataset import BilingualDataset, causal_mask
from tokenizers import Tokenizer
from datasets import load_dataset
import torch
import torch.nn.functional as F

BEAM_SIZE = 4
LENGTH_PENALTY_ALPHA = 0.6


def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=BEAM_SIZE, length_penalty_alpha=LENGTH_PENALTY_ALPHA):
    """Beam search decoding with length penalty. Score = sum_log_prob / ((5+len)/6)^alpha."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    if source.dim() == 1:
        source = source.unsqueeze(0)
    encoder_output = model.encode(source, source_mask)
    decoder_initial = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    # (candidate_tensor, sum_log_prob, length)
    candidates = [(decoder_initial, 0.0, 1)]

    while True:
        if any(cand.size(1) >= max_len for cand, _, _ in candidates):
            break

        new_candidates = []
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

        # Length penalty: ((5+len)/6)^alpha. Rank by normalized score (higher is better).
        def normalized_score(item):
            cand, sum_lp, length = item
            if length == 0:
                return sum_lp
            lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
            return sum_lp / lp

        candidates = sorted(new_candidates, key=normalized_score, reverse=True)[:beam_size]

        if all(cand[0, -1].item() == eos_idx for cand, _, _ in candidates):
            break

    best = candidates[0][0].squeeze(0)
    return best


def translate(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config['seq_len'],
        d_model=config['d_model'], N=config.get('layers', 6), h=config.get('n_head', 8),
        d_ff=config.get('d_ff', 2048), dropout=config.get('dropout', 0.1),
        d_k=config.get('d_k'), d_v=config.get('d_v'),
    ).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # Beam search decode (beam=4, length_penalty alpha=0.6)
        decoder_output = beam_search_decode(
            model, source, source_mask,
            tokenizer_src, tokenizer_tgt, seq_len, device,
            beam_size=BEAM_SIZE, length_penalty_alpha=LENGTH_PENALTY_ALPHA,
        )

        predicted_text = tokenizer_tgt.decode(decoder_output.tolist())

        if label != "":
            print(f"{f'ID: ':>12}{id}")
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "":
            print(f"{f'TARGET: ':>12}{label}")
        print(f"{f'PREDICTED: ':>12}{predicted_text}")

    return predicted_text
    
#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")