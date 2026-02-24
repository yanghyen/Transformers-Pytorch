import os
# PyTorch 2.0.x does not support expandable_segments (added in 2.1+); avoid RuntimeError
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from config.config import get_config, get_weights_file_path, latest_weights_file_path, get_run_id, ensure_run_dirs, get_tensorboard_dir, get_metrics_path, seed_everything
from src.model import build_transformer
from src.dataset.dataset import (
    build_tokenized_dataset,
    causal_mask,
    TokenBasedBatchSampler,
    collate_fn_variable_length,
)
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import warnings
import csv
import time
import math
from tqdm import tqdm
from pathlib import Path

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

def compute_validation_metrics(model, val_dataloader, loss_fn, tokenizer_src, tokenizer_tgt, max_len, device, pad_id, vocab_size):
    """Return (val_loss, val_ppl, val_acc) over validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_non_pad = 0
    num_val_samples = 0

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            b = encoder_input.size(0)
            num_val_samples += b

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            total_loss += loss.item() * b

            pred = proj_output.argmax(dim=-1)
            mask = (label != pad_id)
            total_correct += ((pred == label) & mask).sum().item()
            total_non_pad += mask.sum().item()

    val_loss = total_loss / num_val_samples if num_val_samples else 0.0
    val_ppl = float("inf") if val_loss > 50 else math.exp(min(val_loss, 50))
    val_acc = total_correct / total_non_pad if total_non_pad else 0.0

    return val_loss, val_ppl, val_acc

def get_ds(config):
    train_ds, val_ds, tokenizer_src, tokenizer_tgt = build_tokenized_dataset(config)

    if hasattr(train_ds, "hf_ds") and "src_len" in train_ds.hf_ds.column_names:
        max_len_src = max(train_ds.hf_ds["src_len"])
        max_len_tgt = max(train_ds.hf_ds["tgt_len"])
        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")

    use_token_batching = config.get("tokens_per_batch_src") is not None
    tokens_tgt = config.get("tokens_per_batch_tgt") or config.get("tokens_per_batch_src")

    if use_token_batching:
        batch_sampler = TokenBasedBatchSampler(
            train_ds.lengths,
            config["tokens_per_batch_src"],
            tokens_tgt,
            shuffle=True,
        )
        collate_fn = partial(
            collate_fn_variable_length,
            pad_token_id=tokenizer_tgt.token_to_id("[PAD]"),
            seq_len_max=config["seq_len"],
            causal_mask_fn=causal_mask,
        )
        train_dataloader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=collate_fn)
        print(f"Token-based batching: ~{config['tokens_per_batch_src']} src / ~{tokens_tgt} tgt tokens per batch")
    else:
        train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
        d_model=config['d_model'], N=config['layers'], h=config['n_head'], d_ff=config['d_ff'],
        dropout=config['dropout']
    )
    return model

def train_model(config):
    seed_everything(config.get("seed", 42))
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # runs/run_id/tmodel, runs/run_id/metrics, runs/run_id/checkpoints 생성
    ensure_run_dirs(config)
    run_id = get_run_id(config)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard: runs/run_id/tmodel
    writer = SummaryWriter(str(get_tensorboard_dir(config)))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        state = torch.load(model_filename)
        ckpt_src_vocab = state['model_state_dict']['src_embed.embedding.weight'].shape[0]
        ckpt_tgt_vocab = state['model_state_dict']['tgt_embed.embedding.weight'].shape[0]
        curr_src_vocab = tokenizer_src.get_vocab_size()
        curr_tgt_vocab = tokenizer_tgt.get_vocab_size()
        if ckpt_src_vocab != curr_src_vocab or ckpt_tgt_vocab != curr_tgt_vocab:
            print(f'Preload skipped: vocab size mismatch (checkpoint src={ckpt_src_vocab}, tgt={ckpt_tgt_vocab}; '
                  f'current src={curr_src_vocab}, tgt={curr_tgt_vocab}). Training from scratch.')
        else:
            print(f'Preloading model {model_filename}')
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    pad_id = tokenizer_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config['label_smoothing']).to(device)

    metrics_path = get_metrics_path(config)
    metrics_columns = ["step", "d_model", "n_head", "d_ff", "layers", "seed", "train_loss", "train_ppl", "train_acc", "val_loss", "val_ppl", "val_acc", "gpu_time_sec"]
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_columns)
    metrics_writer.writeheader()
    metrics_file.flush()

    start_time = time.perf_counter()

    try:
        for epoch in range(initial_epoch, config['num_epochs']):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                vocab_size_tgt = tokenizer_tgt.get_vocab_size()
                loss = loss_fn(proj_output.view(-1, vocab_size_tgt), label.view(-1))
                train_ppl = math.exp(min(loss.item(), 50))
                pred = proj_output.argmax(dim=-1)
                mask = (label != pad_id)
                train_correct = ((pred == label) & mask).sum().item()
                train_non_pad = mask.sum().item()
                train_acc = train_correct / train_non_pad if train_non_pad else 0.0
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                gpu_time_sec = time.perf_counter() - start_time
                if global_step > 0 and global_step % 1000 == 0:
                    val_loss, val_ppl, val_acc = compute_validation_metrics(
                        model, val_dataloader, loss_fn, tokenizer_src, tokenizer_tgt,
                        config['seq_len'], device, pad_id, tokenizer_tgt.get_vocab_size(),
                    )
                    metrics_writer.writerow({
                        "step": global_step,
                        "d_model": config["d_model"],
                        "n_head": config["n_head"],
                        "d_ff": config["d_ff"],
                        "layers": config["layers"],
                        "seed": config.get("seed", 42),
                        "train_loss": f"{loss.item():.6f}",
                        "train_ppl": f"{train_ppl:.6f}",
                        "train_acc": f"{train_acc:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "val_ppl": f"{val_ppl:.6f}",
                        "val_acc": f"{val_acc:.6f}",
                        "gpu_time_sec": f"{gpu_time_sec:.2f}",
                    })
                    metrics_file.flush()

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step >= config['max_steps']:
                    break

            if global_step >= config['max_steps']:
                print(f"Reached {config['max_steps']} steps (paper base). Stopping.")
                model_filename = get_weights_file_path(config, "100k")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': config,
                }, model_filename)
                break

            # Run validation at the end of every epoch (console print only; CSV is every 1000 steps)
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch (설정 포함)
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'config': config,
            }, model_filename)
    finally:
        metrics_file.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train transformer from YAML config.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config (default: config.yaml)")
    args = parser.parse_args()
    config = get_config(yaml_path=args.config)
    train_model(config)
