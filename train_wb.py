from config.config import get_config, get_weights_file_path, get_run_id, ensure_run_dirs, seed_everything
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

import warnings
from tqdm import tqdm
import os
from pathlib import Path

import wandb

import torchmetrics

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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
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
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_ds(config):
    train_ds, val_ds, test_ds, tokenizer_src, tokenizer_tgt = build_tokenized_dataset(config)

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
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
        d_model=config['d_model'], N=config['layers'], h=config['n_head'], d_ff=config['d_ff'],
        dropout=config.get('dropout', 0.1),
        d_k=config.get('d_k'), d_v=config.get('d_v'),
    )
    return model

def train_model(config):
    seed_everything(config.get("seed", 42))
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # runs/run_id/tmodel, runs/run_id/metrics, runs/run_id/checkpoints 생성
    ensure_run_dirs(config)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

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
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch (설정 포함)
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'config': config,
        }, model_filename)


if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train transformer with Weights & Biases.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config (default: config/config.yaml)")
    args = parser.parse_args()
    config = get_config(yaml_path=args.config)
    config['num_epochs'] = config.get('num_epochs', 30)
    config['preload'] = config.get('preload', None)

    wandb.init(
        project="pytorch-transformer",
        config=config,
    )
    
    train_model(config)
