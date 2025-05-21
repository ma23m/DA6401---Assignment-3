# Import csv for saving predictions later
import csv

# Import PyTorch and other necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import wandb

from train_PartA import Seq2Seq, build_vocab, TransliterationDataset, collate_fn, train, evaluate

wandb.login(key="580e769ee2f34eafdded556ce52aaf31c265ad3b")

# Define beam search prediction function
def predict_with_beam_search(model, src_seq, src_vocab, tgt_vocab, beam_width=3, max_len=30):
    model.eval()  # Set model to evaluation mode
    sos_token = tgt_vocab['<sos>']  # Start of sequence token
    eos_token = tgt_vocab['<eos>']  # End of sequence token
    pad_token = tgt_vocab['<pad>']  # Padding token
    # Reverse lookup dictionary from index to token
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}

    with torch.no_grad():  # No gradient calculation for inference
        # Convert source sequence to tensor with <sos> and <eos> tokens
        src_seq = torch.tensor(
            [src_vocab['<sos>']] + [src_vocab.get(c, src_vocab['<unk>']) for c in src_seq] + [src_vocab['<eos>']]
        ).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Embed source sequence
        embedded_src = model.embedding_src(src_seq)
        
        # Pass embedded source through encoder
        encoder_output, hidden = model.encoder(embedded_src)

        # If hidden is a tuple (LSTM), unpack hidden and cell states
        if isinstance(hidden, tuple):
            h, c = hidden
        else:
            h, c = hidden, None

        # Ensure hidden states match expected decoder layers count
        expected_layers = model.config.get("decoder_num_layers", 3)
        actual_layers = h.shape[0]
        
        if actual_layers < expected_layers:
            diff = expected_layers - actual_layers
            # Pad hidden state with zeros for missing layers
            extra_h = torch.zeros(diff, h.shape[1], h.shape[2], device=h.device)
            h = torch.cat([h, extra_h], dim=0)
            
            if c is not None:
                # Pad cell state as well if exists
                extra_c = torch.zeros(diff, c.shape[1], c.shape[2], device=c.device)
                c = torch.cat([c, extra_c], dim=0)

        # Initialize beams with sequence containing only <sos> token and score 0
        beams = [(torch.tensor([sos_token], device=device), 0.0, h, c)]

        # Iterate over max length to generate sequences
        for _ in range(max_len):
            new_beams = []
            # For each current beam sequence
            for seq, score, h, c in beams:
                # If last token is <eos>, keep beam as is
                if seq[-1].item() == eos_token:
                    new_beams.append((seq, score, h, c))
                    continue

                # Embed the last predicted token
                embedded = model.embedding_tgt(seq[-1].unsqueeze(0).unsqueeze(0))  # Shape: (1, 1, embed_dim)

                # Decode next token depending on cell type
                if model.config['cell_type'] == 'LSTM':
                    output, (h_new, c_new) = model.decoder(embedded, (h, c))
                else:
                    output, h_new = model.decoder(embedded, h)
                    c_new = None

                # Compute logits from decoder output
                logits = model.fc_out(output.squeeze(1))  # Shape: (1, vocab_size)
                # Compute log probabilities for next tokens
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # Shape: (vocab_size)

                # Pick top-k tokens according to beam_width
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                # Create new candidate beams from top-k tokens
                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])  # Append new token to sequence
                    new_score = score + log_prob.item()  # Update score
                    new_beams.append((new_seq, new_score, h_new, c_new))

            # Keep only top beam_width sequences for next step
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Pick best sequence from beams (highest score)
        best_seq = beams[0][0]
        # Convert indices to tokens, ignore <pad> and <eos> tokens
        return ''.join([tgt_index_to_token[token.item()] for token in best_seq[1:] if token.item() not in [pad_token, eos_token]])


# Sweep training and prediction function to be called by wandb agent
def sweep_train_pred():
    wandb.init()  # Initialize wandb run
    config = wandb.config  # Get current config for sweep
    
    # Load train, dev, and test datasets from tsv files
    train_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    train_df = train_df.loc[train_df.index.repeat(train_df['freq'])].reset_index(drop=True)
    dev_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    test_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])

    # Ensure source and target columns are strings
    train_df['src'] = train_df['src'].astype(str)
    train_df['tgt'] = train_df['tgt'].astype(str)

    # Build vocabularies for source and target languages
    src_vocab = build_vocab(train_df['src'])
    tgt_vocab = build_vocab(train_df['tgt'])

    print(src_vocab)  # Print source vocabulary
    print(tgt_vocab)  # Print target vocabulary

    # Create reverse lookup dictionaries for target vocab
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    # Prepare data tuples for datasets
    train_data = list(zip(train_df['src'], train_df['tgt']))
    dev_data = list(zip(dev_df['src'], dev_df['tgt']))
    test_data = list(zip(test_df['src'], test_df['tgt']))

    # Create dataset objects (assumes TransliterationDataset class is defined)
    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

    # Create DataLoader objects for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the Seq2Seq model with config and vocab sizes
    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)

    # Define optimizer and loss criterion (ignore padding token index)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training loop for the number of epochs
    for epoch in range(config['epochs']):
        # Train model and compute loss and accuracy
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        # Evaluate on validation set
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        # Evaluate on test set
        test_loss, test_acc, test_word_acc = evaluate(model, test_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)

        # Print epoch results
        print(f"Epoch {epoch + 1}")
        print(f"{'train_loss:':20} {train_loss:.4f}")
        print(f"{'val_loss:':20} {val_loss:.4f}")
        print(f"{'test_loss:':20} {test_loss:.4f}")
        print(f"{'train_accuracy:':20} {train_acc * 100:.2f}%")
        print(f"{'val_accuracy:':20} {val_acc * 100:.2f}%")
        print(f"{'test_accuracy:':20} {test_acc * 100:.2f}%")
        print(f"{'train_word_accuracy:':20} {train_word_acc * 100:.2f}%")
        print(f"{'val_word_accuracy:':20} {val_word_acc * 100:.2f}%")
        print(f"{'test_word_accuracy:':20} {test_word_acc * 100:.2f}%")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_accuracy": train_acc * 100,
            "val_accuracy": val_acc * 100,
            "test_accuracy": test_acc * 100,
            "train_word_accuracy": train_word_acc * 100,
            "val_word_accuracy": val_word_acc * 100,
            "test_word_accuracy": test_word_acc * 100
        })

    # Get beam width from config
    beam_width = config.get('beam_width', 1)  # Default beam_width is 1 if not specified
    print('beam_width', beam_width)
        
    results = []
    # Predict using beam search for some test samples
    for sample_src, actual_tgt in test_data[:9229]:  # Adjust range as needed
        pred_seq = predict_with_beam_search(model, sample_src, src_vocab, tgt_vocab, beam_width=beam_width)
        # Filter out special tokens from prediction
        pred_tokens = [c for c in pred_seq if c not in ['<sos>', '<pad>', '<eos>']]
        pred_str = ''.join(pred_tokens)

        # Print input, actual, and predicted outputs
        print(f"Input:      {sample_src}")
        print(f"Actual:     {actual_tgt}")
        print(f"Prediction: {pred_str}")
        print("-" * 30)
        
        # Save results in a list
        results.append({
            "Input": sample_src,
            "Actual": actual_tgt,
            "Prediction": pred_str
        })
    
    # Save all predictions to CSV file
    output_csv_path = "beam_search_predictions.csv"
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Input", "Actual", "Prediction"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved predictions to {output_csv_path}")


# Set device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define sweep config dictionary for wandb hyperparameter sweep
sweep_config = {
    'method': 'random',
    'name': 'DakshinaSweepForPred_Best_without_attn_nothing',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embedding_dim': {'values': [256]},
        'hidden_size': {'values': [128]},
        'encoder_layers': {'values': [2]},
        'decoder_layers': {'values': [3]},
        'cell_type': {'values': ['LSTM']},
        'dropout': {'values': [0.3]},
        'epochs': {'values': [1]},  # Keep epochs low for demo, change as needed
        'beam_width': {'values': [3]}
    }
}

# Initialize wandb sweep and start agent for 1 run
sweep_id = wandb.sweep(sweep_config, project="DL_A3")
wandb.agent(sweep_id, function=sweep_train_pred, count=1)
