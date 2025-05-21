# Importing necessary libraries
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import wandb  # Weights & Biases for experiment tracking
import argparse  # Command-line argument parsing
from torch.utils.data import Dataset, DataLoader  # Dataset and data loading utilities
import pandas as pd  # Data handling with DataFrames

# -------------------- Dataset -------------------- #
# Custom Dataset for transliteration
class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data  # List of (src, tgt) word pairs
        self.src_vocab = src_vocab  # Source vocabulary dictionary
        self.tgt_vocab = tgt_vocab  # Target vocabulary dictionary

    def __len__(self):
        return len(self.data)  # Return total number of examples

    def __getitem__(self, idx):
        src, tgt = self.data[idx]  # Get source and target word at index
        # Convert source characters to indices, add <sos> and <eos>
        src_ids = [self.src_vocab['<sos>']] + [self.src_vocab.get(c, self.src_vocab['<unk>']) for c in src] + [self.src_vocab['<eos>']]
        # Convert target characters to indices, add <sos> and <eos>
        tgt_ids = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(c, self.tgt_vocab['<unk>']) for c in tgt] + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)  # Return as tensors

# Padding function to prepare batches
def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)  # Separate source and target sequences
    # Pad source and target sequences with 0 (index of <pad>)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_padded, tgt_padded  # Return padded batches

# -------------------- Vocab -------------------- #
# Build vocabulary from list of words
def build_vocab(data):
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}  # Special tokens
    idx = 4
    for word in data:
        if isinstance(word, str):  # Only process strings
            for char in word:  # Iterate over characters
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
    return vocab  # Return final vocabulary dictionary

# -------------------- Model -------------------- #
# Seq2Seq model definition
class Seq2Seq(nn.Module):
    def __init__(self, config, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.config = config
        # Embedding layers for source and target
        self.embedding_src = nn.Embedding(src_vocab_size, config.embedding_dim, padding_idx=0)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, config.embedding_dim, padding_idx=0)

        # Select RNN cell type (RNN/GRU/LSTM)
        rnn_cell = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[config.cell_type]
        # Encoder RNN
        self.encoder = rnn_cell(config.embedding_dim, config.hidden_size, config.encoder_layers,
                                dropout=config.dropout, batch_first=True)
        # Decoder RNN
        self.decoder = rnn_cell(config.embedding_dim, config.hidden_size, config.decoder_layers,
                                dropout=config.dropout, batch_first=True)
        # Final linear layer to get vocab probabilities
        self.fc_out = nn.Linear(config.hidden_size, tgt_vocab_size)

    # Forward pass through the model
    def forward(self, src, tgt):
        embedded_src = self.embedding_src(src)  # Embed source
        embedded_tgt = self.embedding_tgt(tgt)  # Embed target

        _, hidden = self.encoder(embedded_src)  # Encode source sequence

        # Ensure encoder hidden state matches decoder layer count
        def expand_hidden(h_enc, required_layers):
            num_enc_layers = h_enc.size(0)
            if num_enc_layers < required_layers:
                extra = torch.zeros(
                    required_layers - num_enc_layers,
                    h_enc.size(1),
                    h_enc.size(2),
                    device=h_enc.device,
                    dtype=h_enc.dtype
                )
                h_enc = torch.cat([h_enc, extra], dim=0)
            else:
                h_enc = h_enc[-required_layers:]
            return h_enc

        if isinstance(hidden, tuple):  # If using LSTM (h, c)
            h, c = hidden
            h = expand_hidden(h, self.config.decoder_layers)
            c = expand_hidden(c, self.config.decoder_layers)
            decoder_output, _ = self.decoder(embedded_tgt, (h, c))  # Decode
        else:
            hidden = expand_hidden(hidden, self.config.decoder_layers)
            decoder_output, _ = self.decoder(embedded_tgt, hidden)  # Decode

        output = self.fc_out(decoder_output)  # Output vocab scores
        return output

# -------------------- Accuracy -------------------- #
# Calculate token-level accuracy
def calculate_accuracy(output, target, pad_idx):
    preds = output.argmax(2)  # Get predicted tokens
    mask = (target != pad_idx)  # Ignore padding
    correct = (preds == target) & mask  # Count correct predictions
    return correct.sum().item() / mask.sum().item()  # Accuracy

# Calculate full-word accuracy
def compute_word_accuracy(output, target, tgt_index_to_token, pad_idx):
    preds = output.argmax(dim=2)
    correct = 0
    total = 0
    for pred_seq, tgt_seq in zip(preds, target):
        # Convert predicted indices to tokens, ignore padding
        pred_tokens = [tgt_index_to_token[idx.item()] for idx in pred_seq if idx.item() != pad_idx]
        tgt_tokens = [tgt_index_to_token[idx.item()] for idx in tgt_seq if idx.item() != pad_idx]
        # Cut sequences at <eos>
        if '<eos>' in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index('<eos>')]
        if '<eos>' in tgt_tokens:
            tgt_tokens = tgt_tokens[:tgt_tokens.index('<eos>')]
        if pred_tokens == tgt_tokens:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

# -------------------- Training -------------------- #
# Training loop
def train(model, dataloader, optimizer, criterion, tgt_pad_idx, tgt_index_to_token):
    model.train()
    total_loss, total_acc, total_word_acc = 0, 0, 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()  # Reset gradients
        output = model(src, tgt[:, :-1])  # Predict next token given previous
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))  # Compute loss
        acc = calculate_accuracy(output, tgt[:, 1:], tgt_pad_idx)  # Token accuracy
        word_acc = compute_word_accuracy(output, tgt[:, 1:], tgt_index_to_token, tgt_pad_idx)  # Word accuracy
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()
        total_acc += acc
        total_word_acc += word_acc
    return total_loss / len(dataloader), total_acc / len(dataloader), total_word_acc / len(dataloader)

# Evaluation loop (no gradient updates)
def evaluate(model, dataloader, criterion, tgt_pad_idx, tgt_index_to_token):
    model.eval()
    total_loss, total_acc, total_word_acc = 0, 0, 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            acc = calculate_accuracy(output, tgt[:, 1:], tgt_pad_idx)
            word_acc = compute_word_accuracy(output, tgt[:, 1:], tgt_index_to_token, tgt_pad_idx)
            total_loss += loss.item()
            total_acc += acc
            total_word_acc += word_acc
    return total_loss / len(dataloader), total_acc / len(dataloader), total_word_acc / len(dataloader)

# -------------------- Main -------------------- #
# Main function to run training
def main(args):
    wandb.login(key="580e769ee2f34eafdded556ce52aaf31c265ad3b")
    wandb.init(project="DL_A3", config=args)  # Initialize wandb
    config = wandb.config  # Get config

    # Read and expand training dataset
    train_df = pd.read_csv(args.train_path, sep="\t", header=None, names=["tgt", "src", "freq"])
    train_df = train_df.loc[train_df.index.repeat(train_df['freq'])].reset_index(drop=True)
    # Read development dataset
    dev_df = pd.read_csv(args.dev_path, sep="\t", header=None, names=["tgt", "src", "freq"])
    train_df['src'] = train_df['src'].astype(str)
    train_df['tgt'] = train_df['tgt'].astype(str)

    # Build vocabularies
    src_vocab = build_vocab(train_df['src'])
    tgt_vocab = build_vocab(train_df['tgt'])
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}

    # Prepare data as list of tuples
    train_data = list(zip(train_df['src'], train_df['tgt']))
    dev_data = list(zip(dev_df['src'], dev_df['tgt']))

    # Create dataset and dataloader
    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, optimizer and loss function
    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc * 100,
            "val_accuracy": val_acc * 100,
            "train_word_accuracy": train_word_acc * 100,
            "val_word_accuracy": val_word_acc * 100
        })

# -------------------- Entry Point -------------------- #
# Entry point to parse arguments and run training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Seq2Seq transliteration model")

    # Add hyperparameter arguments
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--encoder_layers', type=int, default=1)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cell_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='LSTM')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    # Add file path arguments
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)

    args = parser.parse_args()  # Parse command-line args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device
    main(args)  # Run main training function
