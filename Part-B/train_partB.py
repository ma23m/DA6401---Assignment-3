import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -------------------- Dataset -------------------- #
class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        # Convert source and target strings to list of indices with special tokens
        src_ids = [self.src_vocab['<sos>']] + [self.src_vocab.get(c, self.src_vocab['<unk>']) for c in src] + [self.src_vocab['<eos>']]
        tgt_ids = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(c, self.tgt_vocab['<unk>']) for c in tgt] + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_seqs, tgt_seqs = zip(*batch)
    # Pad sequences to equal length within batch
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_padded, tgt_padded

# -------------------- Vocab -------------------- #
def build_vocab(data):
    # Initialize vocab with special tokens
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    # Add each unique character to vocab
    for word in data:
        if isinstance(word, str):
            for char in word:
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
    return vocab

# -------------------- Attention -------------------- #
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size, seq_len, _ = encoder_outputs.size()

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden)
        attention_scores = self.v(energy).squeeze(2)  # (batch, seq_len)
    
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
    
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        return attention_weights    

# -------------------- Seq2Seq Model -------------------- #
class Seq2Seq(nn.Module):
    def __init__(self, config, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.attention = Attention(config.hidden_size)
        self.embedding_src = nn.Embedding(src_vocab_size, config.embedding_dim, padding_idx=0)
        self.embedding_tgt = nn.Embedding(tgt_vocab_size, config.embedding_dim, padding_idx=0)

        rnn_cell = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[config.cell_type]
        self.encoder = rnn_cell(config.embedding_dim, config.hidden_size, config.encoder_layers, dropout=config.dropout, batch_first=True)
        self.decoder_rnn = rnn_cell(config.embedding_dim + config.hidden_size, config.hidden_size, config.decoder_layers, dropout=config.dropout, batch_first=True)
        
        self.fc_out = nn.Linear(config.hidden_size, tgt_vocab_size)

    def forward(self, src, tgt, return_attn=False):
        embedded_src = self.embedding_src(src)
        embedded_tgt = self.embedding_tgt(tgt)
    
        encoder_outputs, hidden = self.encoder(embedded_src)
    
        if isinstance(hidden, tuple):  # LSTM: hidden = (h, c)
            h, c = hidden
            h = self._expand_hidden(h, self.decoder_rnn.num_layers)
            c = self._expand_hidden(c, self.decoder_rnn.num_layers)
            decoder_hidden = (h, c)
        else:
            hidden = self._expand_hidden(hidden, self.decoder_rnn.num_layers)
            decoder_hidden = hidden
    
        batch_size, tgt_len, _ = embedded_tgt.size()
        decoder_outputs = []
        all_attn_weights = []
    
        for t in range(tgt_len):
            tgt_t = embedded_tgt[:, t, :].unsqueeze(1)
    
            if isinstance(decoder_hidden, tuple):
                h_t = decoder_hidden[0][-1]
            else:
                h_t = decoder_hidden[-1]
    
            attn_weights = self.attention(h_t, encoder_outputs)  # (B, S)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
    
            decoder_input = torch.cat((tgt_t, attn_applied), dim=2)
            output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
            decoder_outputs.append(output)
            all_attn_weights.append(attn_weights)
    
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        output = self.fc_out(decoder_outputs)
    
        if return_attn:
            attn_tensor = torch.stack(all_attn_weights, dim=1)  # (B, T, S)
            return output, attn_tensor
        return output

    def _expand_hidden(self, h_enc, required_layers):
        num_enc_layers = h_enc.size(0)
        if num_enc_layers < required_layers:
            extra = torch.zeros(required_layers - num_enc_layers, h_enc.size(1), h_enc.size(2), device=h_enc.device)
            h_enc = torch.cat([h_enc, extra], dim=0)
        else:
            h_enc = h_enc[-required_layers:]
        return h_enc

# -------------------- Accuracy -------------------- #
def calculate_accuracy(output, target, pad_idx):
    preds = output.argmax(2)
    mask = (target != pad_idx)
    correct = (preds == target) & mask
    return correct.sum().item() / mask.sum().item()

def compute_word_accuracy(output, target, tgt_index_to_token, pad_idx):
    preds = output.argmax(dim=2)
    correct = 0
    total = 0

    for pred_seq, tgt_seq in zip(preds, target):
        pred_tokens = [tgt_index_to_token[idx.item()] for idx in pred_seq if idx.item() != pad_idx]
        tgt_tokens = [tgt_index_to_token[idx.item()] for idx in tgt_seq if idx.item() != pad_idx]

        if '<eos>' in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index('<eos>')]
        if '<eos>' in tgt_tokens:
            tgt_tokens = tgt_tokens[:tgt_tokens.index('<eos>')]

        if pred_tokens == tgt_tokens:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

# -------------------- Training & Evaluation -------------------- #
def train(model, dataloader, optimizer, criterion, tgt_pad_idx, tgt_index_to_token):
    model.train()
    total_loss, total_acc, total_word_acc = 0, 0, 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        acc = calculate_accuracy(output, tgt[:, 1:], tgt_pad_idx)
        word_acc = compute_word_accuracy(output, tgt[:, 1:], tgt_index_to_token, tgt_pad_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc
        total_word_acc += word_acc
    return total_loss / len(dataloader), total_acc / len(dataloader), total_word_acc / len(dataloader)

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

# -------------------- Main run function -------------------- #
def main(args):
    wandb.login(key="580e769ee2f34eafdded556ce52aaf31c265ad3b")
    wandb.init(project="DL_A3_Part_B", config=vars(args))
    config = wandb.config

    # Load data from TSV files
    train_df = pd.read_csv(args.train_path, sep="\t", header=None, names=["tgt", "src", "freq"])
    dev_df = pd.read_csv(args.dev_path, sep="\t", header=None, names=["tgt", "src", "freq"])

    train_data = list(zip(train_df["src"], train_df["tgt"]))
    dev_data = list(zip(dev_df["src"], dev_df["tgt"]))

    # Build vocabs from train source and target
    src_vocab = build_vocab([src for src, tgt in train_data])
    tgt_vocab = build_vocab([tgt for src, tgt in train_data])
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}

    # Create datasets and loaders
    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_word_accuracy": train_word_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_word_accuracy": val_word_acc,
            "epoch": epoch + 1,
        })

        print(f"Epoch {epoch+1}/{config.epochs} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Train Word Acc: {train_word_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Word Acc: {val_word_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter > config.patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Seq Transliteration Model")

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--cell_type", type=str, choices=['RNN', 'GRU', 'LSTM'], default='RNN')
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--decoder_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_path", type=str, required=True, help="Path to train.tsv")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to dev.tsv")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
