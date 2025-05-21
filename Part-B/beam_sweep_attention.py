import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import csv

# Import modules
from train_PartB import Seq2Seq, build_vocab, TransliterationDataset, collate_fn, train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_with_beam_search(model, src_seq, src_vocab, tgt_vocab, beam_width=3, max_len=30):
    model.eval()
    sos_token = tgt_vocab['<sos>']
    eos_token = tgt_vocab['<eos>']
    pad_token = tgt_vocab['<pad>']
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}

    with torch.no_grad():
        src_indices = [src_vocab['<sos>']] + [src_vocab.get(c, src_vocab['<unk>']) for c in src_seq] + [src_vocab['<eos>']]
        src_tensor = torch.tensor(src_indices, device=device).unsqueeze(0)
        embedded_src = model.embedding_src(src_tensor)

        encoder_outputs, hidden = model.encoder(embedded_src)
        if isinstance(hidden, tuple):
            h, c = hidden
            h = model._expand_hidden(h, model.config['decoder_layers'])
            c = model._expand_hidden(c, model.config['decoder_layers'])
            decoder_hidden = (h, c)
        else:
            hidden = model._expand_hidden(hidden, model.config['decoder_layers'])
            decoder_hidden = hidden

        beams = [(torch.tensor([sos_token], device=device), 0.0, decoder_hidden)]

        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden_state in beams:
                if seq[-1].item() == eos_token:
                    new_beams.append((seq, score, hidden_state))
                    continue

                embedded = model.embedding_tgt(seq[-1].unsqueeze(0).unsqueeze(0))

                h_last = hidden_state[0][-1] if isinstance(hidden_state, tuple) else hidden_state[-1]

                attn_weights = model.attention(h_last, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

                decoder_input = torch.cat((embedded, context), dim=2)

                output, new_hidden = model.decoder_rnn(decoder_input, hidden_state)
                logits = model.fc_out(output.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + log_prob.item()
                    new_beams.append((new_seq, new_score, new_hidden))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = beams[0][0]
        return ''.join([
            tgt_index_to_token[token.item()] for token in best_seq[1:]
            if token.item() not in [eos_token, pad_token]
        ])

def sweep_train_pred():
    wandb.init()
    config = wandb.config

    train_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    train_df = train_df.loc[train_df.index.repeat(train_df['freq'])].reset_index(drop=True)
    dev_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    test_df = pd.read_csv('/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv', sep="\t", header=None, names=["tgt", "src", "freq"])

    src_vocab = build_vocab(train_df['src'])
    tgt_vocab = build_vocab(train_df['tgt'])
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    train_data = list(zip(train_df['src'], train_df['tgt']))
    dev_data = list(zip(dev_df['src'], dev_df['tgt']))
    test_data = list(zip(test_df['src'], test_df['tgt']))

    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    for epoch in range(config['epochs']):
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        test_loss, test_acc, test_word_acc = evaluate(model, test_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)

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

    beam_width = config.get('beam_width', 3)
    print('beam_width', beam_width)

    results = []
    for sample_src, actual_tgt in test_data[:9229]:
        pred_seq = predict_with_beam_search(model, sample_src, src_vocab, tgt_vocab, beam_width=beam_width)
        pred_tokens = [c for c in pred_seq if c not in ['<sos>', '<pad>', '<eos>']]
        pred_str = ''.join(pred_tokens)
        print(f"Input:      {sample_src}")
        print(f"Actual:     {actual_tgt}")
        print(f"Prediction: {pred_str}")
        print("-" * 30)

        results.append({
            "Input": sample_src,
            "Actual": actual_tgt,
            "Prediction": pred_str
        })

    output_csv_path = "beam_search_predictions_attention.csv"
    with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Input", "Actual", "Prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved predictions to {output_csv_path}")


# -------------------- Run -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Epoch=13,hidden =128,encoder =2,decoder=3,drop=0.3,embeded=64,LSTM
sweep_config = {
    'method': 'random',
    'name': 'DakshinaSweepForAttention_Pred',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embedding_dim': {'values': [64]},
        'hidden_size': {'values': [128]},
        'encoder_layers': {'values': [2]},
        'decoder_layers': {'values': [3]},
        'cell_type': {'values': ['LSTM']},
        'dropout': {'values': [0.3]},
        'epochs': {'values': [13]},
        'beam_width': {'values': [3]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DL_A3_Part_B")
wandb.agent(sweep_id, function=sweep_train_pred, count = 1)