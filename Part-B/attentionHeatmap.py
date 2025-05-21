import wandb  # Import the Weights & Biases library for logging and visualization
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for heatmap plotting
from matplotlib import font_manager as fm  # Import font manager to load custom fonts
from train_PartB import Seq2Seq, build_vocab, TransliterationDataset, collate_fn, train, evaluate

# Load Bangla font from the specified path
bangla_font_path = "/kaggle/input/tiro-bn/TiroBangla-Regular.ttf"
bangla_font = fm.FontProperties(fname=bangla_font_path)

# Function to plot attention heatmap for a single sample
def plot_attention(attentions, src_tokens, tgt_tokens, index, ax):
    sns.heatmap(attentions.cpu().detach().numpy(),  # Plot attention weights as a heatmap
                xticklabels=src_tokens,  # Set x-axis labels as source tokens
                yticklabels=tgt_tokens,  # Set y-axis labels as target tokens
                cmap="viridis",  # Use the 'viridis' color map
                cbar=False,  # Hide the color bar
                ax=ax)  # Plot on the given Axes object

    ax.set_xlabel("Source", fontproperties=bangla_font)  # Label x-axis with Bangla font
    ax.set_ylabel("Target", fontproperties=bangla_font)  # Label y-axis with Bangla font
    ax.set_title(f"Sample {index}", fontproperties=bangla_font)  # Set the title using Bangla font

    # Set custom font for tick labels
    ax.set_yticklabels(tgt_tokens, fontproperties=bangla_font, rotation=0)  # Y-axis ticks (no rotation)
    ax.set_xticklabels(src_tokens, fontproperties=bangla_font, rotation=45)  # X-axis ticks (angled)

# Function to visualize a grid of attention heatmaps
def visualize_attention_grid(model, dataloader, src_vocab, tgt_vocab, device):
    # Create reverse vocabularies to map indices to tokens
    src_index_to_token = {v: k for k, v in src_vocab.items()}
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    model.eval()  # Set model to evaluation mode

    n = 0  # Counter for the number of samples plotted
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Create a 3x3 grid of subplots
    axes = axes.flatten()  # Flatten 2D array of axes to 1D for easy indexing

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for src_batch, tgt_batch in dataloader:  # Loop over data batches
            src_batch = src_batch.to(device)  # Move source batch to device (CPU/GPU)
            tgt_batch = tgt_batch.to(device)  # Move target batch to device
            output, attn_weights = model(src_batch, tgt_batch[:, :-1], return_attn=True)  # Forward pass with attention

            for i in range(src_batch.size(0)):  # Loop over samples in batch
                # Convert source and target indices to tokens, skipping padding
                src_seq = [src_index_to_token[idx.item()] for idx in src_batch[i] if idx.item() != src_vocab['<pad>']]
                tgt_seq = [tgt_index_to_token[idx.item()] for idx in tgt_batch[i, 1:] if idx.item() != tgt_vocab['<pad>']]

                # Select attention weights for this sample (target_len x source_len)
                attn = attn_weights[i, :len(tgt_seq), :len(src_seq)]
                # Plot the attention heatmap on the next subplot
                plot_attention(attn, src_seq, tgt_seq, n+1, axes[n])
                n += 1  # Increment the sample counter

                if n == 9:  # Stop after plotting 9 samples
                    plt.tight_layout()  # Adjust subplot layout to avoid overlap
                    wandb.log({"Attention Heatmap Grid": wandb.Image(fig)})  # Log the figure to Weights & Biases
                    plt.show()  # Display the plot
                    return  # Exit the function after showing the grid

# -------------------- Sweep -------------------- #
def sweep_train_heatmap():
    # Initialize a new Weights & Biases run
    wandb.init()
    
    # Get sweep configuration parameters
    config = wandb.config

    # Load training data with Bangla targets and Latin source words
    train_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    
    # Repeat rows in training data based on frequency value
    train_df = train_df.loc[train_df.index.repeat(train_df['freq'])].reset_index(drop=True)
    
    # Load development and test data
    dev_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    test_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    
    # Convert source and target columns to string type
    train_df['src'] = train_df['src'].astype(str)
    train_df['tgt'] = train_df['tgt'].astype(str)

    # Build vocabularies from training source and target text
    src_vocab = build_vocab(train_df['src'])
    tgt_vocab = build_vocab(train_df['tgt'])

    # Print source and target vocabularies
    print(src_vocab)
    print(tgt_vocab)

    # Create index-to-token mappings for the target vocab
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    # Create data tuples of (source, target) for train/dev/test
    train_data = list(zip(train_df['src'], train_df['tgt']))
    dev_data = list(zip(dev_df['src'], dev_df['tgt']))
    test_data = list(zip(test_df['src'], test_df['tgt']))

    # Create dataset objects from data tuples
    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

    # Create data loaders with batch size and custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the Seq2Seq model with sweep configuration
    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)

    # Define optimizer and loss function (ignoring pad tokens in loss)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training loop for each epoch
    for epoch in range(config['epochs']):
        # Train the model and evaluate on train set
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        
        # Evaluate the model on dev and test sets
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
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

        # Log metrics to Weights & Biases dashboard
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
    
    # Visualize attention weights for test samples using heatmaps
    visualize_attention_grid(model, test_loader, src_vocab, tgt_vocab, device)

# -------------------- Run -------------------- #

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep configuration for W&B hyperparameter tuning
# Sweep includes embedding, hidden size, number of layers, dropout, and LSTM cell
sweep_config = {
    'method': 'random',  # Randomly sample hyperparameters
    'name': 'DakshinaSweepForAttention_heatmapFinal123',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},  # Optimize for validation accuracy
    'parameters': {
        'embedding_dim': {'values': [64]},
        'hidden_size': {'values': [128]},
        'encoder_layers': {'values': [2]},
        'decoder_layers': {'values': [3]},
        'cell_type': {'values': ['LSTM']},
        'dropout': {'values': [0.3]},
        'epochs': {'values': [13]},
    }
}

# Create the sweep and get sweep ID
sweep_id = wandb.sweep(sweep_config, project="DL_A3_Part_B")

# Start sweep agent to run training function with sampled config
wandb.agent(sweep_id, function=sweep_train_heatmap, count = 1)
