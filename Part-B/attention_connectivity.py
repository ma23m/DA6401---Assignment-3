# Import necessary libraries
import wandb                              # For experiment tracking and visualization
import matplotlib.pyplot as plt          # For plotting visualizations
import seaborn as sns                    # For generating heatmaps and styled plots
from matplotlib import font_manager as fm  # For custom font support (used for Bangla font)
import torch                             # PyTorch for deep learning
import imageio.v2                        # For reading images to generate video
import wandb                             # Duplicate import (can be removed)
import os                                # For interacting with the file system
# Import custom functions and classes from training module
from train_PartB import Seq2Seq, build_vocab, TransliterationDataset, collate_fn, train, evaluate

# Load a Bangla font from the Kaggle input directory
bangla_font_path = "/kaggle/input/tiro-bn/TiroBangla-Regular.ttf"
bangla_font = fm.FontProperties(fname=bangla_font_path)

# Function to plot the attention matrix for a single example
def plot_attention(attentions, src_tokens, tgt_tokens, index, ax):
    attn_np = attentions.cpu().detach().numpy()  # Convert attention tensor to NumPy array

    # Create heatmap from attention matrix
    sns.heatmap(attn_np,
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap="YlGnBu",  # Blue-green color map
                cbar=True,
                ax=ax,
                linewidths=0.3,
                linecolor='gray',
                annot=False)

    # Set axis labels and title with Bangla font
    ax.set_xlabel("Input Characters", fontproperties=bangla_font)
    ax.set_ylabel("Output Characters", fontproperties=bangla_font)
    ax.set_title(f"Sample {index}", fontproperties=bangla_font, fontsize=14)

    # Set tick labels with proper font and rotation
    ax.set_yticklabels(tgt_tokens, fontproperties=bangla_font, rotation=0, fontsize=10)
    ax.set_xticklabels(src_tokens, fontproperties=bangla_font, rotation=90, fontsize=10)

    # Highlight the most attended source token for each target token
    for i, row in enumerate(attn_np):
        j = row.argmax()
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1.5))

# Function to visualize attention for a few examples from the dataloader
def visualize_attention_grid(model, dataloader, src_vocab, tgt_vocab, device, epoch):
    # Create index-to-token maps
    src_index_to_token = {v: k for k, v in src_vocab.items()}
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    model.eval()  # Set model to evaluation mode

    n = 0
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Create a 3x3 grid of subplots
    axes = axes.flatten()

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Forward pass with attention returned
            output, attn_weights = model(src_batch, tgt_batch[:, :-1], return_attn=True)

            for i in range(src_batch.size(0)):
                # Decode token indices to actual characters
                src_seq = [src_index_to_token[idx.item()] for idx in src_batch[i] if idx.item() != src_vocab['<pad>']]
                tgt_seq = [tgt_index_to_token[idx.item()] for idx in tgt_batch[i, 1:] if idx.item() != tgt_vocab['<pad>']]

                # Get the attention matrix for this sample
                attn = attn_weights[i, :len(tgt_seq), :len(src_seq)]

                # Plot the attention heatmap
                plot_attention(attn, src_seq, tgt_seq, n+1, axes[n])
                n += 1

                if n == 9:  # Limit to 9 samples
                    plt.tight_layout()
                    save_path = f"attention_epoch_{epoch:03d}.png"
                    plt.savefig(save_path)  # Save the figure
                    plt.close()
                    return  # Exit after plotting 9 examples

# -------------------- Sweep Training Function -------------------- #
def sweep_train_heatmap():
    wandb.init()  # Initialize a new wandb run
    config = wandb.config  # Get sweep configuration

    # Load training, dev, and test datasets
    train_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.train.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    train_df = train_df.loc[train_df.index.repeat(train_df['freq'])].reset_index(drop=True)
    dev_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.dev.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])
    test_df = pd.read_csv("/kaggle/input/dakshina/dakshina_dataset_v1.0/bn/lexicons/bn.translit.sampled.test.tsv", sep="\t", header=None, names=["tgt", "src", "freq"])

    # Ensure all values are strings
    train_df['src'] = train_df['src'].astype(str)
    train_df['tgt'] = train_df['tgt'].astype(str)

    # Build vocabularies for source and target
    src_vocab = build_vocab(train_df['src'])
    tgt_vocab = build_vocab(train_df['tgt'])
    print(src_vocab)
    print(tgt_vocab)
    tgt_index_to_token = {v: k for k, v in tgt_vocab.items()}
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    # Prepare data tuples
    train_data = list(zip(train_df['src'], train_df['tgt']))
    dev_data = list(zip(dev_df['src'], dev_df['tgt']))
    test_data = list(zip(test_df['src'], test_df['tgt']))

    # Create dataset and dataloader
    train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_data, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model, optimizer, and loss function
    model = Seq2Seq(config, len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])

    # Training loop
    for epoch in range(config['epochs']):
        # Train and evaluate model
        train_loss, train_acc, train_word_acc = train(model, train_loader, optimizer, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        val_loss, val_acc, val_word_acc = evaluate(model, dev_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)
        test_loss, test_acc, test_word_acc = evaluate(model, test_loader, criterion, tgt_vocab['<pad>'], tgt_index_to_token)

        # Print metrics
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

        # Visualize and save attention maps for this epoch
        visualize_attention_grid(model, test_loader, src_vocab, tgt_vocab, device, epoch)

# -------------------- Sweep Configuration & Execution -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device

# Define the hyperparameter sweep configuration
sweep_config = {
    'method': 'random',
    'name': 'DakshinaSweepForAttention_heatmap_connectivity',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
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

# Create a sweep and run it
sweep_id = wandb.sweep(sweep_config, project="DL_A3_Part_B")
wandb.agent(sweep_id, function=sweep_train_heatmap, count=1)

# -------------------- Create Video from Saved Attention Images -------------------- #
def create_attention_video(image_dir=".", video_name="attention_video.mp4"):
    images = []
    # Get all saved attention heatmap images
    filenames = sorted([f for f in os.listdir(image_dir) if f.startswith("attention_epoch") and f.endswith(".png")])
    for filename in filenames:
        img = imageio.v2.imread(os.path.join(image_dir, filename))
        images.append(img)

    video_path = os.path.join(image_dir, video_name)
    # Create a video from the list of images
    imageio.mimsave(video_path, images, fps=2)  # 2 frames per second
    return video_path

# Generate attention video
video_path = create_attention_video()

# -------------------- Log the Attention Video to wandb -------------------- #
wandb.init(project="DL_A3_Part_B", name="heatvideoFinal")
wandb.log({"Attention Evolution": wandb.Video("/kaggle/input/attention-connectivity/attention_video.mp4")})
wandb.finish()
