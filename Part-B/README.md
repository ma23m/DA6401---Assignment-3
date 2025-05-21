# Seq2Seq Model for Transliteration with Attention

This repository provides a PyTorch-based implementation of a **Sequence-to-Sequence (Seq2Seq)** model enhanced with an **attention mechanism** for transliterating Latin-script words into their Bangla-script equivalents.

---

##  Dataset: Dakshina (Bengali)

We use the **Dakshina Dataset (Ben)** which includes:
- **Input**: Latin transliterated words  
- **Target**: Corresponding Bangla script transliterations  

###  Dataset Splits:
- **Training Set**: Used to train the model  
- **Validation Set**: Used for tuning and early stopping  
- **Test Set**: Used to evaluate final performance

---

##  Model Overview

###  Encoder
- **Embedding Layer**: Converts input tokens into fixed-size dense vectors  
- **LSTM**: Captures dependencies from both past and future input directions  
- **Dropout Layer**: Helps prevent overfitting  
- **Context Vector**: Summary representation of the input sequence  

###  Decoder
- **Embedding Layer**: Converts decoder input tokens into dense vectors  
- **LSTM Layer**: Generates output sequence based on context and previous predictions  
- **Attention Mechanism**: Focuses on relevant parts of the input at each decoding step to improve accuracy
- **Beam Search**:  Decoding strategy used during inference to improve the quality of generated sequences by exploring multiple candidate outputs
---

##  Model Training

- **Loss Function**: Cross-entropy loss  
- **Optimizer**: Adam  
- **Objective**: Minimize the difference between predicted and actual transliterations  
- **Training Technique**: Teacher forcing is used during training for faster convergence

---

##  Hyperparameter Tuning

Hyperparameters are optimized using **Bayesian Optimization**, including:
- Embedding dimension  
- Number of LSTM layers  
- Hidden size  
- Dropout rate  
- Number of epochs  

**Tracking**: All experiments are logged using **Weights & Biases (Wandb)** for reproducibility and analysis.

---

##  Model Evaluation

After training, the model is tested on the held-out test set:
- **Validation Accuracy**:`39.04%`
- **Test Accuracy**: `39.26%`  
- **Evaluation Metrics**: Cross-entropy loss, character-level accuracy  
- **Qualitative Analysis**: Includes sample transliterations to assess output quality



###  Command-Line Arguments

| *Argument*         | *Type* | *Default* | *Description* |
|----------------------|----------|-------------|------------------|
| --train_path       | str    | Required  | Path to training file (TSV format). |
| --dev_path         | str    | Required  | Path to development/validation file (TSV format). |
| --embedding_dim    | int    | 32,64,128,256        | Size of word embeddings. |
| --hidden_size      | int    | 32,64,128,256       | Hidden size of RNN/GRU/LSTM layers. |
| --cell_type        | str    | 'RNN','GRU','LSTM'     | Type of RNN cell: 'RNN', 'GRU', or 'LSTM'. |
| --encoder_layers   | int    | 1,2,3         | Number of layers in the encoder. |
| --decoder_layers   | int    | 1,2,3         | Number of layers in the decoder. |
| --dropout          | float  | 0.2,0.3       | Dropout rate for encoder/decoder. |
| --batch_size       | int    | 32,64      | Batch size for training. |
| --epochs           | int    | 5,7,10,15,20       | Total number of training epochs. |


---

##  How to Run the Code

  
##  Example Usage

- Use the following command to start training for one combination:

```
python train_partB.py \
  --train_path data/train.tsv \
  --dev_path data/dev.tsv \
  --embedding_dim 64 \
  --hidden_size 128 \
  --cell_type LSTM \
  --encoder_layers 1 \
  --decoder_layers 1 \
  --dropout 0.2 \
  --batch_size 128 \
  --epochs 30
 ```

  ---
- To get the prediction from attention model please run the following command:

         ```
  python beam_sweep_attention.py
  ```

- To get the attention heatmap please run the follwing command

          ```
  attentionHeatmap.py
  ```

- To get the attention heatmap with connectivity for the Q6 please run the follwing command

        ```
  python attention_connectivity.py
  ```

## Note:

- The model is trained using a  ```CUDA-enabled GPU``` (available on ```Kaggle```), since training model requires a lot of computation and would be slow on a regular ```CPU```.



