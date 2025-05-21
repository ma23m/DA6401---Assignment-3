
# Seq2Seq Transliteration without Attention 

This repository contains the PyTorch implementation of a **sequence-to-sequence (Seq2Seq)** model designed for **transliteration from Latin script to Bangla script**. The model is trained on the **Dakshina dataset** without using an attention mechanism.

---

##  Dataset Description

The model is trained on the **Dakshina dataset**, which contains:
- Input: Latin-script words
- Output: Corresponding Bangla-script transliterations

The dataset is split into:
- **Training set**: Used to train the model  
- **Validation set**: Used for hyperparameter tuning and performance monitoring  
- **Test set**: Used to evaluate the final model on unseen data

---

##  Model Design

###  Encoder
- **Embedding Layer**: Converts each input token into a dense vector  
- **Bidirectional RNN/GRU/LSTM**: Captures both forward and backward temporal dependencies  
- **Dropout Layer**: Helps prevent overfitting  
- **Context Vector**: Fixed-size vector summarizing the encoded input sequence

###  Decoder
- **Embedding Layer**: Converts output tokens into dense vectors  
- **RNN/GRU/LSTM**: Generates the output sequence using the context vector and previous outputs  
- **Attention**: Not used in this model, but can enhance performance by focusing on specific input parts
- **Beam Search**: Used during inference to improve decoding by exploring multiple candidate sequences
---

##  Training Procedure

- **Optimizer**: Adam  
- **Loss Function**: Cross-entropy loss  
- **Objective**: Minimize prediction error between model outputs and true transliterations  
- **Teacher Forcing**: Applied during training for faster convergence

---

##  Hyperparameter Tuning

Hyperparameters are optimized using **Bayesian Optimization**:
- Embedding dimension  
- Hidden layer size  
- Number of layers  
- Dropout rate  


All experiments and metrics are tracked with **Weights & Biases (Wandb)** for reproducibility and visualization.

---

##  Evaluation

After training, the model is evaluated on the test set:
- **Validation Accuracy**:`36.06%`
- **Test Accuracy**:`35.62%`  
- **Metrics**: Loss and accuracy  
- **Qualitative Analysis**: Sample transliterations (Latin → Bangla) are included for analysis.

---
##  Command-Line Arguments

| Argument             | Type    | Default | Description                                                                 |
|----------------------|---------|---------|-----------------------------------------------------------------------------|
| --train_path       | str   | —       | Path to the training .tsv file (required)                                |
| --dev_path         | str   | —       | Path to the development/validation .tsv file (required)                  |
| --embedding_dim    | int   | 128   | Dimension of the embedding vectors                                         |
| --hidden_size      | int   | 256   | Hidden size of the RNN units                                               |
| --encoder_layers   | int   | 1     | Number of layers in the encoder                                            |
| --decoder_layers   | int   | 1     | Number of layers in the decoder                                            |
| --dropout          | float | 0.3   | Dropout probability between RNN layers                                     |
| --cell_type        | str   | 'LSTM'| Type of RNN cell (RNN, GRU, or LSTM)                                 |
| --epochs           | int   | 10    | Number of training epochs                                                  |
| --batch_size       | int   | 32    | Number of samples per training batch                                       |

---

## How to Run the Code

- To get the prediction from attention model please run the following command:

   ```python beam_sweep_attention.py```

- To get the attention heatmap please run the follwing command

   ```attentionHeatmap.py```

- To get the attention heatmap with connectivity for the Q6 please run the follwing command

   ```python attention_connectivity.py```


###  Run the training script

Use the following command to start training:


```python train_partA.py \
  --train_path /path/to/bn.translit.sampled.train.tsv \
  --dev_path /path/to/bn.translit.sampled.dev.tsv \
  --embedding_dim 256 \
  --hidden_size 512 \
  --encoder_layers 2 \
  --decoder_layers 2 \
  --dropout 0.3 \
  --cell_type LSTM \
  --epochs 15 \
  --batch_size 64
```


---

## Note:

The model is trained using a  ```CUDA-enabled GPU``` (available on ```Kaggle```), since training model requires a lot of computation and would be slow on a regular ```CPU```.

---


