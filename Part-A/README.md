
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
- Learning rate  

All experiments and metrics are tracked with **Weights & Biases (Wandb)** for reproducibility and visualization.

---

##  Evaluation

After training, the model is evaluated on the test set:
- **Validation Accuracy**:`36.06%`
- **Test Accuracy**:`35.62%`  
- **Metrics**: Loss and accuracy  
- **Qualitative Analysis**: Sample transliterations (Latin → Bangla) are included for analysis.

---



