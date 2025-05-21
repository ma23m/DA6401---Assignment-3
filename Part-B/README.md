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
- **Bidirectional LSTM**: Captures dependencies from both past and future input directions  
- **Dropout Layer**: Helps prevent overfitting  
- **Context Vector**: Summary representation of the input sequence  

###  Decoder
- **Embedding Layer**: Converts decoder input tokens into dense vectors  
- **LSTM Layer**: Generates output sequence based on context and previous predictions  
- **Attention Mechanism**: Focuses on relevant parts of the input at each decoding step to improve accuracy
- **Beam Search**: Optional decoding strategy used during inference to improve the quality of generated sequences by exploring multiple candidate outputs
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
- Learning rate  
- Number of epochs  

**Tracking**: All experiments are logged using **Weights & Biases (Wandb)** for reproducibility and analysis.

---

##  Model Evaluation

After training, the model is tested on the held-out test set:
-**Validation Accuracy**:`39.04%`
- **Test Accuracy**: `39.26%`  
- **Evaluation Metrics**: Cross-entropy loss, character-level accuracy  
- **Qualitative Analysis**: Includes sample transliterations to assess output quality

---

