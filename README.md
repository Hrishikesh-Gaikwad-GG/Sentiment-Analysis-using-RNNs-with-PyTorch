# IMDb Sentiment Analysis with PyTorch RNNs

This project explores **sentiment analysis on IMDb movie reviews** using various **recurrent neural network (RNN) architectures** implemented in **PyTorch**.  
The goal is to classify movie reviews as **positive** or **negative** and systematically compare different RNN-based models.

---

## 📌 Project Overview

- **Task**: Binary sentiment classification (Positive / Negative)
- **Dataset**: IMDb Movie Reviews (50,000 reviews)
- **Framework**: PyTorch
- **Problem Type**: Many-to-one sequence classification

The project focuses on understanding how different recurrent architectures handle long textual sequences and how evaluation strategies affect model performance.

---

## 🧠 Models Implemented

The following models were trained and compared using the same preprocessing pipeline and data splits:

- Simple RNN (baseline)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional LSTM (BiLSTM)

The **BiLSTM** model was selected as the final model due to its superior performance.

---

## 🗂 Dataset Details

- **Total samples**: 50,000
- **Training samples**: 25,000
- **Test samples**: 25,000
- **Classes**: Positive, Negative
- **Class balance**: Balanced (50/50)

The dataset contains long, variable-length reviews, making it well-suited for evaluating sequence models.

---

## ⚙️ Preprocessing Pipeline

- Lowercasing text
- Removing HTML tags
- Tokenisation using a custom vocabulary
- Padding and truncation to fixed sequence length
- `<PAD>` and `<OOV>` token handling

The same preprocessing pipeline is applied consistently across all models and during inference.

---

## 🛡 Regularization Techniques

To mitigate overfitting, especially in LSTM-based models, the following techniques were applied:

- Dropout after the embedding layer
- Dropout in recurrent layers
- Dropout before classification layer
- Weight decay (L2 regularisation)

---

## 📊 Evaluation Strategy

### During Training
- Loss: Binary Cross-Entropy
- Metric: Accuracy (threshold = 0.5)

### Final Evaluation (BiLSTM)
- **ROC–AUC** for threshold-independent evaluation
- Optimal threshold selection using ROC analysis
- **Confusion Matrix** for error analysis

ROC–AUC was preferred over accuracy alone to better assess ranking performance.

---

## 🧪 Inference on Custom Input

The trained BiLSTM model supports inference on custom text input.

- Uses the same preprocessing and vocabulary as training
- Outputs sentiment label and confidence score
- Demonstrates real-world usability

---

## 🏆 Key Results

- Simple RNN struggles with long-term dependencies
- LSTM and GRU significantly outperform Simple RNN
- GRU offers a strong balance between speed and accuracy
- **BiLSTM achieves the best overall performance** by leveraging bidirectional context
- Proper regularisation and ROC-based evaluation improve robustness
