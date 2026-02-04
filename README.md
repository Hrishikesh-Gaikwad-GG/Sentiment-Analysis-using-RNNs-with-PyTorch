# IMDb Sentiment Analysis with PyTorch (Streamlit App)

An end-to-end **sentiment analysis web application** built with **PyTorch** and deployed on **Streamlit Cloud**.  
The app predicts whether an IMDb movie review is **Positive** or **Negative** and allows users to interactively test the model with custom and example inputs.

---

## 🚀 Live Demo

👉 **Streamlit App:**  
https://sentiment-analysis-using-rnns-with-pytorch.streamlit.app/

Features available in the app:
- Custom review input
- Predefined example reviews (positive, negative, sarcastic, mixed)
- Sentiment prediction with confidence score
- Clean, user-friendly interface

---

## 📌 Project Overview

- **Task:** Binary sentiment classification  
- **Dataset:** IMDb Movie Reviews  
- **Framework:** PyTorch  
- **Deployment:** Streamlit Cloud  
- **Model Used:** Bidirectional LSTM (BiLSTM)

The project demonstrates a complete workflow from model training to real-world deployment.

---

## 🧠 Model Summary (Brief)

Multiple recurrent architectures were explored during experimentation:
- Simple RNN
- LSTM
- GRU
- **Bidirectional LSTM (final model)**

The **BiLSTM** was selected for deployment as it provided the best overall performance by leveraging both past and future context in text.

---

## ⚙️ Preprocessing (High Level)

- Text lowercasing and HTML removal
- Tokenization with a custom vocabulary
- Handling of `<PAD>` and `<OOV>` tokens
- Fixed-length padding and truncation

The same preprocessing pipeline is used during both training and inference.

---

## 📊 Evaluation (High Level)

- Accuracy used during training for monitoring
- ROC-AUC used for final model evaluation
- Optimal threshold selection based on ROC analysis
- Confusion matrix for error inspection

---

## 🧪 Streamlit App Highlights

- Real-time inference using trained BiLSTM
- Confidence score for each prediction
- Example buttons for quick testing
- Includes sarcastic and mixed reviews to show model behavior
- CPU-only deployment (lightweight and cloud-friendly)

---
## 🗂 Project Structure
```markdown
project/
├── streamlit-app/
│   ├── app.py              # Streamlit application
│   ├── config.py           # Central configuration
│   ├── sample_reviews.py   # Example & sarcastic reviews
|   ├── requirements.txt
│   └── models/
│       ├── model.pth       # Trained BiLSTM weights
│       └── vocab.pkl       # Vocabulary mapping
├── notebook.ipynb          # Training & experimentation
└── README.md


```
---
## 🛠 Setup Instructions (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Hrishikesh-Gaikwad-GG/Sentiment-Analysis-using-RNNs-with-PyTorch.git
cd Sentiment-Analysis-using-RNNs-with-PyTorch
```

### 2️⃣ Create and activate virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```
🛠 Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

🧠 Key Takeaways

- Gated RNNs significantly outperform Simple RNNs
- GRU offers a strong speed–accuracy trade-off
- BiLSTM performs best for this task
- Sarcastic reviews remain challenging for sequence models
- Streamlit enables fast and effective ML deployment

👤 Author

Hrishikesh Gaikwad | 
AI & Machine Learning Enthusiast
