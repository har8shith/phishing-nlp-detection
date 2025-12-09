
# 📧 Phishing Email Detection & Information Extraction using NLP

An end-to-end NLP-based system that detects phishing emails using **TF-IDF** and **DistilBERT** models, and explains *why* an email is suspicious by extracting key information such as URLs, email addresses, money amounts, organizations, and suspicious actions.

This project includes a complete ML pipeline, API backend, frontend UI, and a dataset of 3000 realistic emails.

---

## 📌 About This Project

Phishing remains one of the most common cyber-attacks targeting individuals and organizations. Attackers craft emails that appear legitimate — making manual identification difficult and error-prone.

This project aims to solve that by:

### ✔ Automatically classifying emails as **Phishing** or **Legitimate**  
### ✔ Highlighting suspicious elements in the email using **NLP Extraction**  
### ✔ Providing real-time predictions via a **FastAPI backend**  
### ✔ Offering an interactive **Streamlit-based UI** for demonstration  

The system uses a **3000-email synthetic dataset** (1500 phishing + 1500 legitimate), covering various phishing categories such as account verification scams, password reset scams, bank fraud, delivery scams, and more.

---

## 🚀 Features

- 🔍 **Phishing Detection**
  - TF-IDF + Logistic Regression baseline model
  - Fine-tuned DistilBERT transformer model

- 🧠 **Information Extraction**
  - Extracts URLs, email addresses, amounts, organizations, dates
  - Detects suspicious action patterns (e.g., *verify account*, *reset password*)

- ⚙️ **Backend API**
  - Endpoints: `/api/predict`, `/api/analyze`, `/api/history`

- 🖥️ **Frontend UI**
  - Streamlit-based interface for email input & visual results
  - Highlights extracted suspicious elements

- 🗄️ **SQLite Database**
  - Stores history of predictions and extracted info

- 📦 Modular, clean, scalable project structure

---

## 🛠️ Tech Stack (Technologies Used)

### **Programming Language**
- Python 3.x

### **Machine Learning**
- TF-IDF Vectorization  
- Logistic Regression  
- HuggingFace Transformers  
- DistilBERT (Fine-Tuned)  
- PyTorch  

### **Natural Language Processing**
- Regex Pattern Matching  
- spaCy NER (`en_core_web_sm`)  
- Text Cleaning + Token Replacement  

### **Backend**
- FastAPI  
- Pydantic  
- Uvicorn  

### **Frontend**
- Streamlit  

### **Database**
- SQLite3  
- SQLAlchemy ORM  

### **Other Tools**
- Git & GitHub  
- Virtual Environments  
- JSON-based extraction & logging  

---

## 📂 Project Structure

phishing_nlp_project/
│
├── app/
│   ├── api/                 # FastAPI routers
│   ├── nlp/
│   │   ├── preprocessing.py
│   │   ├── tfidf_model.py
│   │   ├── bert_model.py
│   ├── services/extractor.py
│   ├── models.py            # SQLite ORM models
│   ├── database.py
│
├── data/
│   └── emails.csv           # 3000-email dataset (ignored in .gitignore)
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_logreg.pkl
│   └── bert_model/          # Saved BERT model files
│
├── train_tfidf.py           # Train TF-IDF model
├── train_bert.py            # Fine-tune DistilBERT
│
├── streamlit_app.py         # UI frontend
│
├── requirements.txt
├── README.md
└── .gitignore

Data Summary:
| Type       | Count    |
| ---------- | -------- |
| Phishing   | 1500     |
| Legitimate | 1500     |
| **Total**  | **3000** |



Each row contains:

subject, body, label
1 → phishing
0 → legitimate

---

## ⚙️ Installation & Setup

### **1️⃣ Create Virtual Environment**

bash
python3 -m venv .venv
source .venv/bin/activate


### **2️⃣ Install Required Libraries**

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

---

## 🧪 Model Training

### **Train TF-IDF Model**

```bash
python3 train_tfidf.py
```

### **Train DistilBERT Model**

```bash
python3 train_bert.py
```

Both models will be saved into the `models/` directory.

---

## 🚀 Running the Application

### **Start FastAPI Backend**

```bash
python3 -m uvicorn app.api.main:app --reload
```

API available at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)
API docs:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### **Start Streamlit Interface**

```bash
streamlit run streamlit_app.py
```

UI available at:
👉 [http://localhost:8501](http://localhost:8501)

Paste an email → choose model → get prediction + extracted clues.

---

## 🧾 Example Output (from UI)

```json
{
  "label": "phishing",
  "probability": 0.97,
  "model_used": "bert",
  "extracted_info": {
    "urls": ["http://verify-account-now.com"],
    "email_addresses": [],
    "money": [],
    "organizations": ["Bank"],
    "actions": ["verify your account"]
  }
}
```

---

## 🎯 Future Improvements

* Train with real-world enterprise email datasets
* Add multilingual detection (Hindi, Telugu, etc.)
* Use stronger models like RoBERTa / BERT-large
* Analyze full email headers (SPF, DMARC)
* Add LIME/SHAP explainability visualizations
* Detect image-based phishing attempts
* Deploy as a web service or browser extension

---

## 👨‍💻 Contributors

* **Banothu Harshith**
* **Shashivardhan Reddy**
* **Ineesh Reddy**
* **Bala Bhargav**
* **Manideep Kaparthi**

---

## ⭐ If you like this project…

Feel free to ⭐ star the repository and share!
Contributions and issues are welcome.

---



