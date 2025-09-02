# 🏦 Loan Approval Prediction System  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)](https://flask.palletsprojects.com/)  
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.6-orange)](https://scikit-learn.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

A **Machine Learning + Flask web app** that predicts whether a loan application will be **Approved ✅** or **Rejected ❌**.  
This project uses a **Logistic Regression model**, trained on historical loan applicant data, and provides **real-time predictions** through a web form.  

---

## 📌 Table of Contents  
- [🚀 Features](#-features)  
- [📊 Workflow](#-workflow)  
- [⚙️ Tech Stack](#️-tech-stack)  
- [🛠️ Installation & Usage](#️-installation--usage)  
- [📈 Example Output](#-example-output)  
- [📌 Future Enhancements](#-future-enhancements)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

---

## 🚀 Features  

- 📂 **Trains Logistic Regression model** on loan dataset  
- 🖥️ **Flask Web UI** with input form (`Form.html`)  
- 🔄 **Feature Encoding** for education & employment details  
- 📊 **Model Evaluation** with Accuracy, Classification Report, Confusion Matrix, Cross-validation  
- 💾 **Pickle Integration** → Model stored as `model.pkl` for deployment  
- ⚡ **Real-time Predictions** with approval probability  

---

## 📊 Workflow  

1. **Data Preprocessing & Training** (`MLPredict.py`)  
   - Loads dataset (`Loan_Approval_Prediction_Dataset.csv`)  
   - Splits into train & test sets  
   - Trains **Logistic Regression** model  
   - Saves model → `model.pkl`  
   - Prints evaluation metrics  

2. **Flask Web App** (`app.py`)  
   - Loads `model.pkl`  
   - Renders form for applicant details  
   - Processes inputs → numeric + categorical encoding  
   - Uses `predict_proba()` for probability-based prediction  
   - Displays **Approved/Rejected** with probability  

---

## ⚙️ Tech Stack  

- **Language:** Python 🐍  
- **Framework:** Flask  
- **Machine Learning:** Scikit-learn (Logistic Regression)  
- **Libraries:** Pandas, NumPy, Matplotlib  
- **Frontend:** HTML (Jinja2 templates)  

---

## 🛠️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional)
python MLPredict.py

➡️ This generates model.pkl.

4️⃣ Run Flask app
python app.py

5️⃣ Open in browser

👉 http://127.0.0.1:5000/
