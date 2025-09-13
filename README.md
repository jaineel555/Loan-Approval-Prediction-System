# 🏦 Loan Approval Prediction System  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)](https://flask.palletsprojects.com/)  
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.6-orange)](https://scikit-learn.org/) 
[![NumPy](https://img.shields.io/badge/NumPy-1.25-blueviolet)](https://numpy.org/)  
[![Pandas](https://img.shields.io/badge/Pandas-2.1-lightblue)](https://pandas.pydata.org/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-red)](https://matplotlib.org/)  
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12-blue)](https://seaborn.pydata.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


A **Machine Learning + Flask web app** that predicts whether a loan application will be **Approved ✅** or **Rejected ❌**.  
This project uses a **Logistic Regression model**, trained on historical loan applicant data, and provides **real-time predictions** with probability analysis, financial ratios, risk factors, and recommendations.  

---

## 📌 Table of Contents  
- [🚀 Features](#-features)  
- [📊 Workflow](#-workflow)  
- [⚙️ Tech Stack](#️-tech-stack)  
- [🛠️ Installation & Usage](#️-installation--usage)   
- [📌 Future Enhancements](#-Future-Enhancements).  
- [🤝 Contributing](#-contributing)  
- [👨‍💻 Author](#-author)  
- [📜 License](#-license)  

---

## 🚀 Features  

- 📂 **Logistic Regression model** trained on 3000+ loan records  
- 🖥️ **Flask Web UI** (`Form.html`) with modern design (sliders, tooltips, responsive)  
- 🔄 **Feature Encoding** for education & employment  
- 📊 **Model Evaluation**: Accuracy, Classification Report, Confusion Matrix, Cross-validation  
- 💾 **Pickle Integration** (`model.pkl`) for deployment-ready model  
- ⚡ **Real-time Predictions** with `predict_proba()`  
- 🧮 **Probability Analysis & Confidence Levels**  
- 📈 **Financial Ratios** → Loan-to-Income, Debt-to-Assets  
- ⚠️ **Risk & Positive Factors** explained for transparency  
- 🎯 **Recommendation System** → suggests approval, caution, or decline  

---

## 📊 Workflow  

1. **Model Training** (`MLPredict.py`)  
   - Loads dataset (`Loan_Approval_Prediction_Dataset.csv`)  
   - Preprocesses numeric + categorical features  
   - Trains multiple Logistic Regression variants  
   - Evaluates using Accuracy & Probability Variance  
   - Saves best model → `model.pkl`  

2. **Web Application** (`app.py`)  
   - Loads trained model (`model.pkl`)  
   - Renders input form (`Form.html`)  
   - Processes applicant data → numeric + binary encoding  
   - Runs prediction with approval probability  
   - Returns detailed results (status, ratios, risk/positive factors, recommendation)  

---

## ⚙️ Tech Stack  

- **Language:** Python 🐍  
- **Framework:** Flask  
- **Machine Learning:** Scikit-learn (Logistic Regression)  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Frontend:** HTML, CSS, JS (Jinja2 templates)  

---

## 🛠️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction

---
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model (optional)
```bash
python MLPredict.py
```

### ➡️ This generates model.pkl.

### 4️⃣ Run Flask app
```bash
python app.py
```

### 5️⃣ Open in browser
```bash
👉 http://127.0.0.1:5000/
```

---

## 📌 Future Enhancements  

- 🔹 Add more ML models (Random Forest, XGBoost)  
- 🔹 Deploy app on **Heroku/Render/AWS**  
- 🔹 Integrate database to store loan applications  
- 🔹 Use **Explainable AI (SHAP, LIME)** for better interpretability  
- 🔹 Multi-language support  

---

## 🤝 Contributing  

Contributions are welcome! 🎉  
- Fork the repo  
- Create a new branch (`feature-xyz`)  
- Commit your changes  
- Submit a Pull Request  

---

## 👨‍💻 Author  

**Jaineel Purani**  

📌 [GitHub](https://github.com/jaineel555)  
📌 [LinkedIn](https://www.linkedin.com/in/jaineel-purani-9a128120b/)  
📌 [Instagram](https://www.instagram.com/jaineel_purani__555/)  
📌 [Email](mailto:jaineelpurani555@gmail.com)  

---

## 📜 License  

This project is licensed under the **MIT License** – feel free to use and improve it with giving credits!  

---
