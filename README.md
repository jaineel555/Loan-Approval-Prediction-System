# 🏦 AI Loan Approval Prediction System

A production‑ready Machine Learning + Flask web app that predicts whether a loan application will be Approved ✅ or Rejected ❌ with explainable, real‑time feedback. [attached_file:2]

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg) ![Flask](https://img.shields.io/badge/Flask-3.x-black.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) [attached_file:2]

---

## 📌 Table of Contents
- [Overview](#overview) [attached_file:2]
- [Features](#🚀Features) [attached_file:2]
- [Architecture](#architecture) [attached_file:2]
- [Tech Stack](#tech-stack) [attached_file:2]
- [Quick Start](#quick-start) [attached_file:2]
- [Model Training](#model-training) [attached_file:2]
- [Usage Guide](#usage-guide) [attached_file:2]
- [Project Structure](#project-structure) [attached_file:2]
- [Example Output](#example-output) [attached_file:2]
- [Roadmap](#roadmap) [attached_file:2]
- [Contributing](#contributing) [attached_file:2]
- [License](#license) [attached_file:2]
- [Contact](#contact) [attached_file:2]

---

## Overview
This repository contains a complete ML workflow and Flask web application that evaluates applicant details and returns an approval decision with probabilities and human‑readable reasons. [attached_file:2]

The app serves an elegant form UI, validates inputs, calls a trained Logistic Regression model, and renders an attractive, high‑contrast result panel. [attached_file:2]

---

## 🚀 Features
- Real‑time approval decision with approval and rejection probabilities. [attached_file:2]
- Clear “Probability Analysis”, “Financial Ratios”, “Risk/Positive Factors”, and “Recommendation” sections. [attached_file:2]
- Clean Flask routes with JSON response for easy UI rendering and integrations. [attached_file:2]
- Reproducible model pipeline with training script and saved model artifact (model.pkl). [attached_file:2]
- Indian currency formatting, credit score banding, and input sliders for great UX. [attached_file:2]
- Fully client‑side enhancements for visibility, accessibility, and mobile responsiveness. [attached_file:2]

---

## 🧭 Architecture
- MLPredict.py: trains Logistic Regression on the loan dataset and saves model.pkl. [attached_file:2]
- app.py: loads model.pkl, exposes “/” for UI and “/predict” for JSON prediction API. [attached_file:2]
- templates/static assets deliver a polished glass‑effect UI with strong text contrast. [attached_file:2]

High level flow:
Applicant Form ➜ Flask “/predict” ➜ Model.predict_proba ➜ JSON ➜ Styled Result Cards. [attached_file:2]

---

## 🛠 Tech Stack
- Language: Python 3.10+ (Pandas, NumPy) for data wrangling and utilities. [attached_file:2]
- Framework: Flask for serving the web UI and prediction API. [attached_file:2]
- ML: scikit‑learn (Logistic Regression) for calibrated, interpretable probabilities. [attached_file:2]
- Frontend: HTML/CSS/JS with modern, responsive UI and contrast‑safe result panels. [attached_file:2]

---

## ⚡ Quick Start
1) Clone
git clone https://github.com/jaineel555/Loan-Approval-Prediction-System
cd Loan-Approval-Prediction-System

2) Create & activate virtual env (recommended)
python -m venv .venv

Windows
.venv\Scripts\activate

macOS/Linux
source .venv/bin/activate

3) Install dependencies
pip install -r requirements.txt

4) (Optional) Train model from CSV and produce model.pkl
python MLPredict.py

5) Run the Flask app
python app.py

6) Open in browser
http://127.0.0.1:5000
[attached_file:2]

---

## 🧪 Model Training
The MLPredict.py script loads Loan_Approval_Prediction_Dataset.csv, performs splits, trains Logistic Regression, evaluates metrics, and saves model.pkl for the app. [attached_file:2]

You can tweak hyperparameters, calibration, or feature processing inside the training script to experiment with performance. [attached_file:2]

---

## 📘 Usage Guide
1) Fill out the form with dependents, education, employment, income, loan amount, term, credit score, and assets. [attached_file:2]
2) Click “Predict Loan Approval” to submit and receive a detailed decision panel. [attached_file:2]
3) The result includes overall status, probabilities, financial ratios, risk/positive factors, and a recommendation. [attached_file:2]

API note: “/predict” returns JSON (success, status, probabilities, ratios, factors), enabling integration with other UIs or services. [attached_file:2]

---

## 🗂 Project Structure
Loan-Approval-Prediction-System/
├─ app.py # Flask app & /predict endpoint
├─ MLPredict.py # Training script → saves model.pkl
├─ model.pkl # Saved model artifact (after training)
├─ requirements.txt # Python dependencies
├─ templates/
│ └─ Form.html # Main UI template
├─ static/ # CSS/JS/assets
└─ Loan_Approval_Prediction_Dataset.csv # Dataset (if included)
[attached_file:2]

---

## 📈 Example Output
- LOAN APPROVED with Approval Probability: 91.1%, Confidence: Very High. [attached_file:2]
- Financial Analysis: Loan‑to‑Income and Debt‑to‑Assets ratios with green/red highlights. [attached_file:2]
- Positive/Risk Factors: bullet list explaining the decision with clear visual cues. [attached_file:2]
- Recommendation: actionable guidance based on profile quality and risk. [attached_file:2]

---

## 🗺 Roadmap
- Add model comparison (Logistic vs RandomForest/Calibrated RF) toggle in UI. [attached_file:2]
- Persist predictions with lightweight DB and admin insights dashboard. [attached_file:2]
- Ship Dockerfile and GitHub Actions for CI/CD and containerized deploys. [attached_file:2]
- Add SHAP or feature impact bars to boost explainability. [attached_file:2]

---

## 🤝 Contributing
Contributions are welcome: fork the repo, create a feature branch, submit a PR, and describe your changes clearly. [attached_file:2]

Please run training and app smoke tests before opening a PR to keep main stable. [attached_file:2]

---

## 📜 License
This project is released under the MIT License—see the LICENSE file for details. [attached_file:2]

---

## 📬 Contact
- Author: Jaineel Purani [attached_file:2]
- GitHub: https://github.com/jaineel555 [attached_file:2]
- Gmail (compose link): https://mail.google.com/mail/?view=cm&tf=cm&to=jaineelpurani555@gmail.com&su=Regarding%20AI%20Loan%20Approval%20Predictor [attached_file:2]
