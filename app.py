# app.py - Updated to return JSON for AJAX requests
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import traceback

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# Load model data
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data.get("model")
    feature_columns = model_data.get("feature_columns", None)
    use_scaling = model_data.get("use_scaling", True)
    meta = model_data.get("meta", {})
    
    print("Model loaded successfully.")
    print("Model type:", meta.get("classifier_type", "Unknown"))
    print("Feature columns expected:", feature_columns)
    print("Using scaling:", use_scaling)
    
    if "performance" in meta:
        perf = meta["performance"]
        print(f"Model performance: Accuracy={perf.get('accuracy', 'N/A'):.4f}, ROC-AUC={perf.get('roc_auc', 'N/A'):.4f}")
        print(f"Probability std={perf.get('prob_std', 'N/A'):.4f}, Unique probs={perf.get('unique_probs', 'N/A')}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def format_rupees(amount):
    """Format amount in Indian Rupees with proper comma separation."""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.1f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.1f} L"
    elif amount >= 1000:    # 1 thousand
        return f"₹{amount/1000:.0f}K"
    else:
        return f"₹{amount:,.0f}"

def to_binary_education(education_str):
    """Convert education to binary (1=Graduate, 0=Not Graduate)."""
    return 1 if str(education_str).strip().lower() == "graduate" else 0

def to_binary_employment(employment_str):
    """Convert employment to binary (1=Self-Employed, 0=Salaried)."""
    return 1 if str(employment_str).strip().lower() == "self-employed" else 0

@app.route("/")
def index():
    return render_template("Form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({
                "success": False,
                "prediction": "Error: Model not loaded properly.",
                "error": "Model loading failed"
            }), 500
        
        form = request.form

        # Parse inputs with validation
        try:
            dependents = max(0, min(5, float(form.get("dependents", 0))))
            education = form.get("education", "not graduate")
            employment = form.get("employment", "salaried")
            income = max(0, float(form.get("income", 0)))
            loan_amount = max(0, float(form.get("loanAmount", 0)))
            loan_term = max(1, float(form.get("loanTerm", 12)))
            credit_score = max(300, min(900, float(form.get("creditScore", 500))))
            res_assets_value = max(0, float(form.get("residentialAssets", 0)))
            bank_assets_value = max(0, float(form.get("bankAssets", 0)))
        except ValueError as ve:
            return jsonify({
                "success": False,
                "prediction": f"Invalid input values: {ve}",
                "error": str(ve)
            }), 400

        # Convert categorical to binary
        education_binary = to_binary_education(education)
        employment_binary = to_binary_employment(employment)

        # Build feature array in training order
        input_features = np.array([
            dependents,
            education_binary,
            employment_binary,
            income,
            loan_amount,
            loan_term,
            credit_score,
            res_assets_value,
            bank_assets_value
        ]).reshape(1, -1)

        print("Raw input features:", input_features.tolist())

        # Get predictions
        proba = model.predict_proba(input_features)[0]
        
        # Handle class mapping properly
        if hasattr(model, "classes_"):
            classes = model.classes_
        else:
            # For calibrated classifiers, try to get classes from base estimator
            try:
                if hasattr(model, "calibrated_classifiers_"):
                    classes = model.calibrated_classifiers_[0].classes_
                else:
                    classes = np.array([0, 1])
            except:
                classes = np.array([0, 1])

        # Get approval probability (class 1)
        if 1 in classes and len(proba) > 1:
            idx_approval = int(np.where(classes == 1)[0][0])
            prob_approval = proba[idx_approval]
        else:
            prob_approval = proba[1] if len(proba) > 1 else proba[0]
        
        prob_rejection = 1.0 - prob_approval
        
        print(f"Raw probabilities: {proba}")
        print(f"Classes: {classes}")
        print(f"Approval probability: {prob_approval:.4f}")

        # Format probabilities
        approval_percentage = prob_approval * 100
        rejection_percentage = prob_rejection * 100

        # Determine confidence level
        max_prob = max(prob_approval, prob_rejection)
        if max_prob > 0.85:
            confidence = "Very High"
        elif max_prob > 0.75:
            confidence = "High"
        elif max_prob > 0.65:
            confidence = "Moderate"
        elif max_prob > 0.55:
            confidence = "Low"
        else:
            confidence = "Very Low"

        # Calculate some useful metrics
        loan_to_income_ratio = (loan_amount / income * 100) if income > 0 else 0
        debt_to_assets_ratio = (loan_amount / (res_assets_value + bank_assets_value) * 100) if (res_assets_value + bank_assets_value) > 0 else 0

        # Determine status and create result message
        status = "APPROVED" if prob_approval > 0.5 else "REJECTED"
        
        result_message = f"""🎯 LOAN PREDICTION RESULT 🎯

Status: Your Loan Application is likely to be {status}

📊 PROBABILITY ANALYSIS:
• Approval Probability: {approval_percentage:.1f}%
• Rejection Probability: {rejection_percentage:.1f}%
• Confidence Level: {confidence}

💰 APPLICATION SUMMARY:
• Applicant: {education} {'Self-Employed' if employment_binary else 'Salaried'} professional
• Dependents: {int(dependents)}
• Annual Income: {format_rupees(income)}
• Loan Amount: {format_rupees(loan_amount)}
• Loan Term: {int(loan_term)} months
• Credit Score: {int(credit_score)}
• Residential Assets: {format_rupees(res_assets_value)}
• Bank Assets: {format_rupees(bank_assets_value)}

📈 FINANCIAL RATIOS:
• Loan-to-Income Ratio: {loan_to_income_ratio:.1f}%
• Debt-to-Assets Ratio: {debt_to_assets_ratio:.1f}%

🔍 RISK ANALYSIS:"""

        # Add risk factor analysis
        risk_factors = []
        positive_factors = []
        
        if loan_to_income_ratio > 500:
            risk_factors.append("Very high loan-to-income ratio (>{:.0f}%)".format(loan_to_income_ratio))
        elif loan_to_income_ratio > 300:
            risk_factors.append("High loan-to-income ratio ({:.0f}%)".format(loan_to_income_ratio))
        else:
            positive_factors.append("Reasonable loan-to-income ratio ({:.0f}%)".format(loan_to_income_ratio))
            
        if credit_score < 550:
            risk_factors.append("Low credit score ({})".format(int(credit_score)))
        elif credit_score > 750:
            positive_factors.append("Excellent credit score ({})".format(int(credit_score)))
        elif credit_score > 650:
            positive_factors.append("Good credit score ({})".format(int(credit_score)))
            
        if dependents > 3:
            risk_factors.append("High number of dependents ({})".format(int(dependents)))
        elif dependents <= 1:
            positive_factors.append("Low dependency burden ({})".format(int(dependents)))
            
        if loan_term < 6:
            risk_factors.append("Very short loan term ({} years)".format(int(loan_term)))
        elif loan_term > 15:
            positive_factors.append("Comfortable repayment period ({} years)".format(int(loan_term)))

        if employment_binary == 1:  # Self-employed
            risk_factors.append("Self-employed status (income variability)")
        else:
            positive_factors.append("Stable salaried employment")
            
        # Add factors to result
        if risk_factors:
            result_message += "\n\n⚠️ RISK FACTORS:\n• " + "\n• ".join(risk_factors)
        if positive_factors:
            result_message += "\n\n✅ POSITIVE FACTORS:\n• " + "\n• ".join(positive_factors)
            
        if not risk_factors and not positive_factors:
            result_message += "\n\n📋 Overall profile appears balanced"

        # Add recommendation
        if prob_approval > 0.8:
            result_message += f"\n\n🎉 RECOMMENDATION: Excellent candidate for loan approval!"
        elif prob_approval > 0.6:
            result_message += f"\n\n👍 RECOMMENDATION: Good candidate, consider approval with standard terms."
        elif prob_approval > 0.4:
            result_message += f"\n\n🤔 RECOMMENDATION: Moderate risk, may require additional verification."
        else:
            result_message += f"\n\n❌ RECOMMENDATION: High risk, consider declining or requiring collateral."

        # Return JSON response for AJAX
        return jsonify({
            "success": True,
            "prediction": result_message,
            "approval_probability": float(approval_percentage),
            "rejection_probability": float(rejection_percentage),
            "confidence": confidence,
            "status": status,
            "loan_to_income_ratio": float(loan_to_income_ratio),
            "debt_to_assets_ratio": float(debt_to_assets_ratio),
            "risk_factors": risk_factors,
            "positive_factors": positive_factors
        })

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "prediction": f"Error in prediction: {str(e)}",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("="*50)
    print("🤖 AI LOAN APPROVAL PREDICTION SYSTEM")
    print("="*50)
    print("Starting Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("All amounts are in Indian Rupees (₹)")
    print("Model ready for predictions!")
    print("="*50)
    app.run(debug=True, host="127.0.0.1", port=5000)
