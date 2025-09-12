from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model data
model_data = pickle.load(open('model.pkl', 'rb'))
model = model_data['model']
scaler = model_data['scaler']
use_scaling = model_data['use_scaling']

print(f"Model loaded successfully!")
print(f"Using scaling: {use_scaling}")
print(f"Feature columns expected: {model_data.get('feature_columns', 'Not specified')}")

@app.route('/')
def hello_world():
    return render_template('Form.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Extract input features from the form
        form_data = request.form
        
        # Based on your dataset structure, the features are:
        # [' no_of_dependents', ' education_binary', ' self_employed_binary', 
        #  ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', 
        #  ' residential_assets_value', ' bank_asset_value']
        
        dependents = float(form_data['dependents'])
        education = form_data['education']
        employment = form_data['employment']
        income = float(form_data['income'])
        loan_amount = float(form_data['loanAmount'])
        loan_term = float(form_data['loanTerm'])
        credit_score = float(form_data['creditScore'])
        res_assets_value = float(form_data['residentialAssets'])
        bank_assets_value = float(form_data['bankAssets'])

        # Convert categorical variables to binary (matching your dataset)
        education_binary = 1 if education.lower() == 'graduate' else 0
        employment_binary = 1 if employment.lower() == 'self-employed' else 0

        # Create input array in the exact order your model expects
        # Order: dependents, education_binary, self_employed_binary, income_annum, 
        #        loan_amount, loan_term, cibil_score, residential_assets_value, bank_asset_value
        input_features = np.array([
            dependents,           # no_of_dependents
            education_binary,     # education_binary  
            employment_binary,    # self_employed_binary
            income,              # income_annum
            loan_amount,         # loan_amount
            loan_term,           # loan_term
            credit_score,        # cibil_score
            res_assets_value,    # residential_assets_value
            bank_assets_value    # bank_asset_value
        ]).reshape(1, -1)

        print(f"Raw input features: {input_features}")
        
        # Apply scaling if the model was trained with scaling
        if use_scaling and scaler is not None:
            input_features = scaler.transform(input_features)
            print(f"Scaled input features: {input_features}")

        # Get probabilities for both classes
        prediction_proba = model.predict_proba(input_features)
        print(f"Raw prediction probabilities: {prediction_proba}")
        
        # probability of rejection (class 0)
        prob_rejection = prediction_proba[0][0]
        # probability of approval (class 1)  
        prob_approval = prediction_proba[0][1]
        
        print(f"Approval prob: {prob_approval:.4f}, Rejection prob: {prob_rejection:.4f}")
        
        # Format probabilities as percentages
        approval_percentage = prob_approval * 100
        rejection_percentage = prob_rejection * 100
        
        # Determine confidence level
        max_prob = max(prob_approval, prob_rejection)
        if max_prob > 0.8:
            confidence = "High"
        elif max_prob > 0.65:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        # Create detailed result message
        if prob_approval > 0.5:
            result_message = f"""Your Loan Application is likely to be APPROVED.

Approval Probability: {approval_percentage:.1f}%
Rejection Probability: {rejection_percentage:.1f}%

Confidence: {confidence}

Model Input Summary:
- Dependents: {dependents}
- Education: {education} (Binary: {education_binary})
- Employment: {employment} (Binary: {employment_binary})
- Annual Income: {income:,.0f}
- Loan Amount: {loan_amount:,.0f}
- Loan Term: {loan_term} months
- Credit Score: {credit_score}
- Residential Assets: {res_assets_value:,.0f}
- Bank Assets: {bank_assets_value:,.0f}"""
        else:
            result_message = f"""Your Loan Application is likely to be REJECTED.

Rejection Probability: {rejection_percentage:.1f}%
Approval Probability: {approval_percentage:.1f}%

Confidence: {confidence}

Model Input Summary:
- Dependents: {dependents}
- Education: {education} (Binary: {education_binary})
- Employment: {employment} (Binary: {employment_binary})
- Annual Income: {income:,.0f}
- Loan Amount: {loan_amount:,.0f}
- Loan Term: {loan_term} months
- Credit Score: {credit_score}
- Residential Assets: {res_assets_value:,.0f}
- Bank Assets: {bank_assets_value:,.0f}"""

        return render_template('Form.html', pred=result_message)
        
    except Exception as e:
        error_message = f"Error in prediction: {str(e)}"
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('Form.html', pred=error_message)

if __name__ == "__main__":
    print("Starting Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
