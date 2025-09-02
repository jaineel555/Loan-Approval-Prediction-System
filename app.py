from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('Form.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Extract input features from the form
    form_data = request.form
    dependents = float(form_data['dependents'])
    education = form_data['education']
    employment = form_data['employment']
    income = float(form_data['income'])
    loan_amount = float(form_data['loanAmount'])
    loan_term = float(form_data['loanTerm'])
    credit_score = float(form_data['creditScore'])
    res_assets_value = float(form_data['residentialAssets'])
    bank_assets_value = float(form_data['bankAssets'])



    education_binary = 1 if education.lower() == 'graduate' else 0
    employment_binary = 1 if employment.lower() == 'self-employed' else 0


    input_features = np.array([dependents, education_binary, employment_binary, income, loan_amount, credit_score, res_assets_value, bank_assets_value, loan_term]).reshape(1, -1)


    prediction = model.predict_proba(input_features)
    
    output = '{0:.{1}f}'.format(prediction[0][1], 2)


    if float(output) > 0.5:
        return render_template('Form.html', pred='Your Loan Application may get Approved.\nProbability is {}'.format(output))
    else:
        return render_template('Form.html', pred='Your Loan Application may get Rejected.\nProbability is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)



