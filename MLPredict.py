import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import warnings
import pickle
import math
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan Approval Prediction Dataset.csv')
data = np.array(data)

x = data[:, 1:10]  # Columns B to J (index 1 to 9)
y = data[:, 10]    # Column K (index 10)


#print(x[:5])  # Display first 5 rows of x
#print("Independent Variables (x):")

#print("\nDependent Variable (y):")
#print(y[:5])  # Display first 5 rows of y

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pickle.dump(log_reg, open('model.pkl', 'wb'))
model=pickle.load(open('model.pkl','rb'))