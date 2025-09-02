import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
import warnings
import pickle
import math
warnings.filterwarnings('ignore')

data = pd.read_csv('Loan_Approval_Prediction_Dataset.csv')
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
   
   
   
   
##EVALUATION METRICS   


# Step 1: Make predictions
y_pred = model.predict(X_test)

# Step 2: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 3: Count correct predictions
correct_predictions = np.sum(y_test == y_pred)
print(f"Number of correct predictions: {correct_predictions}/{len(y_test)}")

# Step 4: Show full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Show confusion matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))


scores = cross_val_score(log_reg, x, y, cv=5)
print("Cross-validation scores:", scores)
print("Average score:", scores.mean())

print("Dataset shape:", x.shape, y.shape)
print("Target classes:", np.unique(y, return_counts=True))
