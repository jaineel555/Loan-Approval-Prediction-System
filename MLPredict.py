import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LARGE DATASET ANALYSIS (3000+ rows)")
print("="*60)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('Loan_Approval_Prediction_Dataset.csv')
print(f"Dataset loaded: {data.shape}")

# Display basic info
print(f"\nDataset Overview:")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print(f"Column names: {list(data.columns)}")

print(f"\nFirst 3 rows:")
print(data.head(3))

print(f"\nLast 3 rows:")
print(data.tail(3))

print(f"\nData types:")
for i, (col, dtype) in enumerate(zip(data.columns, data.dtypes)):
    print(f"  {i:2d}. {col:25s} - {dtype}")

# Check for missing values
print(f"\nMissing values per column:")
missing_vals = data.isnull().sum()
for col, missing in missing_vals.items():
    if missing > 0:
        print(f"  {col}: {missing} missing ({missing/len(data)*100:.1f}%)")

if missing_vals.sum() == 0:
    print("  No missing values found")

# Analyze each column to understand the data structure
print(f"\n" + "="*50)
print("COLUMN ANALYSIS")
print("="*50)

for i, col in enumerate(data.columns):
    print(f"\nColumn {i}: '{col}'")
    unique_vals = data[col].unique()
    n_unique = len(unique_vals)
    
    print(f"  Unique values: {n_unique}")
    
    if n_unique <= 10:  # Show all values if few unique values
        print(f"  Values: {unique_vals}")
        if n_unique == 2:
            print(f" BINARY COLUMN (potential target or categorical feature)")
    else:  # Show range for continuous variables
        if pd.api.types.is_numeric_dtype(data[col]):
            print(f"  Range: {data[col].min()} to {data[col].max()}")
            print(f"  Mean: {data[col].mean():.2f}, Std: {data[col].std():.2f}")
        else:
            print(f"  Sample values: {unique_vals[:5]}...")
    
    # Check if this could be the target variable
    if n_unique == 2:
        value_counts = data[col].value_counts()
        print(f"  Distribution: {dict(value_counts)}")
        
        # Calculate balance
        balance = min(value_counts) / max(value_counts)
        print(f"  Balance ratio: {balance:.3f} {'(Good)' if balance > 0.3 else '(Imbalanced)'}")

# Try to automatically identify the target column
print(f"\n" + "="*50)
print("TARGET VARIABLE IDENTIFICATION")
print("="*50)

# Look for common target column patterns
potential_targets = []

for i, col in enumerate(data.columns):
    col_lower = col.lower()
    unique_vals = len(data[col].unique())
    
    # Check for binary columns with loan-related names
    if unique_vals == 2:
        if any(keyword in col_lower for keyword in ['loan', 'approval', 'approved', 'status', 'result', 'target', 'class', 'label']):
            potential_targets.append((i, col, 'Name suggests target'))
        elif i == len(data.columns) - 1:  # Last column
            potential_targets.append((i, col, 'Last column (common target position)'))
        elif i == 10:  # Column K (index 10) as mentioned in original code
            potential_targets.append((i, col, 'Column K (as in original code)'))

print(f"Potential target columns found:")
if potential_targets:
    for idx, col, reason in potential_targets:
        values = data[col].unique()
        counts = data[col].value_counts()
        print(f"  Column {idx}: '{col}' - {reason}")
        print(f"    Values: {values}")
        print(f"    Distribution: {dict(counts)}")
else:
    print("  No obvious target column found. Checking last column and column 10...")
    
    # Check last column
    last_col_idx = len(data.columns) - 1
    last_col = data.columns[last_col_idx]
    last_unique = len(data[last_col].unique())
    print(f"  Last column ({last_col_idx}): '{last_col}' has {last_unique} unique values")
    if last_unique == 2:
        print(f"    Distribution: {dict(data[last_col].value_counts())}")
        potential_targets.append((last_col_idx, last_col, 'Last column with 2 values'))
    
    # Check column 10 if it exists
    if len(data.columns) > 10:
        col_10 = data.columns[10]
        col_10_unique = len(data[col_10].unique())
        print(f"  Column 10: '{col_10}' has {col_10_unique} unique values")
        if col_10_unique == 2:
            print(f"    Distribution: {dict(data[col_10].value_counts())}")
            potential_targets.append((10, col_10, 'Column 10 with 2 values'))

# Select the best target column
if not potential_targets:
    print("❌ No suitable binary target column found!")
    print("\nAll columns and their unique value counts:")
    for i, col in enumerate(data.columns):
        print(f"  {i}: {col} -> {len(data[col].unique())} unique values")
    exit()

# Use the first potential target (prioritize by order of likelihood)
target_idx, target_col, target_reason = potential_targets[0]
print(f"\nUsing column {target_idx} ('{target_col}') as target")
print(f"  Reason: {target_reason}")

# Prepare features and target
print(f"\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

# Use all columns except the target (and possibly ID column if first column looks like ID)
feature_cols = list(range(len(data.columns)))
feature_cols.remove(target_idx)

# Check if first column might be an ID (many unique values, sequential, etc.)
first_col = data.columns[0]
first_col_unique = len(data[first_col].unique())
if first_col_unique > len(data) * 0.8:  # More than 80% unique values
    print(f"Removing first column '{first_col}' (appears to be ID with {first_col_unique} unique values)")
    if 0 in feature_cols:
        feature_cols.remove(0)

print(f"Using columns {feature_cols} as features")
print(f"Using column {target_idx} as target")

# Extract features and target
X_raw = data.iloc[:, feature_cols]
y_raw = data.iloc[:, target_idx]

print(f"Features shape: {X_raw.shape}")
print(f"Target shape: {y_raw.shape}")

# Process features
print(f"\nProcessing features...")
X_processed = X_raw.copy()

# Handle categorical variables
categorical_cols = []
numeric_cols = []

for col in X_processed.columns:
    if pd.api.types.is_numeric_dtype(X_processed[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# Process categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} categories")

# Handle missing values in numeric columns
for col in numeric_cols:
    if X_processed[col].isnull().sum() > 0:
        median_val = X_processed[col].median()
        X_processed[col].fillna(median_val, inplace=True)
        print(f"  Filled missing values in '{col}' with median: {median_val}")

# Process target variable
y_processed = y_raw.copy()
target_unique = y_processed.unique()

if len(target_unique) == 2:
    # Ensure target is 0/1
    if not all(val in [0, 1] for val in target_unique):
        le_target = LabelEncoder()
        y_processed = le_target.fit_transform(y_processed)
        target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
        print(f"Target mapping: {target_mapping}")
    else:
        target_mapping = None

print(f"Final target distribution:")
target_dist = pd.Series(y_processed).value_counts().sort_index()
print(target_dist)
for val, count in target_dist.items():
    print(f"  Class {val}: {count} samples ({count/len(y_processed)*100:.1f}%)")

# Convert to numpy
X = X_processed.values.astype(float)
y = y_processed.values

# Train and evaluate models
print(f"\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Model 1: Standard Logistic Regression
print(f"\n1. Standard Logistic Regression:")
lr1 = LogisticRegression(random_state=42, max_iter=2000)
lr1.fit(X_train, y_train)

y_pred1 = lr1.predict(X_test)
y_prob1 = lr1.predict_proba(X_test)

acc1 = accuracy_score(y_test, y_pred1)
prob_stats1 = {
    'min': y_prob1[:, 1].min(),
    'max': y_prob1[:, 1].max(),
    'mean': y_prob1[:, 1].mean(),
    'std': y_prob1[:, 1].std(),
    'unique': len(np.unique(np.round(y_prob1[:, 1], 4)))
}

print(f"  Accuracy: {acc1:.4f}")
print(f"  Probability stats: min={prob_stats1['min']:.4f}, max={prob_stats1['max']:.4f}")
print(f"  Mean={prob_stats1['mean']:.4f}, Std={prob_stats1['std']:.4f}")
print(f"  Unique probabilities: {prob_stats1['unique']}")

# Model 2: Scaled Logistic Regression
print(f"\n2. Scaled Logistic Regression:")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr2 = LogisticRegression(random_state=42, max_iter=2000)
lr2.fit(X_train_scaled, y_train)

y_pred2 = lr2.predict(X_test_scaled)
y_prob2 = lr2.predict_proba(X_test_scaled)

acc2 = accuracy_score(y_test, y_pred2)
prob_stats2 = {
    'min': y_prob2[:, 1].min(),
    'max': y_prob2[:, 1].max(),
    'mean': y_prob2[:, 1].mean(),
    'std': y_prob2[:, 1].std(),
    'unique': len(np.unique(np.round(y_prob2[:, 1], 4)))
}

print(f"  Accuracy: {acc2:.4f}")
print(f"  Probability stats: min={prob_stats2['min']:.4f}, max={prob_stats2['max']:.4f}")
print(f"  Mean={prob_stats2['mean']:.4f}, Std={prob_stats2['std']:.4f}")
print(f"  Unique probabilities: {prob_stats2['unique']}")

# Model 3: Regularized Logistic Regression
print(f"\n3. L2 Regularized Logistic Regression:")
lr3 = LogisticRegression(random_state=42, max_iter=2000, C=0.1)  # Stronger regularization
lr3.fit(X_train_scaled, y_train)

y_pred3 = lr3.predict(X_test_scaled)
y_prob3 = lr3.predict_proba(X_test_scaled)

acc3 = accuracy_score(y_test, y_pred3)
prob_stats3 = {
    'min': y_prob3[:, 1].min(),
    'max': y_prob3[:, 1].max(),
    'mean': y_prob3[:, 1].mean(),
    'std': y_prob3[:, 1].std(),
    'unique': len(np.unique(np.round(y_prob3[:, 1], 4)))
}

print(f"  Accuracy: {acc3:.4f}")
print(f"  Probability stats: min={prob_stats3['min']:.4f}, max={prob_stats3['max']:.4f}")
print(f"  Mean={prob_stats3['mean']:.4f}, Std={prob_stats3['std']:.4f}")
print(f"  Unique probabilities: {prob_stats3['unique']}")

# Choose best model based on probability variation
models = [
    ('Standard LR', lr1, None, prob_stats1['std'], acc1),
    ('Scaled LR', lr2, scaler, prob_stats2['std'], acc2),
    ('Regularized LR', lr3, scaler, prob_stats3['std'], acc3)
]

best_model = max(models, key=lambda x: x[3])  # Choose highest std
model_name, model, model_scaler, model_std, model_acc = best_model

print(f"\nBest model: {model_name}")
print(f"  Probability variation (std): {model_std:.4f}")
print(f"  Accuracy: {model_acc:.4f}")

# Save the model
model_data = {
    'model': model,
    'scaler': model_scaler,
    'use_scaling': model_scaler is not None,
    'feature_columns': list(X_processed.columns),
    'target_column': target_col,
    'target_mapping': target_mapping if 'target_mapping' in locals() else None,
    'label_encoders': label_encoders
}

pickle.dump(model_data, open('model.pkl', 'wb'))
print(f"Model saved to model.pkl")

# Test with diverse samples
print(f"\n" + "="*50)
print("TESTING WITH SAMPLE DATA")
print("="*50)

# Create test samples that vary across the feature space
print("Creating diverse test samples...")

# Get feature ranges
feature_mins = X.min(axis=0)
feature_maxs = X.max(axis=0)
feature_means = X.mean(axis=0)

test_samples = []

# Sample 1: All minimum values (worst case)
test_samples.append(feature_mins)

# Sample 2: All maximum values (best case)  
test_samples.append(feature_maxs)

# Sample 3: All mean values (average case)
test_samples.append(feature_means)

# Sample 4: Mixed values (25th percentile)
test_samples.append(np.percentile(X, 25, axis=0))

# Sample 5: Mixed values (75th percentile)
test_samples.append(np.percentile(X, 75, axis=0))

# Test each sample
print(f"Testing {len(test_samples)} diverse samples:")

for i, sample in enumerate(test_samples, 1):
    sample_2d = sample.reshape(1, -1)
    
    if model_scaler:
        sample_2d = model_scaler.transform(sample_2d)
    
    prob = model.predict_proba(sample_2d)
    approval_prob = prob[0][1] * 100
    
    print(f"Sample {i}: {approval_prob:.1f}% approval probability")

# Final diagnosis
print(f"\n" + "="*60)
print("FINAL DIAGNOSIS")
print("="*60)

if model_std < 0.01:
    print("ERROR: Very low probability variation!")
    print("   Your model is likely overfitted or the data has issues.")
    print("   This explains why you're getting 100% predictions.")
    print("\n   Possible solutions:")
    print("   1. Check if target variable is balanced")
    print("   2. Try different regularization parameters")
    print("   3. Remove highly correlated features")
    print("   4. Check for data leakage")
elif prob_stats2['unique'] < 10:
    print("LOW VARIATION: Model produces few unique probabilities")
    print("   This might cause limited prediction ranges.")
else:
    print("GOOD: Model shows reasonable probability variation")
    print("   The 100% issue might be in the Flask app or input processing.")

print(f"\nModel Performance Summary:")
print(f"  Probability range: [{min(prob_stats1['min'], prob_stats2['min'], prob_stats3['min']):.4f}, {max(prob_stats1['max'], prob_stats2['max'], prob_stats3['max']):.4f}]")
print(f"  Best model std: {model_std:.4f}")
print(f"  Dataset size: {len(data)} rows")
print(f"  Features used: {X.shape[1]}")
print(f"  Target balance: {target_dist.values}")
