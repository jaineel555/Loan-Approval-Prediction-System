# MLPredict.py - Improved Random Forest with Better Probability Calibration
import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- CONFIG ----------
CSV_PATH = "Loan_Approval_Prediction_Dataset.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.30
# ----------------------------

def load_and_prepare(csv_path):
    """Load CSV and prepare X, y and feature names."""
    df = pd.read_csv(csv_path)
    arr = np.array(df)

    # Extract features and target
    X = arr[:, 1:10].astype(float)
    y = arr[:, 10].astype(int)

    feature_names = [
        "no_of_dependents",
        "education_binary", 
        "self_employed_binary",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "bank_asset_value"
    ]

    return X, y, feature_names

def create_optimized_random_forest():
    """Create a Random Forest optimized for smooth probability outputs."""
    return RandomForestClassifier(
        n_estimators=500,          # More trees for smoother probabilities
        max_depth=15,              # Prevent overfitting but allow complexity
        min_samples_split=20,      # Require more samples to split (smoother)
        min_samples_leaf=10,       # Require more samples per leaf (smoother)
        max_features='sqrt',       # Use sqrt of features for diversity
        bootstrap=True,            # Use bootstrap sampling
        oob_score=True,            # Out-of-bag scoring
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'    # Handle class imbalance
    )

def train_multiple_models_and_select_best(X, y, feature_names, model_path):
    """Train multiple models and select the one with best probability distribution."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution - Training: {np.bincount(y_train)}")
    print(f"Class distribution - Test: {np.bincount(y_test)}")
    
    models_to_try = []
    
    # Model 1: Optimized Random Forest with Platt Scaling
    print("\n=== Training Model 1: Optimized Random Forest + Platt Scaling ===")
    rf_optimized = create_optimized_random_forest()
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", rf_optimized)
    ])
    
    # Use Platt scaling (sigmoid) which often works better for Random Forest
    rf_calibrated = CalibratedClassifierCV(
        estimator=rf_pipeline, 
        cv=5, 
        method="sigmoid"  # Platt scaling
    )
    
    rf_calibrated.fit(X_train, y_train)
    rf_pred = rf_calibrated.predict(X_test)
    rf_proba = rf_calibrated.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'name': 'Random Forest + Platt Scaling',
        'model': rf_calibrated,
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'f1': f1_score(y_test, rf_pred),
        'prob_std': rf_proba.std(),
        'unique_probs': len(np.unique(np.round(rf_proba, 4))),
        'prob_range': (rf_proba.min(), rf_proba.max())
    }
    models_to_try.append(rf_metrics)
    
    # Model 2: Random Forest with Isotonic Calibration
    print("\n=== Training Model 2: Random Forest + Isotonic Calibration ===")
    rf_isotonic = CalibratedClassifierCV(
        estimator=rf_pipeline, 
        cv=5, 
        method="isotonic"  # Isotonic regression
    )
    
    rf_isotonic.fit(X_train, y_train)
    iso_pred = rf_isotonic.predict(X_test)
    iso_proba = rf_isotonic.predict_proba(X_test)[:, 1]
    
    iso_metrics = {
        'name': 'Random Forest + Isotonic Calibration',
        'model': rf_isotonic,
        'accuracy': accuracy_score(y_test, iso_pred),
        'roc_auc': roc_auc_score(y_test, iso_proba),
        'f1': f1_score(y_test, iso_pred),
        'prob_std': iso_proba.std(),
        'unique_probs': len(np.unique(np.round(iso_proba, 4))),
        'prob_range': (iso_proba.min(), iso_proba.max())
    }
    models_to_try.append(iso_metrics)
    
    # Model 3: Logistic Regression (for comparison)
    print("\n=== Training Model 3: Logistic Regression (Reference) ===")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=2000,
            class_weight='balanced'
        ))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    
    lr_metrics = {
        'name': 'Logistic Regression',
        'model': lr_pipeline,
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'f1': f1_score(y_test, lr_pred),
        'prob_std': lr_proba.std(),
        'unique_probs': len(np.unique(np.round(lr_proba, 4))),
        'prob_range': (lr_proba.min(), lr_proba.max())
    }
    models_to_try.append(lr_metrics)
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':<10} {'ROC-AUC':<10} {'F1':<10} {'Prob_Std':<10} {'Unique_Probs':<12}")
    print("-" * 80)
    
    for m in models_to_try:
        print(f"{m['name']:<35} {m['accuracy']:<10.4f} {m['roc_auc']:<10.4f} {m['f1']:<10.4f} {m['prob_std']:<10.4f} {m['unique_probs']:<12}")
        print(f"{'  Probability range:':<35} [{m['prob_range'][0]:.4f}, {m['prob_range'][1]:.4f}]")
        print()
    
    # Select best model based on combination of performance and probability distribution
    # Priority: ROC-AUC > Probability Standard Deviation > Unique Probabilities
    def model_score(m):
        return m['roc_auc'] * 0.6 + (m['prob_std'] * 0.3) + (min(m['unique_probs'], 500) / 500.0 * 0.1)
    
    best_model = max(models_to_try, key=model_score)
    
    print(f"SELECTED MODEL: {best_model['name']}")
    print(f"Selection criteria score: {model_score(best_model):.4f}")
    print(f"This model has:")
    print(f"  - ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f"  - Probability Standard Deviation: {best_model['prob_std']:.4f}")
    print(f"  - Unique Probabilities: {best_model['unique_probs']}")
    print(f"  - Probability Range: [{best_model['prob_range'][0]:.4f}, {best_model['prob_range'][1]:.4f}]")
    
    # Test with diverse samples to verify probability distribution
    print(f"\n" + "="*50)
    print("TESTING PROBABILITY DISTRIBUTION")
    print("="*50)
    
    # Create test samples across different ranges
    feature_ranges = {
        'dependents': (0, 5),
        'income': (200000, 9900000),
        'loan_amount': (300000, 39500000),
        'credit_score': (300, 900),
        'residential_assets': (0, 29100000),
        'bank_assets': (0, 14700000)
    }
    
    test_samples = [
        # Very bad profile
        [5, 0, 0, 200000, 20000000, 6, 300, 100000, 50000],
        # Bad profile  
        [4, 0, 0, 500000, 15000000, 8, 400, 500000, 200000],
        # Below average
        [3, 1, 0, 1500000, 10000000, 12, 500, 2000000, 1000000],
        # Average
        [2, 1, 1, 3000000, 8000000, 16, 600, 5000000, 3000000],
        # Above average
        [1, 1, 1, 6000000, 5000000, 18, 700, 10000000, 6000000],
        # Good profile
        [1, 1, 0, 8000000, 3000000, 20, 800, 15000000, 8000000],
        # Very good profile
        [0, 1, 1, 9900000, 1000000, 20, 900, 25000000, 12000000]
    ]
    
    print("Testing with 7 diverse profiles:")
    print("Profile descriptions: Very Bad -> Bad -> Below Avg -> Average -> Above Avg -> Good -> Very Good")
    
    probabilities = []
    for i, sample in enumerate(test_samples):
        sample_array = np.array(sample).reshape(1, -1)
        prob = best_model['model'].predict_proba(sample_array)[0][1]
        probabilities.append(prob * 100)
        print(f"Profile {i+1}: {prob*100:5.1f}% approval probability")
    
    # Check if we have good distribution
    prob_std = np.std(probabilities)
    prob_range = max(probabilities) - min(probabilities)
    
    print(f"\nProbability Distribution Analysis:")
    print(f"  Standard Deviation: {prob_std:.1f}%")
    print(f"  Range: {prob_range:.1f}%")
    print(f"  Min: {min(probabilities):.1f}%, Max: {max(probabilities):.1f}%")
    
    if prob_std < 5:
        print("  WARNING: Low variation in probabilities!")
    elif prob_range < 30:
        print("  WARNING: Limited probability range!")
    else:
        print("  GOOD: Model shows good probability variation")
    
    # Save the best model
    model_data = {
        "model": best_model['model'],
        "feature_columns": feature_names,
        "use_scaling": True,
        "meta": {
            "classifier_type": best_model['name'],
            "random_state": RANDOM_STATE,
            "performance": {
                "accuracy": best_model['accuracy'],
                "roc_auc": best_model['roc_auc'],
                "f1_score": best_model['f1'],
                "prob_std": best_model['prob_std'],
                "unique_probs": best_model['unique_probs']
            }
        }
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    return best_model

if __name__ == "__main__":
    X, y, feature_names = load_and_prepare(CSV_PATH)
    print(f"Dataset shape: {X.shape}, {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    best_model = train_multiple_models_and_select_best(X, y, feature_names, MODEL_PATH)
    print(f"\nTraining complete! Best model: {best_model['name']}")
