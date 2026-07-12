import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & PLOT SETTINGS ---
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["figure.dpi"] = 100

# Load final feature engineered dataset
df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --- 2. TRAIN-TEST SPLIT BY PARTICIPANT (GENERALIZATION TEST) ---
# Create a strict train/test split based on participants to avoid data leakage
df_train = df[df["participant"] != "E"]
df_test = df[df["participant"] == "E"]

# Separate Target and Features
X_train = df_train.drop(["label", "participant", "category", "set"], axis=1)
y_train = df_train["label"]

X_test = df_test.drop(["label", "participant", "category", "set"], axis=1)
y_test = df_test["label"]

# --- 3. DEFINE FEATURE SUBSETS FOR ITERATIVE EVALUATION ---
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = [col for col in X_train.columns if "_r" in col]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [col for col in X_train.columns if "_roll_" in col]
freq_features = [col for col in X_train.columns if "_freq_" in col or "_pse" in col]
cluster_features = ["cluster"]

# Combine progressive subsets
feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# --- 4. MODEL SELECTION PIPELINE ---
class ClassificationModelSelection:
    def __init__(self):
        self.models = {
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Neural Network (MLP)": MLPClassifier(alpha=1, max_iter=1000, random_state=42)
        }
        
    def evaluate_feature_sets(self, X_train, y_train, X_test, y_test, feature_sets):
        results = []
        for name, model in self.models.items():
            print(f"Training {name}...")
            for i, f_set in enumerate(feature_sets, start=1):
                # Fit model on custom subset
                model.fit(X_train[f_set], y_train)
                
                # Predict
                y_pred = model.predict(X_test[f_set])
                acc = accuracy_score(y_test, y_pred)
                
                results.append({
                    "Model": name,
                    "Feature Set": f"Set {i}",
                    "Accuracy": acc
                })
        return pd.DataFrame(results)

# Execute evaluation across all subsets
selector = ClassificationModelSelection()
feature_sets = [feature_set_1, feature_set_2, feature_set_3, feature_set_4]
performance_df = selector.evaluate_feature_sets(X_train, y_train, X_test, y_test, feature_sets) 

# --- 5. VISUALIZE FEATURE PERFORMANCE COMPARISON ---
print("\n--- Performance Metrics Matrix ---")
print(performance_df.pivot(index="Model", columns="Feature Set", values="Accuracy"))

sns.barplot(data=performance_df, x="Model", y="Accuracy", hue="Feature Set")
plt.title("Model Accuracy Across Progressive Feature Engineering Subsets")
plt.ylim(0.4, 1.0)
plt.ylabel("Test Set Accuracy Score")
plt.show()

# --- 6. CHOSEN MODEL EVALUATION: CONFUSION MATRIX ---
# Selecting Random Forest on Feature Set 4 (producing ~99% accuracy)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train[feature_set_4], y_train)
y_pred_best = best_model.predict(X_test[feature_set_4])

# Generate Confusion Matrix
classes = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred_best, labels=classes)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix: Top Classifier (Random Forest - Set 4)")
plt.xlabel("Predicted Exercise Class")
plt.ylabel("Actual Exercise Class")
plt.show()