import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


excel_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\output\final_merged_EEG_128channels_analysis.xlsx"
output_dir = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\output"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_excel(excel_path)

print("Columns in Excel:")
print(df.columns)



group_map = {"HC": 0, "MDD": 1}
df["Group_encoded"] = df["Group"].map(group_map)

# Drop any rows that have missing data in your columns of interest:
df = df.dropna(subset=["Group_encoded", "Avg_Alpha_Power", "PHQ-9", "CTQ-SF",
                       "LES", "SSRS", "GAD-7", "PSQI", "PCA1", "PCA2"])


features = [
    "Avg_Alpha_Power",
    "PHQ-9",
    "CTQ-SF",
    "LES",
    "SSRS",
    "GAD-7",
    "PSQI",
    "PCA1",
    "PCA2"
]
X = df[features].values
y = df["Group_encoded"].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


clf = LogisticRegression()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]  # For ROC curve


class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print("\nROC AUC:", roc_auc)


report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Classification Report\n")
    f.write(class_report)
    f.write(f"\nConfusion Matrix:\n{cm}")
    f.write(f"\nROC AUC: {roc_auc:.4f}\n")

print(f"\nClassification report saved to: {report_path}")


plt.figure()
plt.plot(fpr, tpr, label=f"Logistic (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")

roc_curve_path = os.path.join(output_dir, "logistic_roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()

print(f"ROC curve saved to: {roc_curve_path}")
print("\nDone!")
