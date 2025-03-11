import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


data_excel = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\EEG_bandpower_features.xlsx"
output_excel = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\logistic_regression_results.xlsx"
output_txt = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\logistic_regression_results.txt"

df = pd.read_excel(data_excel)
print("Data loaded. Columns:", df.columns.tolist())


if df['Group'].dtype == 'object':
    df['Group'] = df['Group'].map({'MDD': 1, 'HC': 0})


feature_cols = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]


df = df.dropna(subset=feature_cols + ['Group'])


X = df[feature_cols].values
y = df['Group'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


clf = LogisticRegression(solver='liblinear', random_state=42)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=skf)


cv_report = classification_report(y, y_pred_cv, target_names=["HC", "MDD"])
print("=== Cross-Validated Classification Report ===")
print(cv_report)


clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
test_report = classification_report(y_test, y_pred_test, target_names=["HC", "MDD"])
print("=== Test Set Classification Report ===")
print(test_report)


cm = confusion_matrix(y_test, y_pred_test)
print("Test Set Confusion Matrix:")
print(cm)


with open(output_txt, "w", encoding="utf-8") as f:
    f.write("=== Cross-Validated Classification Report ===\n")
    f.write(cv_report + "\n\n")
    f.write("=== Test Set Classification Report ===\n")
    f.write(test_report + "\n\n")
    f.write("Test Set Confusion Matrix:\n")
    f.write(np.array2string(cm))
print(f"Results saved to text file: {output_txt}")


cv_report_lines = cv_report.splitlines()
df_cv = pd.DataFrame(cv_report_lines, columns=["Cross-Validated Report"])

test_report_lines = test_report.splitlines()
df_test = pd.DataFrame(test_report_lines, columns=["Test Set Report"])

df_cm = pd.DataFrame(cm, index=["HC", "MDD"], columns=["Pred_HC", "Pred_MDD"])

with pd.ExcelWriter(output_excel) as writer:
    df_cv.to_excel(writer, sheet_name="CV_Report", index=False)
    df_test.to_excel(writer, sheet_name="Test_Report", index=False)
    df_cm.to_excel(writer, sheet_name="Confusion_Matrix")
print(f"Results saved to Excel file: {output_excel}")
