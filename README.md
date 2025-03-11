# Extraction-and-Analysis-of-Multi-modal-Dataset-for-Mental-Disorder-Detection
Multi-modal Mental Disorder Analysis
This repository contains code, data processing scripts, and analysis results for a multi-modal study aimed at developing objective biomarkers for Major Depressive Disorder (MDD). The project integrates EEG, behavioral, and audio data with clinical questionnaires to explore the neurophysiological and cognitive profiles associated with depression.

Overview
Depression diagnosis traditionally relies on clinical interviews and self-report scales. Our project leverages a multi-modal dataset—comprising EEG recordings (from both a 128-channel HydroCel Geodesic Sensor Net and a wearable 3-channel frontal system), behavioral data from a dot-probe task, and audio recordings—to identify potential neurophysiological markers of depression. Clinical questionnaires (PHQ-9, GAD-7, PSQI, CTQ, LES, SCSQ, EPQ-RSC, SSRS) provide additional context and enable correlation analyses between brain signals and symptom severity.

Key Components
EEG Analysis:

Resting State: Preprocessing, Power Spectral Density (PSD) analysis, and time-series inspection from both 128-channel and 3-channel systems.
Dot Probe Task: Event-related potential (ERP) extraction, time–frequency representations (TFR), and PSD analysis during an emotional face-pair task.
Feature Extraction: Computation of canonical frequency bands (delta, theta, alpha, beta, gamma).
Behavioral Analysis:

Extraction of reaction times (RT) and accuracy (ACC) from the dot-probe task.
Group comparisons (MDD vs. HC) and logistic regression classification.
Model Interpretability:

Use of SHAP analysis to interpret the contributions of individual EEG features in predictive models.
Visualization of SHAP force and summary (beeswarm) plots.
Unsupervised Techniques:

Principal Component Analysis (PCA) and k-means clustering to uncover latent structures and assess feature collinearity.
Clinical Correlations:

Investigation of associations between EEG power measures and clinical scales (PHQ-9, GAD-7, PSQI) as well as psychosocial indicators (CTQ, LES, SCSQ, EPQ-RSC, SSRS).
Repository Structure
/data:

Contains raw and processed datasets (EEG, behavioral, audio, and clinical data).
/scripts:

Python scripts for data preprocessing, EEG analysis (PSD, ERP, TFR), statistical tests, machine learning models (logistic regression, PCA), and SHAP analysis.
Example scripts:
preprocess_eeg.py – Preprocessing raw EEG data.
psd_analysis.py – Computation of PSD using Welch’s method.
dot_probe_analysis.py – ERP and TFR extraction for the dot probe task.
behavioral_analysis.py – Statistical analysis of reaction times and accuracy.
model_interpretation.py – SHAP analysis for model interpretability.
pca_clustering.py – PCA and clustering analysis on EEG features.
/results:

Output files (CSV, TXT, PNG) containing descriptive statistics, group comparison results, regression outputs, confusion matrices, SHAP plots, PCA scatter plots, and correlation matrices.
Report.docx:

A detailed report summarizing the project methodology, analyses, and findings.
How to Run the Project
Installation:
Ensure you have Python 3.7+ installed. Install required packages via:

bash
Copy
pip install pandas numpy scipy statsmodels scikit-learn mne openpyxl shap
Additional packages such as mne-microstates may be required for specific analyses.

Data Setup:
Place the raw data files (EEG, behavioral logs, clinical questionnaires) into the /data folder as per the provided directory structure.

Executing Scripts:
Run the analysis scripts in sequence, or execute them individually to generate intermediate outputs:

Preprocess EEG data: python scripts/preprocess_eeg.py
Run PSD analysis: python scripts/psd_analysis.py
Analyze dot probe data: python scripts/dot_probe_analysis.py
Perform behavioral analysis: python scripts/behavioral_analysis.py
Interpret model outputs with SHAP: python scripts/model_interpretation.py
Perform PCA and clustering: python scripts/pca_clustering.py
Review Results:
Check the /results folder for all generated outputs including PSD overlays, ERP/TFR plots, statistical test results, confusion matrices, SHAP plots, and PCA scatter plots.

Project Highlights
Resting State Analysis:
Our resting state EEG analysis revealed trends toward reduced alpha power in MDD subjects (approximately 1.00×10⁻⁵) compared to HC (around 1.20×10⁻⁵), though differences were not statistically significant (t = 0.296, p = 0.7688). Weak correlations with clinical scales indicate that these power measures alone may not be sensitive enough for diagnostic purposes.

Dot Probe Analysis:
In the dot probe task, behavioral measures showed significant differences in reaction times (MDD: ~450 ms vs. HC: ~410 ms, t = 9.536, p < 0.0001) and ERP analyses indicated attenuated responses in MDD. However, EEG power metrics during the task did not differentiate the groups significantly, suggesting that task-evoked neural reactivity may require more refined features for robust discrimination.

Model Interpretability:
SHAP analysis consistently ranked alpha power as the most influential feature, though overall model sensitivity for MDD was low (~15%), emphasizing the need for multi-modal feature integration.

Unsupervised Analysis:
PCA revealed that nearly 100% of the variance was captured by a single principal component, highlighting high collinearity among EEG power measures. Clustering analysis (via k-means) suggests latent subgroups that could reflect distinct electrophysiological subtypes.

Future Directions
The current findings underscore that while EEG power metrics provide valuable insights into cortical activity, they may lack sufficient discriminatory power when used in isolation. Future work should focus on fusing EEG features with behavioral, audio, and clinical data to develop a more sensitive and specific biomarker framework for depression. Techniques such as functional connectivity, source localization, and microstate analysis could further enhance our understanding of the neurophysiological underpinnings of MDD and support the development of personalized diagnostic tools.

Acknowledgments
This project was supported in part by the National Key Research and Development Program of China (Grant No. 2019YFA0706200), the National Natural Science Foundation of China (Grant Nos. 61632014, 61627808, 61210010), the National Basic Research Program of China (973 Program, Grant No. 2014CB744600), and the Program of Beijing Municipal Science & Technology Commission (Grant No. Z171100000117005).

References
Please refer to the provided report and documentation for complete reference details.

