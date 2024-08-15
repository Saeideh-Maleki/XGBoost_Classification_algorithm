Project Title: XGBoost Classifier for Multi-Class Classification

Description:
This project implements an XGBoost classifier using the xgboost library to perform multi-class classification on a dataset containing four classes: 'Other', 'TRN', 'SOJ', and 'MIS'. The code is designed for spatio-temporal transferability, enabling the creation of maps for one location and time period (e.g., Dijon 2020) using training data from another (e.g., Tarbes 2021). The script evaluates the model's performance by computing various metrics, including accuracy, F1 score, precision, recall, Cohen's kappa, and the confusion matrix.

Script Overview:

Libraries Used:

numpy: For numerical operations.
sys and os: For system operations and handling input/output files.
xgboost: For the XGBoost classifier.
scikit-learn: For additional evaluation metrics.
seaborn and matplotlib: For data visualization, specifically the confusion matrix.

Input Data:

The script expects the training and testing datasets to be in .npz format.
The datasets should include specific arrays (e.g., Polari_as_train, Polari_as_test, y_modified).
Output:

The script generates and saves performance metrics in a text file, and the confusion matrix as a .png image.
The output is saved in the specified directory as a .txt file.
Customization:

Feature Selection:

You can customize the features used for training and testing by modifying the x_train and x_test variables in the script.
Model Parameters:

The XGBoost model's parameters can be adjusted by modifying the param dictionary in the script.
This includes parameters like max_depth, learning_rate, n_estimators, and more.
Requirements:

Python

Required Python packages:
numpy
xgboost
scikit-learn
matplotlib
seaborn
