import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

# loading the input data and exploring the class distribution
train_data = pd.read_csv("A3_TrainData.tsv", delimiter="\t")
class_distribution = train_data.iloc[:, -1].value_counts(normalize=True)
print("Class Distribution:")
print(class_distribution)

# A plot to show the class distribution
sns.displot(train_data['label'], kde='True')


# 10-fold stratified cross-validation for logistic regression (baseline model)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
lr_base_model = LogisticRegression()  # Using default hyper-parameters
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
lr_base_scores = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lr_base_model.fit(X_train, y_train)
    lr_base_scores.append(lr_base_model.score(X_test, y_test))

print("Mean accuracy of Logistic Regression (baseline model):", np.mean(lr_base_scores))
print("Standard Deviation of Logistic Regression (baseline model):", np.std(lr_base_scores))

# Exploring other classification approaches, perform grid-search cross-validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# 1. Gradient Boosting 
gb_model = GradientBoostingClassifier()
gb_param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [10, 100, 500],
    'subsample' : [0.5, 0.7, 1.0],
    'max_depth': [3, 5, 7]
}
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
gb_grid_search.fit(X, y)
print("Best hyperparameters for Gradient Boosting:", gb_grid_search.best_params_)
print("Best CV score for Gradient Boosting:", gb_grid_search.best_score_)


# Support Vector Machine (SVM)
svm_model = SVC(probability=True)
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
svm_grid_search.fit(X, y)
print("Best hyperparameters for SVM:", svm_grid_search.best_params_)
print("Best CV score for SVM:", svm_grid_search.best_score_)

# display grid search cross-validation results for Gradient Boosting model
gb_means = gb_grid_search.cv_results_['mean_test_score']
gb_stds = gb_grid_search.cv_results_['std_test_score']
gb_params = gb_grid_search.cv_results_['params']
for mean, stdev, param in zip(gb_means, gb_stds, gb_params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# display grid search cross-validation results for Support Vector Machine model
svm_means = svm_grid_search.cv_results_['mean_test_score']
svm_stds = svm_grid_search.cv_results_['std_test_score']
svm_params = svm_grid_search.cv_results_['params']
for mean, stdev, param in zip(svm_means, svm_stds, svm_params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Create precision-recall curves and ROC curves for each method
# Define function to plot PR curve
def plot_precision_recall_curve(y_true, y_score, label, ax):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    ax.plot(recall, precision, label=f"{label} (AP={average_precision:.3f})", lw=2)

# Define function to plot ROC curve
def plot_roc_curve(y_true, y_score, label, ax):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})", lw=2)

# Get predictions from each model
lr_y_base_score = lr_base_model.predict_proba(X)[:, 1]
gb_y_score = gb_grid_search.best_estimator_.predict_proba(X)[:, 1]
svm_y_score = svm_grid_search.best_estimator_.predict_proba(X)[:, 1]


# Initialize figure 1
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
# Plot precision-recall curves
plot_precision_recall_curve(y, lr_y_base_score, "Base Logistic Regression", ax1)
plot_precision_recall_curve(y, gb_y_score, "Gradient Boosting", ax1)
plot_precision_recall_curve(y, svm_y_score, "Support Vector Machine", ax1)
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curves')
ax1.legend()
plt.tight_layout()
plt.show()

# Initialize figure 2
fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))
# Plot ROC curves
plot_roc_curve(y, lr_y_base_score, "Base - Logistic Regression", ax2)
plot_roc_curve(y, gb_y_score, "Gradient Boosting", ax2)
plot_roc_curve(y, svm_y_score, "Support Vector Machine", ax2)
# Plotting random classifier (baseline)
ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves')
ax2.legend()
plt.tight_layout()
plt.show()


# Select best model overall, create  final model for all of the training data.
# compute AUROC scores for each model
lr_auc = roc_auc_score(y, lr_y_base_score)
gb_auc = roc_auc_score(y, gb_y_score)
svm_auc = roc_auc_score(y, svm_y_score)

print("Linear Regression ROC = ", lr_auc)
print("Random Forest ROC = ", gb_auc)
print("Support Vector Machine ROC = ", svm_auc)

# choose the best model based on AUROC score
best_model = None
if lr_auc >= gb_auc and lr_auc >= svm_auc:
    best_model = lr_base_model
    best_model_name = "Logistic Regression"
elif gb_auc >= lr_auc and gb_auc >= svm_auc:
    best_model = gb_grid_search.best_estimator_
    best_model_name = "Gradient Boosting"
else:
    best_model = svm_grid_search.best_estimator_
    best_model_name = "Support Vector Machine"

print("Best model = ", best_model_name)

# fitting the best model on all training data
best_model.fit(X, y)


# Step 6: Use the final model to predict the likelihood to belong to class 1 for the test instances
# Read the test data
test_data = pd.read_csv("A3_TestData.tsv", delimiter="\t")

# Predict the probabilities for the test instances using the final model
test_y_proba = best_model.predict_proba(test_data)[:, 1]

# Writing the predicted probabilities to a text file
with open("A3_predictions_202286594.txt", "w") as f:
    for prob in test_y_proba:
        f.write(f"{prob}\n")


'''
Intructions on how to run the code...
1. Open a Terminal or Command Prompt:
Open a terminal or command prompt window on your operating system. 

2. Navigate to the Directory Containing The Python File:
Use the cd command to navigate to the directory where the .py file is located. 
Ensure the dataset A2data.tsv is also located in this directory

cd path/to/directory/assignment3

3. Run the Python File:
Once you're in the correct directory, you can run your Python file by typing python followed by the name of the .py file:

python 202286594_Daniel_Wiredu_A3.py

If you have multiple versions of Python installed on your system, you may need to specify the version you want to use. 
For Python 3, you would use the command python3 instead:

python3 202286594_Daniel_Wiredu_A3.py

4. Review Output:
After running the command, the Python program will execute, and any output or errors will be displayed in the terminal window. 
'''
