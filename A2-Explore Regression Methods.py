import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv('A2data.tsv', sep='\t')

#explore dataset
df.info()

sns.displot(df['Y'], kde='True')

#drop column
df = df.drop('InstanceID', axis=1)


# implement a cross-validation model evaluation function
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics

FOLDS = 5

def evaluate_model(classifier, X, y, title='Evaluation'):
    cv = KFold(n_splits=FOLDS)
    
    y_real = []
    y_pred = []

    i = 0
    for train, test in cv.split(X, y):
        classifier.fit(X.iloc[train], y.iloc[train])
        pred = classifier.predict(X.iloc[test])
        ytest = y.iloc[test]

        y_real.append(ytest)
        y_pred.append(pred)

        coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
        #print('coeff:\n', coeff_df)
        
        i += 1
    
    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    
    plt.scatter(y_real,y_pred)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.title(title)
    
    print('MAE:', metrics.mean_absolute_error(y_real, y_pred))
    print('MSE:', metrics.mean_squared_error(y_real, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_real, y_pred)))


# BASE MODEL
# Implement and evaluate Base Model Regression Method 
features = df.columns[:-1]
X = df[features]
y = df['Y']

# Linear Classifier
lm = LinearRegression()

evaluate_model(lm, X, y, 'Base Model')



# PRE-PROCESSING MODEL
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Y',axis=1))
scaled_features = scaler.transform(df.drop('Y',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

# Implement and evaluate Base Model Regression Method 
X1 = df_feat
y1 = df['Y']

# Linear Classifier
lm = LinearRegression()

evaluate_model(lm, X1, y1, 'Pre-Processing Model')



# MODEL3 - FEATURE SELECTION MODEL
# K-best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = df.columns[:-1]
X = df[features]
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.4,
                                                   shuffle=False,
                                                   random_state=42)

mse_score_list = []

for k in range(10, 100, 10):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X_train, y_train)
    
    sel_X_train = selector.transform(X_train)
    sel_X_test = selector.transform(X_test)
    
    lm.fit(sel_X_train, y_train)
    kbest_preds = lm.predict(sel_X_test)
    
    mse_score_kbest = round(mean_squared_error(y_test, kbest_preds), 3)
    
    mse_score_list.append(mse_score_kbest)

    fig, ax = plt.subplots()

# plot feature evaluation
x_featrues = np.arange(10, 100, 10)
y_scores = mse_score_list

ax.bar(x_featrues, y_scores, width=1.5)
ax.set_xlabel('Number of features selected using mutual information')
ax.set_ylabel('Mean Squared Error (weighted)')
ax.set_ylim(0, max(y_scores)+1)
ax.set_xticks(np.arange(10, 100, 10))
ax.set_xticklabels(np.arange(10, 100, 10), fontsize=12)

for i, v in zip(range(10, 100, 10), y_scores):
    plt.text(x=i, y=v+0.1, s=str(v), ha='center')
    
plt.tight_layout()

# get top 10 features
selector = SelectKBest(mutual_info_regression, k=10)
selector.fit(X_train, y_train)

selected_feature_mask = selector.get_support()
selected_features = X_train.columns[selected_feature_mask]
selected_features

# Implement and evaluate selected features regression method
X2 = df[selected_features]
y2 = df['Y']

# Linear Classifier
lm = LinearRegression()

evaluate_model(lm, X2, y2, 'Selected Features Model')


'''
Intructions on how to run the code...
1. Open a Terminal or Command Prompt:
Open a terminal or command prompt window on your operating system. 

2. Navigate to the Directory Containing The Python File:
Use the cd command to navigate to the directory where the .py file is located. 
Ensure the dataset A2data.tsv is also located in this directory

cd path/to/directory/assignment2

3. Run the Python File:
Once you're in the correct directory, you can run your Python file by typing python followed by the name of the .py file:

python 202286594_Daniel_Wiredu_A2.py

If you have multiple versions of Python installed on your system, you may need to specify the version you want to use. 
For Python 3, you would use the command python3 instead:

python3 202286594_Daniel_Wiredu_A2.py

4. Review Output:
After running the command, the Python program will execute, and any output or errors will be displayed in the terminal window. 
'''
