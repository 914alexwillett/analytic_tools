# import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from scipy import stats
# from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def logisticRegress(data, _y, _X: list, _label_names=[0, 1], _random_state=9):
    """
    easily fits a logistic regression model and evaluates
    its efficacy
    data: dataframe of the data in use
    _y: single variable
    _x: single or list of variarles
    
    
    """
    # Churn vs Payment Delay
    y = data[_y]
    if len(_X) == 1:
        X = np.array(data[_X]).reshape(-1, 1)
    else:
        X = data[_X]
        
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=_random_state)
    
    # instantiate the model (using default parameters)
    logreg = LogisticRegression(random_state=16)
    
    # fit the model with the data
    logreg.fit(X_train, y_train)
    
    # testing the model
    y_pred = logreg.predict(X_test)
    
    # confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    # plotting the confusion matrix
    class_names = _label_names
    plt.subplot(2, 1, 1)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
#     ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    target_names = _label_names
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 2)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    
    return classification_report(y_test, y_pred, target_names=target_names)

def calc_linear_regression(_x, _y):
    x, y = _x, _y
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    y_predict = []
    for item in x:
        y_predict.append((slope * item) + intercept)
    r2 = r2_score(y, y_predict)
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('Pearson R-score:', r)
    print('P-value:', p) 
    print('Standard Error:', std_err)
    print('R-squared:', r2)
    plt.rcParams['figure.figsize'] = 9, 7 
    sns.regplot(x=x, y=y)
    return [slope, intercept, r, p, std_err]

def make_MA_col(df, column, window, name):
    """
    creates a moving average for a specified window length
    df: dataframe
    column: str
    window: int
    name: str
    """
    col = pd.Series(df[column])
    colMA = col.rolling(window).mean()
    df[name] = colMA
    return df