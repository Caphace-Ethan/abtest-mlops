import numpy as np
from numpy import mean, std
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
import warnings

import mlflow
import mlflow.sklearn


@st.cache
def loadData():
    warnings.filterwarnings('ignore')

    v_browser = pd.read_csv('../data/v_browser.csv')
    v_platform = pd.read_csv('../data/v_platform.csv')
    return v_browser, v_platform

if __name__ == "__main__":
    mlflow.set_experiment(experiment_name='Experiment1')
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)