import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline
from scipy import interp

### note
# please install Dask library and mlxtend
# pip install dask[complete] toolz cloudpickle
# pip install mlxtend

from mlxtend.plotting import plot_confusion_matrix
import dask.dataframe as dd
import sys
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('latin-1')
if __name__ == '__main__':

    current_file = os.path.abspath(os.path.dirname(__file__))
    csv_filename = os.path.join(current_file, 'data/train.csv')

    requiredColumns = ["Sex", "MaritalStatus", "Age", "AreaCode", "PaymentDay", "ResidenceType", "MonthsInResidence","MothersName","FathersName",
              "WorkingTown", "WorkingState", "MonthsInJob", "ProfessionCode", "NetIncome", "OtherCards", "AdditionalCards"]

    categoricalColumns =['Sex','MaritalStatus','ResidenceType','MothersName','FathersName','WorkingTown', 'WorkingState','OtherCards']

    cols = ["","IDShop","Sex","MaritalStatus","Age","QuantDependents","Education","ResidencialPhone","AreaCode","PaymentDay",
            "ShopRank","ResidenceType","MonthsInResidence","MothersName","FathersName","WorkingTown","WorkingState","MonthsInJob",
            "ProfessionCode","MateIncome","PostalAddress","OtherCards","QuantBankAccounts","Reference1","Reference2",
            "MobilePhone","ContactPhone","NetIncome","ApplicationBooth","AdditionalCards","InsuranceOption","Label"]

    df = pd.read_csv(csv_filename, sep=',', encoding="latin-1",
                     names=cols, header=0, dtype='str')

    # print df.index.tolist()
    # exit()
    train = df[requiredColumns]

    # Define target feature
    target = 'Label'

    RANDOM_STATE = 42
    LW = 2

    cv = StratifiedKFold(n_splits=3)

    y = df["Label"].astype(int)
    idx = [11639, 11641, 11642, 11643, 11644, 11645, 11646, 11647, 11648, 11649]

    training = pd.get_dummies(df[requiredColumns], columns=categoricalColumns)

    X = training

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lr = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag')
    # print X.dtypes.tolist()
    # exit()
    # for train, test in cv.split(X, y):
    #     print X.iloc[train].astype(float)
    #
    # exit()
    stdsc = StandardScaler()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    Xstd = stdsc.fit_transform(X)
    ypred = cross_val_predict(lr, Xstd, y, n_jobs=-1, verbose=1, cv=10)
    print metrics.accuracy_score(y, ypred)

    print metrics.classification_report(y, ypred)

    fpr, tpr, threshold = metrics.roc_curve(y, ypred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    classes = np.unique(y)

    binary = confusion_matrix(y, ypred, classes)

    fig, ax = plot_confusion_matrix(conf_mat=binary)
    plt.show()