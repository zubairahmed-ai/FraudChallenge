import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from imblearn.pipeline import make_pipeline
from scipy import interp
from  sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':

    current_file = os.path.abspath(os.path.dirname(__file__))
    csv_filename = os.path.join(current_file, 'data/train.csv')

    requiredColumns = ["Sex", "MaritalStatus", "Age", "AreaCode", "PaymentDay", "ResidenceType", "MonthsInResidence",
                       "MothersName", "FathersName",
                       "WorkingTown", "WorkingState", "MonthsInJob", "ProfessionCode", "NetIncome", "OtherCards",
                       "AdditionalCards"]

    categoricalColumns = ['Sex', 'MaritalStatus', 'ResidenceType', 'MothersName', 'FathersName', 'WorkingTown',
                          'WorkingState', 'OtherCards']

    cols = ["", "IDShop", "Sex", "MaritalStatus", "Age", "QuantDependents", "Education", "ResidencialPhone", "AreaCode",
            "PaymentDay",
            "ShopRank", "ResidenceType", "MonthsInResidence", "MothersName", "FathersName", "WorkingTown",
            "WorkingState", "MonthsInJob",
            "ProfessionCode", "MateIncome", "PostalAddress", "OtherCards", "QuantBankAccounts", "Reference1",
            "Reference2",
            "MobilePhone", "ContactPhone", "NetIncome", "ApplicationBooth", "AdditionalCards", "InsuranceOption",
            "Label"]

    df = pd.read_csv(csv_filename, sep=',', encoding="latin-1",
                     names=cols, header=0, dtype='str')

    train = df[requiredColumns]

    # Define target feature
    target = 'Label'
    RANDOM_STATE = 42
    LW = 2
    cv = StratifiedKFold(n_splits=3)

    y = df["Label"].astype(int)

    X = pd.get_dummies(df[requiredColumns], columns=categoricalColumns)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    gbm0 = GradientBoostingClassifier(random_state=10, learning_rate=0.3, n_estimators=5000, min_samples_leaf=70,
                                      min_samples_split=900)

    samplers = [['SMOTE', SMOTE(random_state=RANDOM_STATE, ratio=1.0, kind='borderline1')]]
    classifier = ['gbm', gbm0]
    pipelines = [
        ['{}-{}'.format(sampler[0], classifier[0]),
         make_pipeline(sampler[1], classifier[1])]
        for sampler in samplers
    ]
    stdsc = StandardScaler()
    cv = StratifiedKFold(n_splits=3)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    Xstd = stdsc.fit_transform(X)
    scores = []
    recalls = []
    precisions = []
    confusion = np.array([[0, 0], [0, 0]])
    for name, pipeline in pipelines:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for tr, ts in cv.split(Xstd, y):
            xtrain = Xstd[tr]
            ytrain = y[tr]
            test = y[ts]
            xtest = Xstd[ts]
            pipeline.fit(xtrain, ytrain)
            probas_ = pipeline.predict_proba(xtest)
            fpr, tpr, thresholds = roc_curve(test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)

            predictions = pipeline.predict(xtest)
            confusion += confusion_matrix(test, predictions)

            precision, recall, fscore, support = score(test, predictions)
            recalls.append(recall)
            precisions.append(precision)
            scores.append(fscore)

        mean_tpr /= cv.get_n_splits(Xstd, y)
        mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(mean_auc)

    print('Score:', sum(scores) / len(scores))
    print('Recall:', sum(recalls) / len(recalls))
    print('Precision:', sum(precisions) / len(precisions))
    print('Confusion matrix:')
    print(confusion)

    plt.plot(mean_fpr, mean_tpr, linestyle='--',
             label='(area = %0.2f)'.format(mean_auc, lw=LW))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k',
             label='Luck')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

# 0.624378397915
# ('Score:', array([ 0.86633076,  0.20408241]))
# ('Recall:', array([ 0.92559176,  0.14787228]))
# ('Precision:', array([ 0.81421262,  0.32991565]))
# Confusion matrix:
# [[25961  2087]
#  [ 5924  1028]]