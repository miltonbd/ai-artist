import numpy as np
from sklearn import metrics

"""
Receiver Operating Characteristics
"""

def cofusionMetrics():
    pass

def precisionVsRecall():
    pass

def averagePrecision():
    pass


def ROCcurve():
    return


if __name__ == '__main__':
    y_true = [2, 0, 2, 2, 9, 1, 6, 8, 8, 2, 2, 0]
    y_pred = [0, 0, 1, 2, 0, 2, 6, 7, 8, 0, 0, 0]

    accuracy = metrics.accuracy_score(y_true, y_pred)

    print(accuracy)

    """
    total sum of confusion matrix value is same as total number items in test set.
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    auc = metrics.roc_auc_score(y_true, y_pred)
    print(auc)

    #f1_score = metrics.f1_score(y_true, y_pred)

    #print(f1_score)

    # average_precision = metrics.average_precision_score(y_true, y_pred)
    #
    # print(average_precision)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print(TP)
    print(TN)
    #
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    #print(ACC.mean())
