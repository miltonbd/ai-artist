import numpy as np
 
# These are the labels we predicted.
pred_labels = np.asarray([0,1,1,0,1,0,0])
print 'pred labels:\t\t', pred_labels
 
# These are the true labels.
true_labels = np.asarray([0,0,1,0,0,1,0])
print 'true labels:\t\t', true_labels

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
 
print 'TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN)