import numpy as np
from sklearn import metrics

# not in use yet

def eval(preds, y_true, test=False):
    y_pred_label = np.asarray([np.argmax(pred) for pred in preds])
    accuracy = metrics.accuracy_score(y_true, y_pred_label)
    f1_weighted = metrics.f1_score(y_true, y_pred_label, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred_label, average='macro')
    f1_micro = metrics.f1_score(y_true, y_pred_label, average='micro')
    precision_weighted = metrics.precision_score(y_true, y_pred_label, average='weighted')
    precision_macro = metrics.precision_score(y_true, y_pred_label, average='macro')
    precision_micro = metrics.precision_score(y_true, y_pred_label, average='micro')
    recall_weighted = metrics.recall_score(y_true, y_pred_label, average='weighted')
    recall_macro = metrics.recall_score(y_true, y_pred_label, average='macro')
    recall_micro = metrics.recall_score(y_true, y_pred_label, average='micro')
    results = {"accuracy": accuracy,
               "f1_weighted": f1_weighted,
               "f1_macro": f1_macro,
               "f1_micro": f1_micro,
               "precision_weighted": precision_weighted,
               "precision_macro": precision_macro,
               "precision_micro": precision_micro,
               "recall_weighted": recall_weighted,
               "recall_macro": recall_macro,
               "recall_micro": recall_micro
               }