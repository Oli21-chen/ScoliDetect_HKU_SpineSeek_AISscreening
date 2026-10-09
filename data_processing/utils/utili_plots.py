import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics

def plot_metrics(history):
    # metrics =['loss', 'categorical_accuracy','Specificity','Sensitivity','precision_co','precision_pt']
    # pdb.set_trace()
    metrics = ['loss', 'screen_accuracy']  # ,'severity_loss','severity_accuracy']#,'co_recall','pt_recall'mse
    plt.figure(figsize=(15,10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1] + 0.1])
        elif metric == 'severity_accuracy':
            plt.ylim([0, 1.2])

    plt.legend()
    plt.tight_layout()
    plt.show()
    return


def plot_cm(labels,
            predictions,
            th=0.5,
            draw=True,
            screen=True,
            severity=True):
    # if len(predictions)>1:
    #     predictions = np.squeeze(predictions)

    if screen:
        predictions[predictions > th] = 1
        predictions[predictions <= th] = 0
        cm = confusion_matrix(labels, predictions)
        print(cm)
        if draw == True:
            display = ConfusionMatrixDisplay(cm).plot()
            display.plot()
            #
    if severity:
        cm = confusion_matrix(labels, predictions)
        if draw == True:
            plt.figure(figsize=(8, 8))
            sns.set(font_scale=1.4)
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title('Confusion matrix of severity')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.show()
    return

def roc_auc(y_test,
            P_hat_prob,
            query_th=0.5):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, P_hat_prob)
    roc_auc = metrics.roc_auc_score(y_test, P_hat_prob)
    
    idx = np.argmin(np.abs(thresholds - query_th))
    _fpr1, _tpr1, _threshold1 = fpr[idx], tpr[idx], thresholds[idx]
    
    fig = plt.figure(figsize=(6.4, 4.8), dpi=70)
    plt.plot(fpr, tpr, label='AUC=%.3f' %(roc_auc))
    
    plt.plot(_fpr1, _tpr1, 'o', label='threshold: %.2f' %(_threshold1))
   
    
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.plot([0, 0, 1], [0, 1, 1], '--', color='purple')
    
    plt.grid(alpha=0.4)
    
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("ROC curve for prediction")
    plt.legend()
    plt.show()
    
    return roc_auc

def plt_kinetic_sig(t,name):
    '''
    t (B,T,N)
    '''
    # Ensure name is string
    name = str(name)   
    nums, xlen, _ = np.shape(t)
    time_axis = np.arange(xlen)
    plt.figure(figsize=(10, 6))
    for i in range(nums):
        # Average across features for each time point
        signal = np.mean(t[i], axis=1)
        # Normalize to 0-1 range
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        plt.plot(time_axis, signal, label=f'Signal {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Normalized Signal')
    plt.title('Average Feature Signals Over Time- {}'.format(name))
    plt.grid(True)
    plt.show()
    return