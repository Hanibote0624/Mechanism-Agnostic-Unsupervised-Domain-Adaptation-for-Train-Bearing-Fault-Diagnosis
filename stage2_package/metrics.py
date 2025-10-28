
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score

def compute_metrics(y_true, y_prob, classes):
    y_pred = np.argmax(y_prob, axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pr = {}
    for k, cls in enumerate(classes):
        y_true_bin = (y_true == k).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:,k])
        ap = average_precision_score(y_true_bin, y_prob[:,k])
        pr[cls] = {"precision": precision.tolist(), "recall": recall.tolist(), "AP": float(ap)}
    return {"macro_f1": float(macro_f1),
            "per_class_f1": {classes[i]: float(per_class_f1[i]) for i in range(len(classes))},
            "confusion_matrix": cm.tolist(),
            "pr": pr}

def plot_confusion_matrix(cm, classes, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    xs, ys = [], []
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.any(m):
            acc = np.mean(accuracies[m])
            conf = np.mean(confidences[m])
            w = np.mean(m.astype(float))
            ece += w * abs(acc - conf)
            xs.append(conf); ys.append(acc)
    return float(ece), np.array(xs), np.array(ys)

def plot_reliability_diagram(xs, ys, out_path):
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1], 'k--', linewidth=1)
    plt.scatter(xs, ys, s=20)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title("Reliability")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
