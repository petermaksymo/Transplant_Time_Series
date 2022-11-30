import matplotlib.pyplot as plt
from sklearn import metrics
import torch as torch
import time
import itertools
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os

# from common.dataloader import Dataset, collate_fn, make_train_loader

MODEL_PATH = '../../local_data/models/Transformer/clean/dim64_heads4_levels4/lr0.0005_b1_0.9_b2_0.999_drop0.1_l2_0.01/scripted_best_auc_model.pt'
VALID_PATH = '../../Preprocessing/data/processed_data/valid_tensors/'

CLASS_LABELS = ['lived', 'cardio', 'gf', 'cancer', 'inf']
COLOURS = ['b','g','r','c','m']


def plot_confusion_matrix(orig_cm, path, filename=str(time.time()), normalize=True, title='Confusion matrix', cmap=plt.cm.Greens):
    orig_cm = np.asarray(orig_cm)

    if normalize:
        # Normalize across each row if enabled
        cm = orig_cm.astype('float') / orig_cm.sum(axis=1)[:, np.newaxis]
    else:
        # User original cm otherwise
        cm = orig_cm

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=30)
    #plt.colorbar(aspect=4)
    tick_marks = np.arange(len(CLASS_LABELS))
    plt.xticks(tick_marks, CLASS_LABELS, rotation=45, size=26)
    plt.yticks(tick_marks, CLASS_LABELS, rotation=45, size=26)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Color is based on underlying block color, to make it easier to read
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, format(cm[i, j], fmt), fontsize=24, horizontalalignment="center", color=color)
        if normalize:
            # show raw numbers if normalization is enabled
            plt.text(j, i+0.25, f"{orig_cm[i, j]:^9}", bbox=dict(fill=None, edgecolor=color), fontsize=24, horizontalalignment="center", color=color)

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=28, fontweight='bold')
    plt.xlabel('Predicted label', size=28, fontweight='bold')
    # filename = path + '/' + filename + '.png'
    # plt.savefig(filename, bbox_inches="tight")
    # plt.close()
    return filename


def plot_roc_curves(y_true, y_pred):
    # Plot the ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    for i in range(len(CLASS_LABELS)):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    names = ['Survival', 'Cardiovascular causes', 'Graft failure', 'Cancer', 'Infection']
    for i in range(len(names)):
        ax.plot(fpr[i], tpr[i], label=f'{names[i]} (AUROC {roc_auc[i]:.3f})')
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    plt.show()


def generate_curves(model, val_loader, year, device):
    correct = np.zeros(5)
    pos = np.zeros(5)
    total = 0

    y_pred = np.empty((0, len(CLASS_LABELS)))
    y_true = np.empty((0, len(CLASS_LABELS)))
    with torch.no_grad():
        for batch, labels, seq_len in val_loader:
            labels = labels[:, :, -5:] if year == 1 else labels[:, :, :5]
            # pass to GPU if available
            batch, labels = batch.to(device), labels.to(device)

            softmax = torch.nn.Softmax(dim=2)
            outputs = softmax(model(batch))

            # Validation accuracy
            for i in range(labels.shape[0]):
                targets = labels.data[i][int(seq_len[i]) - 1].unsqueeze(dim=0).cpu().numpy()
                prob = outputs.data[i][int(seq_len[i]) - 1].unsqueeze(dim=0).cpu().numpy()

                prediction = np.zeros(targets.shape)
                prediction[np.arange(prediction.shape[0]), np.argmax(prob, axis=1)] = 1
                match = (targets == prediction)

                pos += np.sum(prediction, axis=0).astype(int)
                correct += np.sum(match, axis=0).astype(int)
                total += targets.shape[0]

                y_pred = np.append(y_pred, prob, axis=0)
                y_true = np.append(y_true, targets, axis=0)

    cm = metrics.confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plot_confusion_matrix(cm, '.')
    plot_roc_curves(y_true, y_pred)


if __name__ == '__main__':
    # LOAD MODEL
    model = torch.jit.load(MODEL_PATH)
    model.eval()

    generate_curves(model)