import matplotlib.pyplot as plt
from sklearn import metrics
import torch as torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os

# from common.dataloader import Dataset, collate_fn, make_train_loader

MODEL_PATH = '../../local_data/models/Transformer/clean/dim64_heads4_levels4/lr0.0005_b1_0.9_b2_0.999_drop0.1_l2_0.01/scripted_best_auc_model.pt'
VALID_PATH = '../../Preprocessing/data/processed_data/valid_tensors/'

CLASS_LABELS = ['lived', 'cardio', 'gf', 'cancer', 'inf']
COLOURS = ['b','g','r','c','m']

# parser = argparse.ArgumentParser(description='confusion matrix options')
# parser.add_argument('--disable-cuda', action='store_true',
#                     help='Disable CUDA')
# parser.add_argument('--year', action='store', default=1, type=int,
#                     help='year for prediction')
# args = parser.parse_args()
#
# ######## __GPU_SETUP__ ########
# if not args.disable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
#     torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# else:
#     args.device = torch.device('cpu')
#     torch.set_default_tensor_type('torch.DoubleTensor')


def plot_confusion_matrix(y_true, y_pred):
    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
    disp.plot()


def plot_roc_curves(y_true, y_pred):
    # Plot the ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = np.eye(len(CLASS_LABELS))[y_true.astype(int)]
    y_pred = np.eye(len(CLASS_LABELS))[y_pred.astype(int)]

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

    y_pred = np.array([])
    y_true = np.array([])
    with torch.no_grad():
        for batch, labels, seq_len in val_loader:
            labels = labels[:, :, -5:] if year == 1 else labels[:, :, 5]
            # pass to GPU if available
            batch, labels = batch.to(device), labels.to(device)

            outputs = model(batch)

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

                y_pred = np.append(y_pred, np.argmax(prob, axis=1))
                y_true = np.append(y_true, np.argmax(targets, axis=1))

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_pred)


if __name__ == '__main__':
    # LOAD MODEL
    model = torch.jit.load(MODEL_PATH)
    model.eval()

    generate_curves(model)