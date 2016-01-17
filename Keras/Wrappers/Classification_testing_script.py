# -*- coding: utf-8 -*-
'''A basic script for testing binary classification models

Input
     - Currently setup to read in Numpy Datasets for both X and y
     - Define your model files and datasets
    
Output
      - Confusion Matrix
      - ROC Curver
      - JSON Document with model metrics

'''


from keras.models import model_from_json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import roc_curve, auc, recall_score, log_loss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import logging
from time import time


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] - [%(levelname)s] - [%(message)s]")


t0 = time()
# Parameters for the model and dataset
directory = "[SOME LOCATION]"        
named_datasets = {'Dataset_1': {'X': 'X_Dataset_1.txt',
                                     'y': 'y_Dataset_1.txt'},
                  'Dataset_2': {'X': 'X_Dataset_2.txt',
                                     'y': 'y_Dataset_2.txt'}}
model_weights = "model_weights.h5"
model_config = "model_architecture.json"                                


def plot_confusion_matrix(
        data, title='_Confusion Matrix_', cmap=plt.cm.Blues, name=''):
    logging.debug("Producing Chart - Confusion Matrix - {}".format(name))
    plt.imshow(data,
               interpolation='nearest',
               cmap=cmap)
    plt.title(title)
    plt.colorbar()
    labels = np.array(['Negative',
                       'Positive'])
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,
               labels,
               rotation=45)
    plt.yticks(tick_marks,
               labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./Plots/" + name + title + '.png',
                bbox_inches='tight')


def plot_roc_curve(fpr, tpr, roc_auc, name=''):
    logging.debug("Producing Chart - ROC Curve - {}".format(name))
    plt.figure()
    plt.plot(fpr,
             tpr,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             'k--')
    plt.xlim([0.0,
              1.0])
    plt.ylim([0.0,
              1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("./Plots/" + name + '_ROC_Curve.png',
                bbox_inches='tight')


logging.info("Loading the Model Architecture")
t1 = time()
model = model_from_json(open().read())
logging.info(
    "Loading the Model Weights - Time to Load Architecture: {}s".format(time() - t1))
t1 = time()
model.load_weights(model_weights)
logging.info("Time to Load Weights: {}s".format(time() - t1))


for each_name in named_datasets.keys():
    t2 = time()
    logging.info("Testing Dataset: {}".format(each_name))

    dataset_name = each_name
    training_dataset = named_datasets[each_name]["X"]
    known_labels = named_datasets[each_name]["y"]
    logging.debug(
        "Getting Dataset Attributes - X: {} - y: {}".format(training_dataset, known_labels))

    logging.debug("Loading the X Dataset")
    X = np.loadtxt(fname=directory + training_dataset,
                   delimiter=' ')

    logging.info("Getting Prediction for '{}'".format(dataset_name))
    y_pred = model.predict_classes(X, batch_size=128, verbose=1)

    del X

    logging.info("Calculating Metrics")
    y_true = np.loadtxt(fname=directory + known_labels,
                        delimiter=' ')
    acc_score = accuracy_score(y_true,
                               y_pred)
    F1_score = f1_score(y_true,
                        y_pred)
    precision = precision_score(y_true,
                                y_pred)
    logLoss = log_loss(y_true,
                       y_pred)
    recall = recall_score(y_true,
                          y_pred)
    fpr, tpr, _ = roc_curve(y_true,
                            y_pred)
    cm = confusion_matrix(y_true,
                          y_pred)
    roc_auc = auc(fpr,
                  tpr)
    plot_confusion_matrix(cm, name=dataset_name)
    del y_true
    del y_pred
    logging.info("Dataset Results: Acc - {}, F1 - {}, Precision - {}, Recall - {}, LogLoss - {}".format(acc_score * 100,
                                                                                                  F1_score * 100,
                                                                                                        precision * 100,
                                                                                                        recall * 100,
                                                                                                        logLoss))

    plot_roc_curve(fpr,
                   tpr,
                   roc_auc,
                   name=dataset_name)

    logging.debug("Saving Results to Dictionary")
    json_document = {
        "dataset_name": dataset_name,
        "training_dataset": training_dataset,
        "known_labels": known_labels,
        "acc_score": acc_score * 100,
        "f1_score": F1_score * 100,
        "precision": precision * 100,
        "logLoss": logLoss,
        "recall": recall * 100,
        "roc_auc": roc_auc * 100,
        "plot_roc_fname": dataset_name + '_roc_curve.png',
        "plot_confusion_matrix_fname": dataset_name + '_confusion_matrix.png',
        "Upper_Left": cm[0, 0],
        "Upper_Right": cm[0, 1],
        "Lower_Left": cm[1, 0],
        "Lower_Right": cm[1, 1]
    }

    logging.info("Results for '{}' Saved".format(dataset_name))
    with open("./Results/" + dataset_name + '_result.json', 'w') as fp:
        json.dump(json_document, fp)

    logging.info("Time to Process '{}': {}".format(dataset_name,
                                                   time() - t2))
