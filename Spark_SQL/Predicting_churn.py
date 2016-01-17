#
# Python - Predicting customer churn with scikit-learn
# Credit to Eric Chiang from yHat
# (C) 2014 Daniel Dixey
#

# Import Modules
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from ggplot import *
import warnings

# Functions


def run_cv(X, y, clf_class, **kwargs):
    # Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    # Return Predicted Results
    return y_pred

# Function - Obtain the accuracy


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred) * 100

# producing predictions


def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


def calibration(prob, outcome, n_bins=10):
    """Calibration measurement for a set of predictions.
    When predicting events at a given probability, how far is frequency
    of positive outcomes from that probability?
    NOTE: Lower scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete"
        probabilities aren't required.
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob, judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += np.sum(in_bin) * ((predicted_prob - true_bin_prob) ** 2)
    return c / len(prob)


def discrimination(prob, outcome, n_bins=10):
    """Discrimination measurement for a set of predictions.
    For each judgement category, how far from the base probability
    is the true frequency of that bin?
    NOTE: High scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete"
        probabilities aren't required.
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    d = 0.0
    # Base frequency of outcomes
    base_prob = np.mean(outcome)
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob, judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between true and base times num of obs
        d += np.sum(in_bin) * ((true_bin_prob - base_prob) ** 2)
    return d / len(prob)


def print_measurements(pred_prob):
    churn_prob, is_churn = pred_prob[:, 1], y == 1
    print "  %-20s %.4f" % ("Calibration Error", calibration(churn_prob, is_churn))
    print "  %-20s %.4f" % ("Discrimination", discrimination(churn_prob, is_churn))
    print "Note -- Lower calibration is better, higher discrimination is better"

# Start Algorithm:
if __name__ == "__main__":
    # Read in Data
    churn_df = pd.read_csv(
        'https://raw.githubusercontent.com/EricChiang/churn/master/data/churn.csv')
    # Obtain a List of the Column Headers
    col_names = churn_df.columns.tolist()
    # Get a refined list of Features
    to_show = col_names[:6] + col_names[-6:]
    # Check the Subsetting of the Dataset has been applied correctly
    churn_df[to_show].head(6)
    # Isolate target data
    churn_result = churn_df['Churn?']
    # Get a Boolean Label Of the Churn Result
    y = np.where(churn_result == 'True.', 1, 0)
    # Remove columns that are not required
    to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
    churn_feat_space = churn_df.drop(to_drop, axis=1)
    # 'yes'/'no' has to be converted to boolean values
    # NumPy converts these from boolean to 1. and 0. later
    yes_no_cols = ["Int'l Plan", "VMail Plan"]
    churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
    # Pull out features for future use
    features = churn_feat_space.columns
    # Getting Feature List for training in the future
    X = churn_feat_space.as_matrix().astype(np.float)
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print "Feature space holds %d observations and %d features" % X.shape
    print "Unique target labels:", np.unique(y)
    # End of Preprocessing of Data Points

    # Start Testing Models
    print "\nTesting Models (Accuracy):"
    print "Support vector machines:"
    print "%.3f" % accuracy(y, run_cv(X, y, SVC))
    print "Random forest:"
    print "%.3f" % accuracy(y, run_cv(X, y, RF))
    print "K-Nearest-neighbors:"
    print "%.3f" % accuracy(y, run_cv(X, y, KNN))
    # Create Labels
    class_names = np.unique(y)
    # Calculate Confusion Matrix
    confusion_matrices = [
        ("Support Vector Machines", confusion_matrix(y, run_cv(X, y, SVC))),
        ("Random Forest", confusion_matrix(y, run_cv(X, y, RF))),
        ("K-Nearest-Neighbors", confusion_matrix(y, run_cv(X, y, KNN)))]
    # Display Confusion Matrix Graphic
    for model in confusion_matrices:
        # Show confusion matrix in a separate window
        plt.matshow(model[1])
        plt.title('Confusion Matrix: ' + model[0])
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    warnings.filterwarnings('ignore')
    # Use 10 estimators so predictions are all multiples of 0.1
    pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
    pred_churn = pred_prob[:, 1]
    is_churn = y == 1
    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_churn)
    # calculate True probabilities
    true_prob = {}
    for prob in counts.index:
        true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)
    # Join Dataframes
    counts = pd.concat([counts, true_prob], axis=1).reset_index()
    counts.columns = ['pred_prob', 'count', 'true_prob']
    counts
    # Produce PLOT
    baseline = np.mean(is_churn)
    ggplot(counts, aes(x='pred_prob', y='true_prob', size='count')) + \
        geom_point(color='blue') + \
        stat_function(fun=lambda x: x, color='red') + \
        stat_function(fun=lambda x: baseline, color='green') + \
        xlim(-0.05,  1.05) + ylim(-0.05, 1.05) + \
        ggtitle("Random Forest") + \
        xlab("Predicted probability") + ylab("Relative frequency of outcome")
    # GET Results
    print "Support vector machines:"
    print_measurements(run_prob_cv(X, y, SVC, probability=True))
    print "Random forests:"
    print_measurements(run_prob_cv(X, y, RF, n_estimators=18))
    print "K-nearest-neighbors:"
    print_measurements(run_prob_cv(X, y, KNN))
