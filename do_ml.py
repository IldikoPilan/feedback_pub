import os
import random
import spacy
import scipy
import pickle
import numpy as np
from extract_features import extract_features, extract_features_LA
from collections import Counter
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

def to_float(feature_vals):
    num_vals = []
    for feature_val in feature_vals:
        try:
            num_vals.append(float(feature_val))
        except ValueError:
            num_vals.append(feature_val)
    return num_vals

def get_ml_data_stats(X,y):
    revision_success_mapping = {"same":0, "rem":1, "bad":2, "good":3, "alt":4, "?":5}
    map_back_rs = {v:k for k,v in revision_success_mapping.items()}
    label_distr = Counter()
    for label in y:
        label_distr[label] += 1
    for k,v in label_distr.items():
        print(map_back_rs[int(k)], "\t", v)

def load_ml_data(features_file_name, label_file_name, feat_names_fname):

    with open(features_file_name, newline='') as features_file:
        X = np.array([to_float(line.strip("\n").split(",")) for line in features_file.readlines()])
    with open(label_file_name, newline='') as label_file:
        y = np.array([line.strip("\n") for line in label_file.readlines()])
    with open(feat_names_fname, newline='') as fn_f:
        feature_names = [line.strip("\n") for line in fn_f.readlines()]
    print("Feature names")
    for name in feature_names:
       print(name)
    print()
    return (X, y, feature_names)

def split_train_test(X, y, test_ratio, balance):
    """ mimics sklearn's train_test_split() which raises error
    on dataset.
    """
    random.seed(9)
    random.shuffle(X)
    random.shuffle(y)
    merged = []
    for ix, x in enumerate(X):
        inst = np.append(x,float(y[ix]))
        merged.append(inst)
    if balance:
        revised = [x for x in merged if x[-1] == 3.0][:404]
    else:
        revised = [x for x in merged if x[-1] == 3.0]
    same = [x for x in merged if x[-1] == 0.0]
    test_amount_rev = int(round(len(revised)*test_ratio))
    test_amount_same = int(round(len(same)*test_ratio))
    X_train = np.array([x[:-1] for x in revised[test_amount_rev:]+same[test_amount_same:]])
    y_train = np.array([str(x[-1])[0] for x in revised[test_amount_rev:]+same[test_amount_same:]])
    X_test = np.array([x[:-1] for x in revised[:test_amount_rev]+same[:test_amount_same]])
    y_test = np.array([str(x[-1])[0] for x in revised[:test_amount_rev]+same[:test_amount_same]])
    print("Label distr (train)")
    get_ml_data_stats(X_train,y_train)
    print("Label distr (test)")
    get_ml_data_stats(X_test,y_test)
    print()
    return (X_train, y_train, X_test, y_test)

def eval_features(X_train, y_train, feature_names):
    chi2_vals, p_vals = chi2(X_train, y_train)
    for ix, name in enumerate(feature_names):
        print("{:<20}{:<8}{:<8}".format(name, round(chi2_vals[ix], 3), round(p_vals[ix], 3)))
    print()

def get_classifiers(svm_cl=False):
    b_line = ("Baseline   ", DummyClassifier(strategy="most_frequent"))
    l1_LR_clf = ("Log_Regr   ", LogisticRegression(solver='lbfgs'))
    if svm_cl:
        svc_clf = ("SVM_lin    ", svm.SVC(gamma="scale", C=0.1, kernel="rbf"))
        return [b_line, l1_LR_clf, svc_clf] 
    else:
        return [b_line, l1_LR_clf]

def eval_cl(X,y,clfs,feature_names,cv_folds,test_ratio=0.2, balance=False, select_f=True):
    dev_scores = {} # will contain: {"svm":(acc,f1,prec,recall)} etc
    if select_f:
        selector = SelectKBest(mutual_info_classif, k=12) #chi2 f_classif, mutual_info_classif
        #selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X = selector.fit_transform(X, y)
        feat_scores = sorted(zip(selector.scores_, feature_names), reverse=True)
        for score, name in feat_scores:
            print("{:<20}\t{:<8}".format(name, round(score, 3))) 
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_ratio, balance=balance)
    print(X_train.shape)
    print(X_test.shape)
    #eval_features(X_train, y_train, feature_names)
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    for clf_tuple in clfs:
        clf_name, clf = clf_tuple
        # train
        clf.fit(X_train, y_train)
        # classify
        y_pred = clf.predict(X_test)
        # evaluate
        print(clf_name, round(clf.score(X_test, y_test),3), "test")

def plot_correl(x,y):
    matplotlib.style.use('ggplot')
    plt.scatter(x, y)
    plt.show()

def get_vals_per_feat(X, feature_names):
    # get values for all instances per feature
    vals_per_feat = {}
    for ix, feature_name in enumerate(feature_names):
        vals_per_feat[feature_name] = [row[ix] for row in list(X)]
    return vals_per_feat

def map_labels(y, data_type = "rev_suc"):
    labels = []
    for lbl in list(y):
        if data_type == "rev_suc" and lbl == "2":          # bad
            labels.append(-1)
        elif data_type == "rev_suc" and lbl == "3":        # good
            labels.append(1)
        else:
            try:
                labels.append(int(lbl))
            except ValueError:
                labels.append(float(lbl))
    return labels

def compute_correls(vals_per_feat, labels, feature_names):
    """Calculates a Pearson and Spearman correlation coefficient and the p-value 
    for testing non-correlation between two (or more) continuous variables.
    (two (or more) normally distributed interval variables)
    TO DO: map label to continupus (edit dist) 
    """
    #plot_correl(vals_per_feat["meta_ratio"],y)
    
    print("{:<20}\t{:<8}\t{:<8}\t{:<8}".format("Feat name", "Mean", "Corr", "p"))
    for feat_name, vals in vals_per_feat.items():
        #coef, pval = scipy.stats.pearsonr(vals, labels)
        coef, pval = scipy.stats.spearmanr(vals, labels)
        if pval < 0.05:
            print("{:<20}\t{:<8}\t{:<8}\t{:<8} *".format(feat_name, round(np.mean(vals), 3), round(coef, 3), round(pval, 3)))
        else:
            print("{:<20}\t{:<8}\t{:<8}\t{:<8}".format(feat_name, round(np.mean(vals), 3), round(coef, 3), round(pval, 3)))        
    nr_items = len(vals)
    print("Nr items: ", nr_items)
    print("% of bad", round(len([lb for lb in labels if lb == 2]) / nr_items * 100, 2))
    #print("% of no revision", round(len([lb for lb in labels if lb == 0]) / nr_items * 100, 2))

def compute_ttest(vals_per_feat, labels):
    """ Calculate the T-test for the means of two independent samples of scores.
    (null hypothesis = independent samples have identical average)
    Tests whether the mean is the same for an interval dependent variable 
    for two independent groups ('revised' vs. 'same' here).
    """
    print("T-test")
    print("{:<20}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format("Feat name", "Mean-G", "Mean-B", "Diff", "Stat", "p"))
    for feat_name, vals in vals_per_feat.items():
        good = [val for (lbl, val) in zip(labels, vals_per_feat[feat_name]) if lbl == 1]
        bad = [val for (lbl, val) in zip(labels, vals_per_feat[feat_name]) if lbl == -1]
        stat, pval = stats.ttest_ind(good,bad, equal_var = False)
        mean_good = round(np.mean(good), 3)
        mean_bad = round(np.mean(bad), 3)
        mean_diff = round(mean_good-mean_bad, 3)
        if pval < 0.05:
            print("{:<20}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8} **".format(feat_name, mean_good, mean_bad, mean_diff,
                                                              round(stat, 3), round(pval, 3)))
        elif pval < 0.10:
            print("{:<20}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8} *".format(feat_name, mean_good, mean_bad, mean_diff,
                                                              round(stat, 3), round(pval, 3)))
        else:
            print("{:<20}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}".format(feat_name, mean_good, mean_bad, mean_diff, 
                                                              round(stat, 3), round(pval, 3)))
