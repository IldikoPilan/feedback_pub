import os
from extract_features import extract_features, extract_features_LA
from sklearn.linear_model import LinearRegression
from do_ml import *

# Function calls

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
path = '/Users/ildiko/Documents/work/projects/CityU/venv_fbgen/results/' 

data_file = "/Users/ildikop/Documents/projects/CityU/venv_fbgen/annotation/annotated/all_annot_info2.csv"
features_file_name = path + "revision_success2.data"
label_file_name = path + "revision_success2.target"
feature_names_fname = path + "feature_names.txt"
extract_features_LA(data_file, features_file_name, label_file_name, feature_names_fname, nlp, add_extra_var=True)

X, y, feature_names = load_ml_data(features_file_name, label_file_name, "")
clfs = get_classifiers(svm_cl=True)
eval_cl(X,y,clfs,feature_names,cv_folds=3, balance=True, select_f=True)
get_ml_data_stats(X,y)

vals_per_feat = get_vals_per_feat(X, feature_names)
labels = map_labels(y)
compute_correls(vals_per_feat, labels, feature_names) # do on sent_align only (labels = interval)
compute_ttest(vals_per_feat, labels)
reg = LinearRegression().fit(X, y)
print(reg.score(X, labels))
print(reg.intercept_)
for coef, fn in sorted(zip(reg.coef_, feature_names), reverse=True):
   print("{:<20}\t{:<10}".format(fn, round(coef, 2)))
