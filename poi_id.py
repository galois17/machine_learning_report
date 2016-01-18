#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append( "../outliers/" )

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot
from outlier_cleaner import outlierCleaner
import random
import numpy as np
from sklearn import preprocessing

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'ratio_email_sent', 'ratio_email_received'] # You will need to use more features

print("*** Selected features are:")
print(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

print("*** number of data points: %d" % len(data_dict))
# salary is independent variable and bonus is dependent variable

# Find keys to data points that should be removed
keys_to_data_for_removal = []

# Add new feature which is ratio of emails sent to POI over all emails sent and
# ratio of emails from POI over all emails received
ratio_email_sent_list = []
ratio_email_received_list = []
for k,v in data_dict.items():
  if k == "TOTAL":
    keys_to_data_for_removal.append(k)
  if v["to_messages"] != "NaN" and int(v["to_messages"]) > 0:
    v["ratio_email_sent"] = float(v["from_this_person_to_poi"])/float(v["to_messages"])
    ratio_email_sent_list.append(v["ratio_email_sent"])
  else:
    v["ratio_email_sent"] = 0
    ratio_email_sent_list.append(v["ratio_email_sent"])
  if v["from_messages"] != "NaN" and int(v["from_messages"]) > 0:
    v["ratio_email_received"] = float(v["from_poi_to_this_person"])/float(v["from_messages"])
    ratio_email_received_list.append(v["ratio_email_received"])
  else:
    v["ratio_email_received"] = 0
    ratio_email_received_list.append(v["ratio_email_received"])

unscaled_data_dict = data_dict.copy()

# Remove dirty data
for k in keys_to_data_for_removal:
  print("@@ deleting %s " % k)
  del data_dict[k]

# feature scaling
salary_list = []
bonus_list = []
total_stock_value_list = []

for k,v in data_dict.items():
  if v["salary"] != "NaN":
    salary_list.append(float(v["salary"]))
  if v["bonus"] != "NaN":
    bonus_list.append(float(v["bonus"]))
  if v["total_stock_value"] != "NaN":
    total_stock_value_list.append(float(v["total_stock_value"]))

# Build min max scalers for each feature and fit them
salary_min_max_scaler = preprocessing.MinMaxScaler()
salary_min_max_scaler.fit(salary_list)

bonus_min_max_scaler = preprocessing.MinMaxScaler()
bonus_min_max_scaler.fit(bonus_list)

total_stock_value_min_max_scaler = preprocessing.MinMaxScaler()
total_stock_value_min_max_scaler.fit(total_stock_value_list)

ratio_email_sent_min_max_scaler = preprocessing.MinMaxScaler()
ratio_email_sent_min_max_scaler.fit(ratio_email_sent_list)

ratio_email_received_min_max_scaler = preprocessing.MinMaxScaler()
ratio_email_received_min_max_scaler.fit(ratio_email_received_list)

# Step through each item in the data dictionary and transform (normalize) it
for k,v in data_dict.items():
  if v["salary"] != "NaN":
    elem = salary_min_max_scaler.transform([float(v["salary"])])
    v["salary"] = elem[0]
  if v["bonus"] != "NaN":
    elem = bonus_min_max_scaler.transform([float(v["bonus"])])
    v["bonus"] = elem[0]
  if v["total_stock_value"] != "NaN":
    elem = total_stock_value_min_max_scaler.transform([float(v["total_stock_value"])])
    v["total_stock_value"] = elem[0]
  if v["to_messages"] != "NaN" and int(v["to_messages"]) > 0:
    elem = ratio_email_sent_min_max_scaler.transform([v["ratio_email_sent"]])
    v["ratio_email_sent"] = elem[0]
  if v["from_messages"] != "NaN" and int(v["from_messages"]) > 0:
    elem = ratio_email_received_min_max_scaler.transform([v["ratio_email_received"]])
    v["ratio_email_received"] = elem[0]

data_cleanup = featureFormat(data_dict, features_list, sort_keys = True)

my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn import svm
# clf = svm.SVC(kernel="rbf")

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=3)

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=100)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)

# from sklearn.svm import SVC
# from sklearn.decomposition import RandomizedPCA
# from sklearn.grid_search import GridSearchCV

# from sklearn.cross_validation import StratifiedShuffleSplit

# parameters = {'kernel': ('linear', 'rbf'), 'C': [10, 5e4, 1e5], 'gamma': [0.005, 0.01, 0.1]}
# svr = svm.SVC()
# clf = GridSearchCV(svr, parameters)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

features_train_scaled_pca = features_train
features_test_scaled_pca = features_test

# Fit it
clf.fit(features_train_scaled_pca, labels_train)
Z = clf.predict(features_test_scaled_pca)

print("@@ labels")
print(np.array(labels_test))
print("@@ predicted")
print(np.array(Z))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)