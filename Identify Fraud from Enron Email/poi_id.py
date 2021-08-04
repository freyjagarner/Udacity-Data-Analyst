#!/usr/bin/python3

#import statements

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tester import *
from feature_format import *
from sklearn.pipeline import Pipeline


from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

### Load the dictionary containing the dataset
with open(r"C:\Users\chels\udacity-git-course\new-git-project\ud120-projects\final_project\final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.

# financial features lists
salary_list = []
bonus_list = []
loan_advances_list = []
deferral_payments_list= []
deferred_income_list = []
expenses_list = []
director_fees_list = []
other_list = []
total_payments_list = []
long_term_incentive_list = []

restricted_stock_deferred_list = []
exercised_stock_options_list = []
restricted_stock_list = []
total_stock_value_list = []

#email features list
email_list = []
to_messages_list = []
from_messages_list = []
from_poi_to_this_person_list = []
from_this_person_to_poi_list = []
shared_receipt_with_poi_list = []

# poi_value
poi_value_list = []

# list of above lists
list_of_lists = [salary_list,
bonus_list,
loan_advances_list,
deferral_payments_list,
deferred_income_list,
expenses_list,
director_fees_list,
other_list,
total_payments_list,
long_term_incentive_list,
restricted_stock_deferred_list,
exercised_stock_options_list,
restricted_stock_list,
total_stock_value_list,
email_list,
to_messages_list,
from_messages_list,
from_poi_to_this_person_list,
from_this_person_to_poi_list,
shared_receipt_with_poi_list,
poi_value_list]

# names for the lists
list_names = ["salary", "email_address"
"bonus",
"loan_advances",
"deferral_payments",
"deferred_income",
"expenses",
"director_fees",
"other",
"total_payments",
"long_term_incentive",
"restricted_stock_deferred",
"exercised_stock_options",
"restricted_stock",
"total_stock_value",
"email_address",
"to_messages",
"from_messages",
"from_poi_to_this_person",
"from_this_person_to_poi",
"shared_receipt_with_poi",
"poi"]


# finds if each feature is NaN or not and adds non-nan items to appropriate list
for person, attribute in data_dict.items():
        for attribute, a in attribute.items():
            if (attribute == "bonus") and (a != "NaN"):
                bonus_list.append(a)
            elif (attribute == "salary") and (a != "NaN"):
                salary_list.append(a)
            elif (attribute == "loan_advances") and (a != "NaN"):
                loan_advances_list.append(a)
            elif (attribute == "deferral_payments") and (a != "NaN"):
                deferral_payments_list.append(a)
            elif (attribute == "deferred_income") and (a != "NaN"):
                deferred_income_list.append(a)
            elif (attribute == "expenses") and (a != "NaN"):
                expenses_list.append(a)
            elif (attribute == "director_fees") and (a != "NaN"):
                director_fees_list.append(a)
            elif (attribute == "other") and (a != "NaN"):
                other_list.append(a)
            elif (attribute == "long_term_incentive") and (a != "NaN"):
                long_term_incentive_list.append(a)
            elif (attribute == "total_payments") and (a != "NaN"):
                total_payments_list.append(a)
            elif (attribute == "restricted_stock_deferred") and (a != "NaN"):
                restricted_stock_deferred_list.append(a)
            elif (attribute == "exercised_stock_options") and (a != "NaN"):
                exercised_stock_options_list.append(a)
            elif (attribute == "restricted_stock") and (a != "NaN"):
                restricted_stock_list.append(a)
            elif (attribute == "total_stock_value") and (a != "NaN"):
                total_stock_value_list.append(a)
            elif (attribute == "email_address") and (a != "NaN"):
                email_list.append(a)
            elif (attribute == "to_messages") and (a != "NaN"):
                to_messages_list.append(a)
            elif (attribute == "from_messages") and (a != "NaN"):
                from_messages_list.append(a)
            elif (attribute == "from_poi_to_this_person") and (a != "NaN"):
                from_poi_to_this_person_list.append(a)
            elif (attribute == "from_this_person_to_poi") and (a != "NaN"):
                from_this_person_to_poi_list.append(a)
            elif (attribute == "shared_receipt_with_poi") and (a != "NaN"):
                shared_receipt_with_poi_list.append(a)
            elif (attribute == "poi") and (a != "NaN"):
                poi_value_list.append(a)

# prints the appropriate list name and how many people we have data for in that list followed by a % overall
count = 0
for i in list_of_lists:
    print(f"{list_names[count]} feature: {len(i)} people or {round((len(i)/146)*100, 2)}% of people")
    count += 1

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi",
"salary",
"bonus",
"loan_advances",
"deferral_payments",
"deferred_income",
"expenses",
"director_fees",
"other",
"total_payments",
"long_term_incentive",
"restricted_stock_deferred",
"exercised_stock_options",
"restricted_stock",
"total_stock_value",
"to_messages",
"from_messages",
"from_poi_to_this_person",
"from_this_person_to_poi",
"shared_receipt_with_poi"]

# converts numeric values to floats and replaces NaN with 0 for all instances
for person in data_dict:
    for feature in data_dict.get(person):
        if (feature != 'email_address') and (feature != 'poi'):
            float(data_dict.get(person)[feature])
        if data_dict.get(person)[feature] == 'NaN':
            data_dict.get(person)[feature] = 0


### Task 2: Remove outliers

# alteration of draw function from k-means mini project to visualize data
def Draw(data, feature_1, feature_2, name="image.png"):
    data = featureFormat(data, [feature_1, feature_2])
    for item in data:
        x = item[0]
        y = item[1]
        plt.scatter(x, y)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.savefig(name)
    plt.show()


# makes a scatter plot of salary & exercised stock options
Draw(data_dict, feature_1="salary", feature_2="exercised_stock_options", name="financialoutlier1.png")

# removing our outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

# redraws scatterplot with outlier removed
Draw(data_dict, feature_1="salary", feature_2="exercised_stock_options", name="financialoutlier2.png")

### Task 3: Create new feature(s)

# adds feature emails_to_poi_vs_total which gives a value to 2 decimal places of fraction of emails from this person to poi vs all from_messages
data_dict.get(person)["emails_to_poi_vs_total"] = 0
for person in data_dict:
    for a in person:
        data_dict.get(person)["emails_to_poi_vs_total"] = 0
        if data_dict.get(person)["from_messages"] != 0 and (data_dict.get(person)["from_this_person_to_poi"] != 0):
            data_dict.get(person)["emails_to_poi_vs_total"] = round(
                data_dict.get(person)['from_this_person_to_poi'] / data_dict.get(person)['from_messages'], 2)

# adds feature emails_from_poi_vs_total which gives a value to 2 decimal places of fraction of emails to this person from poi vs all to_messages
for person in data_dict:
    data_dict.get(person)["emails_from_poi_vs_total"] = 0
    if (data_dict.get(person)["to_messages"] != 0) and (data_dict.get(person)["from_poi_to_this_person"] != 0):
        data_dict.get(person)["emails_from_poi_vs_total"] = round(
            data_dict.get(person)["from_poi_to_this_person"] / data_dict.get(person)["to_messages"], 2)

# check if our new features are working
print(data_dict["SKILLING JEFFREY K"]["emails_from_poi_vs_total"])
print(data_dict["SKILLING JEFFREY K"]["emails_to_poi_vs_total"])

#makes a list with new features then add that to full features list
new_features = ["emails_to_poi_vs_total", "emails_from_poi_vs_total"]
features_list += new_features

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#uses pipeline and gridsearchcv to decide on k best features
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('selector', SelectKBest(f_classif)),
                 ('classifier', GaussianNB())])

# makes a list k_values which stores the result of GridSearchCV on kbest with as many inputs as features
k_values = []
for i in range(1, 24):
    params = [{'selector__k': [i]}]
    clf = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        n_jobs=-1,
        cv=10,
        verbose=0)
    clf = clf.fit(features, labels)
    k_values.append((i, clf.best_score_))
    print(f"For k = {i} the score is: {clf.best_score_}")

#sorts the given k_values by their scores and prints the highest score along with it's k value
k_values_sorted = sorted(k_values, key=lambda a: a[1], reverse=True)
print(f"The k value for k best with the highest score is {k_values_sorted[0][0]} with a score of {k_values_sorted[0][1]}")

# find k best features to use, 10 best features
kbest = SelectKBest(f_classif, k=4)
kbest.fit_transform(features, labels)

#make a list of tuples with the name and score of kbest features and sorts them highest to lowest
kb_name_score = zip(features_list, kbest.scores_)
kbest_list = sorted(kb_name_score, key = lambda a: a[1], reverse = True)

#prints all of the features ranked by kbest
count = 0
for i in kbest_list:
    print(f"{count}. Feature: {kbest_list[count][0]} \n Score: {kbest_list[count][1]}")
    count+=1

#prints 4 kbest features with score and adds their names to a list kbest_features
print("List of 4 kbest features")
count = 1
kbest_features = []
while count < 5:
    print(f"{count}. Feature: {kbest_list[count][0]} \n Score: {kbest_list[count][1]}")
    kbest_features.append(kbest_list[count][0])
    count+=1

# adds kbest_features to the data as our feature list
data = featureFormat(my_dataset, kbest_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scale our features using minmaxscaler to make financial data more evenly weighted
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

#Divides our data into training and testing portions to test our models
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Naive Bayes Classifier Test
# clf = GaussianNB()
# clf.fit(features_train, labels_train)

#Results
#GaussianNB()
# Accuracy: 0.73900    Precision: 0.22604    Recall: 0.39500    F1: 0.28753    F2: 0.34363
# Total predictions: 15000    True positives:  790    False positives: 2705    False negatives: 1210    True negatives: 10295

# Decision Tree Classifier Test
clf = DecisionTreeClassifier(criterion= 'gini',
 max_depth= None,
 max_features= 3,
 min_samples_leaf= 1,
 min_samples_split= 13,
 random_state = 8,
 splitter= 'best')
clf.fit(features_train, labels_train)

#Before Tuning
#DecisionTreeClassifier()
#Accuracy: 0.79633    Precision: 0.22879    Recall: 0.22250    F1: 0.22560    F2: 0.22373
#Total predictions: 15000    True positives:  445    False positives: 1500    False negatives: 1555    True negatives: 11500

#After Tuning
# DecisionTreeClassifier(max_features=9, min_samples_split=13, random_state=8)
# 	Accuracy: 0.83653	Precision: 0.37416	Recall: 0.33600	F1: 0.35406	F2: 0.34300
# 	Total predictions: 15000	True positives:  672	False positives: 1124	False negatives: 1328	True negatives: 11876

# SVM SVC Classifier Test
# clf = SVC(gamma= 'scale', kernel= 'poly', shrinking = 1)
# clf.fit(features_train, labels_train)

#Before Tuning
#SVC()
#Accuracy: 0.86267    Precision: 0.30000    Recall: 0.02250    F1: 0.04186    F2: 0.02761
#Total predictions: 15000    True positives:   45    False positives:  105    False negatives: 1955    True negatives: 12895

# After Tuning
# Accuracy: 0.86820	Precision: 0.55502	Recall: 0.05800	F1: 0.10502	F2: 0.07065
# Total predictions: 15000	True positives:  116	False positives:   93	False negatives: 1884	True negatives: 12907

# KNearestNeighbors classifier test
#clf = KNeighborsClassifier(
# leaf_size= 1,
# metric= 'minkowski',
# n_neighbors= 3,
# p= 1,
# weights= 'distance')
#clf.fit(features_train, labels_train)

#Before Tuning
# KNeighborsClassifier()
# Accuracy: 0.87920    Precision: 0.65461    Recall: 0.19900    F1: 0.30521    F2: 0.23118
# Total predictions: 15000    True positives:  398    False positives:  210    False negatives: 1602    True negatives: 12790

#Random Forest Classifier Test
# clf = RandomForestClassifier(criterion= 'gini',
# max_depth= 2,
# max_features= 'log2',
# n_estimators= 50)
#clf.fit(features_train, labels_train)

#Before Tuning
#RandomForestClassifier()
# Accuracy: 0.86027    Precision: 0.42027    Recall: 0.12650    F1: 0.19447    F2: 0.14706
# Total predictions: 15000    True positives:  253    False positives:  349    False negatives: 1747    True negatives: 12651

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#KNearest Neighbors GridSearchCV
#parameters = {
#    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#    'n_neighbors': range(1,8),
#    'leaf_size': range(1,50),
#    'p': [1,2],
#    'weights': ['uniform', 'distance'],
#   'metric': ['minkowski', 'chebyshev', 'euclidean', 'manhattan']
#}
#KNN_clf = GridSearchCV(clf, parameters, n_jobs = -1)
#KNN_clf.fit(features_train, labels_train)
#KNN_clf.best_params_

#Decision Tree GridSearchCV
# parameters = {'criterion': ["gini", "entropy"],
#              'splitter': ['best', 'random'],
#              "max_features": range(1,10),
#              "min_samples_split": range(1, 15),
#              "min_samples_leaf": range(1, 15),
#             "random_state": range(1,150)
#             }

#DT_clf = GridSearchCV(clf, parameters, cv = 2, n_jobs = -1)
#DT_clf.fit(features_train, labels_train)
#DT_clf.best_params_

# SCV GridSearchCV
# parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#              'gamma': ['scale','auto'],
#              'shrinking': [1, 0]
#              }

# SVC_clf = GridSearchCV(estimator = clf, param_grid=parameters, cv = 2, n_jobs = -1)
# SVC_clf.fit(features_train, labels_train)
# SVC_clf.best_params_

# Random Forest GridSearchCV
# parameters = {'criterion': ["gini", "entropy"],
#              'n_estimators': [50, 60, 70, 80,90, 100],
#              'max_depth': range(1, 11),
#              "max_features": ['auto', 'sqrt', 'log2']
#              }


# RF_clf = GridSearchCV(estimator = clf, param_grid=parameters, cv = 2, n_jobs = -1)
# RF_clf.fit(features_train, labels_train)
# RF_clf.best_params_

# RandomForestClassifier(max_depth=2, max_features='log2', n_estimators=50)
#	Accuracy: 0.86760	Precision: 0.52381	Recall: 0.07700	F1: 0.13426	F2: 0.09284
#	Total predictions: 15000	True positives:  154	False positives:  140	False negatives: 1846	True negatives: 12860

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)