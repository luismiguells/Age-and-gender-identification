# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:33:28 2018

@author: luismiguells
"""

from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np


def read_labels(file, labels_names):
    """
    file: Archive with the labels.
    labels_names: A list with the labels.
    Returns: A list with the index of the labels. 
    """
    label = []
    with open(file) as content_file:
        for line in content_file:
            category = line.strip()
            label.append(labels_names.index(category))
    return label


def read_file(filename, conv):
    """
    filename: File with the index of the users.
    conv: Cloud be int or str depends on the file.
    Returns: A list with index of the users.
    """
    list_file = []
    if(conv == 'str'):
        with open(filename, encoding="utf-8") as file:
            for line in file:
                list_file.append(line.rstrip())
    else:
        with open(filename, encoding="utf-8") as file:
            for line in file:
                list_file.append(int(line.rstrip()))

    return list_file
    
def most_common(L):
    """
    L: A list.
    Returns: The value most common in a list.
    """
    return max(set(L), key=L.count)

#File directories
lang = 'eng'
year = 2015
main_dir = '/home/luismiguells/Dropbox/TesisG/Data/'
prob = 'age'
user_file = main_dir+"usernames_index_list.txt"
labels_file = main_dir + "users_age_list.txt"
all_user = main_dir + "all_pins_users.txt"
all_labels = main_dir + "all_pins_age.txt"

if prob == 'gender':
    labels_names = ['M', 'F']
else:
    if year == 2015:
        labels_names = ['18-24', '25-34', '35-49', '50+']
    else:
        labels_names = ['18-24', '25-34', '35-49', '50-64', '65-xx']
        
        
labels_list = read_labels(labels_file, labels_names)
labels = np.asarray(labels_list)
corpus = []
corpus = read_file(user_file, 'int')

#Create the CFV with 10 folds
skf = StratifiedKFold(n_splits=10, random_state=0)


#Extract the indexes of the user
all_users = read_file(all_user, 'int')
all_labels = read_labels(all_labels, labels_names)
all_labels = np.asarray(all_labels)


#Create a dictionary key: user value: a list with the pins of the user
d_pins = {}

i = 0
for idx_user in all_users:
    if idx_user not in d_pins:
        d_pins[idx_user] = []
    d_pins[idx_user].append(i)
    i += 1


all_features_dir = "/mnt/disk1/data/2018_luis_miguel_pinterest/resnet50_transformation.txt"
all_features_load = np.loadtxt(all_features_dir)

scores_accuracy = []
scores_precission_macro = []
scores_recall_macro = []
scores_f1_macro = []
scores_kapha = [] 

#Normalize the data
all_features = normalize(all_features_load, norm='l2')
all_features = all_features.tolist()

#Begin classification
y = 1
for train_index, test_index in skf.split(corpus, labels):
    data_train_index = []
    labels_train_index = []
    data_test_index = []
    labels_test_index = []
    
    data_train = []
    data_test = []
    labels_train = []
    labels_test = []
    labels_test_data = []
    labels_train_d = []
    
    len_index = []
    predicted = []
    
    print('Fold :', y)
    fold_train = [corpus[x] for x in train_index]
    fold_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    
    cs = [0.01, 0.1, 1, 10, 100]
    best_c = 0
    best_score = 0

    #Sub-classification to obtain the best C or K
    skf_inner = StratifiedKFold(n_splits=3, random_state=0)
    for c in cs:
        print(c)
        sub_scores = []
        
        
        clf_inner = svm.LinearSVC(C=c)
        
        
        for sub_train_index, sub_test_index in skf_inner.split(fold_train, labels_train):
                
            
            sub_data_train_index = []
            sub_labels_train_index = []
            sub_data_test_index = []
            sub_labels_test_index = []
        
            sub_data_train = []
            sub_data_test = []
            sub_labels_train = []
            sub_labels_test = []
            sub_labels_train_d = []
        
            len_sub_index = []
            sub_predicted = []

            sub_fold_train = [fold_train[x] for x in sub_train_index]
            sub_fold_test = [fold_train[x] for x in sub_test_index]
            sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
        
            
            for idx_user in sub_fold_train:
                sub_data_train_index.extend(d_pins[idx_user])
                sub_labels_train_index.extend(d_pins[idx_user])
        
            for idx_user in sub_fold_test:
                sub_data_test_index.extend(d_pins[idx_user])
                sub_labels_test_index.extend(d_pins[idx_user])
                len_sub_index.append(len(d_pins[idx_user]))
            
            sub_data_train = [all_features[idx] for idx in sub_data_train_index]
            sub_labels_train_d = [all_labels[idx] for idx in sub_labels_train_index]
            sub_data_test = [all_features[idx] for idx in sub_data_test_index]
    
            sub_data_train = np.asarray(sub_data_train)
            sub_labels_train_d = np.asarray(sub_labels_train_d)

            
            clf_inner.fit(sub_data_train, sub_labels_train_d)
            
            sub_predicted_data = clf_inner.predict(sub_data_test)
            
            sub_count = 0
            sub_predicted_data = sub_predicted_data.tolist()

            for sub_e in len_sub_index:
                sub_predicted.append(most_common(sub_predicted_data[sub_count:sub_count+sub_e]))
                sub_count = sub_count + sub_e
            
            sub_predicted = np.asarray(sub_predicted)
            
            
            sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
            sub_scores.append(sub_f1_macro)
            
            
        sub_score = np.mean(sub_scores)
        
        if sub_score > best_score:
            best_score = sub_score
            best_c = c
        
            
    
    for idx_user in fold_train:
        data_train_index.extend(d_pins[idx_user])
        labels_train_index.extend(d_pins[idx_user])
    
    for idx_user in fold_test:
        data_test_index.extend(d_pins[idx_user])
        labels_test_index.extend(d_pins[idx_user])
        len_index.append(len(d_pins[idx_user]))
    
    data_train = [all_features[idx] for idx in data_train_index]
    labels_train_d = [all_labels[idx] for idx in labels_train_index]

    data_test = [all_features[idx] for idx in data_test_index]

    #Classification with the best value for the hyper-parameter 
    clf = svm.LinearSVC(C=best_c)
    
    data_train = np.asarray(data_train)
    labels_train_d = np.asarray(labels_train_d)
    
    clf.fit(data_train, labels_train_d)
    
    predicted_data = clf.predict(data_test)
    
    predicted_data = predicted_data.tolist()
    count = 0
    
    for e in len_index:
        predicted.append(most_common(predicted_data[count:count+e]))
        count = count + e
    
    predicted = np.asarray(predicted)
    
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted)
    
    scores_accuracy.append(accuracy)
    scores_precission_macro.append(precission_macro)
    scores_recall_macro.append(recall_macro)
    scores_f1_macro.append(f1_macro)
    scores_kapha.append(kapha)
     
    y += 1

    
#Print the results
print(prob[0].upper()+prob[1:]+' Accuracy: %0.2f (+/- %0.2f)' % (np.mean(scores_accuracy), np.std(scores_accuracy) * 2))
print(prob[0].upper()+prob[1:]+' Precssion: %0.2f (+/- %0.2f)' % (np.mean(scores_precission_macro), np.std(scores_precission_macro) * 2))
print(prob[0].upper()+prob[1:]+' Recall: %0.2f (+/- %0.2f)' % (np.mean(scores_recall_macro), np.std(scores_recall_macro) * 2))
print(prob[0].upper()+prob[1:]+' F1: %0.2f (+/- %0.2f)' % (np.mean(scores_f1_macro), np.std(scores_f1_macro) * 2))
print(prob[0].upper()+prob[1:]+' Kapha: %0.2f (+/- %0.2f)' % (np.mean(scores_kapha), np.std(scores_kapha) * 2))
    

    
