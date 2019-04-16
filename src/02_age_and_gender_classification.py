#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:44:07 2018

@author: luismiguells
"""

#Libraries
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
import time


def my_tokenizer(s):
    """
    s: A string. 
    Returns: A list with only the words.
    """
    return s.split()



def read_users(file):
    """
    file: Archive with the usernames.
    Returns: A list with all the users.
    """
    users = []
    with open(file) as content_file:
        for line in content_file:
            users.append(int(line.strip()))
    return users


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


def clean_words(words, stop_words):
    """
    words: A list of words. 
    stop_words: The list of the stopwords.
    Returns: A list of clean words.
    """
    text = ' '.join([word for word in words if len(word)>2 and len(word)<35 and word not in stop_words])
    return text

def read_text_data_with_emos(lang, text_file, emo_file):
    """
    lang: The language (default English).
    text_file: The file with the text content.
    emo_file: The file with emoji/emoticons.
    Returns: An array with words + emojis/emoticons.
    """
    data = []
    if lang == 'spa':
        stop_words = stopwords.words('spanish')
    elif lang == 'eng':
        stop_words = stopwords.words('english')
    with open(text_file, encoding='utf-8') as text_content, open(emo_file, encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            text += ' '+emo_line.rstrip()
            data.append(text)
    return data

def read_emos(emo_file):
    """
    emoji_file: The file with the emojis.
    Returns: An array with the features.
    """
    data = []
    with open(emo_file, encoding='utf-8') as emo_content:
        for emo_line in emo_content:
            emo = emo_line.rstrip()
            data.append(emo)
    return data

def read_text_data(lang, file):
    """
    lang: The language (default English).
    file: The file with the text content.
    Returns: An array with the features.
    """
    data = []
    if lang == 'spa':
        stop_words = stopwords.words('spanish')
    elif lang == 'eng':
        stop_words = stopwords.words('english')
    with open(file, encoding='utf-8') as content_file:
        for line in content_file:
            words = line.rstrip().split()
            text = clean_words(words, stop_words)
            data.append(text)
    return data

    
#File directories
lang = 'eng'
year = 2015
main_dir = '/Users/luismiguells/Dropbox/TesisG/Data/'
prob = 'gender'
labels_file = main_dir+'users_gender_list.txt'
words_file = main_dir+'words_file.txt'
tweets_file = main_dir+'all_dataset_text.txt'
users_file = main_dir+'usernames_index_list.txt'
hashs_file = main_dir+'hash_file.txt'
ats_file = main_dir+'ats_file.txt'
emo_file = main_dir+'emo_file.txt'
links_file = main_dir+'links_file.txt'
if prob == 'gender':
    labels_names = ['M', 'F']
else:
    if year == 2015:
        labels_names = ['18-24', '25-34', '35-49', '50+']
    else:
        labels_names = ['18-24', '25-34', '35-49', '50-64', '65-xx']

labels_list = read_labels(labels_file, labels_names)
corpus = []
corpus = read_text_data(lang, words_file)
#corpus = read_text_data_with_emos(lang, words_file, emo_file) #For words + emojis/emoticons
#corpus = read_emos(emo_file) #For emojis/emoticons
labels = np.asarray(labels_list)
labels_set = set(labels_list)



#Machine learning models
clf_nb = MultinomialNB()
clf_svm = svm.LinearSVC(C=10)
clf_log = LogisticRegression(C=100, penalty='l2', solver='liblinear')
clf_rdf = RandomForestClassifier()
clf_knn = KNeighborsClassifier()


#Lists to storage the metrics and the evaluation technique
skf = StratifiedKFold(n_splits=10, random_state=0) #10 
scores_accuracy = []
scores_precission_macro = []
scores_recall_macro = []
scores_f1_macro = []
scores_kapha = [] 
scores_roc = []

i = 0

start = time.time()

#Begin classification process
for train_index, test_index in skf.split(corpus, labels):
    print('Fold :',i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(data_train)
    
    #Find the best value for the hyper-parameter
    ks = [1, 2, 3, 5, 10] #For KNN
    #cs = [0.01, 0.1, 1, 10, 100] #For LSVC and LR
    #best_c = 0
    best_score = 0
    best_k = 0

    for k in ks:
        #print(c)
        clf_inner = KNeighborsClassifier(n_neighbors=k)
        sub_skf = StratifiedKFold(n_splits=3, random_state=0)
        scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
        score = np.mean(scores_inner)
        if score > best_score:
            best_score = score
            best_k = k
            
    #Classification with the best value of the hyper-parameter
    clf = KNeighborsClassifier(n_neighbors=best_k)
    
    #clf = RandomForestClassifier()
    clf.fit(train_tfidf, labels_train)
    test_tfidf = vec.transform(data_test)
    predicted = clf.predict(test_tfidf)
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted)

    
    #print(metrics.confusion_matrix(labels_test, predicted))
    
    scores_accuracy.append(accuracy)
    scores_precission_macro.append(precission_macro)
    scores_recall_macro.append(recall_macro)
    scores_f1_macro.append(f1_macro)
    scores_kapha.append(kapha)
    i += 1

end = time.time()

#Print the results
print(prob[0].upper()+prob[1:]+' Accuracy: %0.2f (+/- %0.2f)' % (np.mean(scores_accuracy), np.std(scores_accuracy) * 2))
print(prob[0].upper()+prob[1:]+' Precssion: %0.2f (+/- %0.2f)' % (np.mean(scores_precission_macro), np.std(scores_precission_macro) * 2))
print(prob[0].upper()+prob[1:]+' Recall: %0.2f (+/- %0.2f)' % (np.mean(scores_recall_macro), np.std(scores_recall_macro) * 2))
print(prob[0].upper()+prob[1:]+' F1: %0.2f (+/- %0.2f)' % (np.mean(scores_f1_macro), np.std(scores_f1_macro) * 2))
print(prob[0].upper()+prob[1:]+' Kapha: %0.2f (+/- %0.2f)' % (np.mean(scores_kapha), np.std(scores_kapha) * 2))
print("Time of training + testing: %0.2f " % (end - start))

