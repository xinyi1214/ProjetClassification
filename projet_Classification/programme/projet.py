#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jav 14

@author: Xinyi Shen
"""
'''
Conseigne:
a)qui soit capalbe de faire la classification pour les données avec deux features différents et quatre algorithmes différents
b)une fois cela réalisé, vous aves les résultats (precision, recall, fi-score et support) pour cette classification

Il est donc attendu:
1) un fichier csv/tsv qui a deux colonnes: la première colonne est les textes, la deuxième colonne est les labels
2) pour avoir différents résultats par rapport deux features et quatre algorithlmes, il faut changer la fonction à la fin de la programmation
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import spacy
from spacy import displacy
nlp = spacy.load('fr_core_news_md')

'''
Prétaitrement, les deux premières functions sont pour tf-idf, les deux dernières sont pour word embedding
'''
def tokenize(text): #tokeniser tous les texte
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(text)

def lemmatize(text): #faire tous les token en infinitif
	lemmatizer = WordNetLemmatizer()
	return ' '.join([lemmatizer.lemmatize(word) for word in text])

def spacy_average_vectors(phrase, nlp):#embedding
    dec = nlp(phrase)
    return sum(w.vector for w in dec)/len(dec)

def spacy_word2vec_features(X, nlp):
    feats = np.vstack([spacy_average_vectors(p, nlp) for p in X])
    return feats

'''
appliquer le prétraitement et ajouter l'alogorithme
'''
def simple_tfidf_naiveBayes(file):
	Corpus = pd.read_csv(file, sep='\t')	

	Corpus['text'] = Corpus['text'].map(tokenize)
	Corpus['text'] = Corpus['text'].map(lemmatize)
	labels = Corpus['label']

	count_vect = CountVectorizer() #Convertir une collection de documents texte en une matrice de nombres de tokens
	text_counts = count_vect.fit_transform(Corpus['text'])  #appliquer la dernière ligne pour les textes
	tfidf_transformer = TfidfTransformer() #Transforme une matrice de comptage en une représentation tf ou tf-idf normalisée
	text_tfidf = tfidf_transformer.fit_transform(text_counts) #appliquer la dernière ligne pour les textes
	X_train, X_test, y_train, y_test = train_test_split(text_tfidf, labels, test_size=0.25) #split les données en entrainêment et evaluation
	clf = MultinomialNB().fit(X_train, y_train) #ajouter l'algorithme naive bayes

	y_pred = clf.predict(X_test) #évaluation
	print("le rapport de la classification pour tfidf_naiveBayes: ")
	print(classification_report(y_test, y_pred)) #résultat

def simple_tfidf_svm(file):
	Corpus = pd.read_csv(file, sep='\t')

	Corpus['text'] = Corpus['text'].map(tokenize)
	Corpus['text'] = Corpus['text'].map(lemmatize)
	labels = Corpus['label']

	count_vect = CountVectorizer()
	text_counts = count_vect.fit_transform(Corpus['text'])
	tfidf_transformer = TfidfTransformer()
	text_tfidf = tfidf_transformer.fit_transform(text_counts)
	X_train, X_test, y_train, y_test = train_test_split(text_tfidf, labels, test_size=0.25)
	clf = SVC(C=1, kernel="linear").fit(X_train, y_train)#ajouter l'algorithme SVM

	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour tfidf_svm:")
	print(classification_report(y_test, y_pred))

def simple_tf_idf_logisticRegression(file):
	Corpus = pd.read_csv(file, sep='\t')

	Corpus['text'] = Corpus['text'].map(tokenize)
	Corpus['text'] = Corpus['text'].map(lemmatize)
	labels = Corpus['label']

	count_vect = CountVectorizer()
	text_counts = count_vect.fit_transform(Corpus['text'])
	tfidf_transformer = TfidfTransformer()
	text_tfidf = tfidf_transformer.fit_transform(text_counts)
	X_train, X_test, y_train, y_test = train_test_split(text_tfidf, labels, test_size=0.25)
	clf = LogisticRegression(random_state=0).fit(X_train, y_train) #ajouter l'algorithme logistic regression
	y_pred = clf.predict(X_test)

	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour tf_idf_logisticRegression : ")
	print(classification_report(y_test, y_pred))

def simple_tf_idf_randomForest(file):

	Corpus = pd.read_csv(file, sep='\t')

	Corpus['text'] = Corpus['text'].map(tokenize)
	Corpus['text'] = Corpus['text'].map(lemmatize)
	labels = Corpus['label']

	count_vect = CountVectorizer()
	text_counts = count_vect.fit_transform(Corpus['text'])
	tfidf_transformer = TfidfTransformer()
	text_tfidf = tfidf_transformer.fit_transform(text_counts)
	X_train, X_test, y_train, y_test = train_test_split(text_tfidf, labels, test_size=0.25)

	clf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train) #ajouter l'algorithme random forest
	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour tf_idf_randomForest : ")
	print(classification_report(y_test, y_pred))

'''
Naive Bayes peut pas avoir les nombre négatif, du coup pour le prétraitement wordembedding, on supprime cet algorithme
'''
def wordEmbedding_svm(file):
	Corpus = pd.read_csv(file, sep='\t')
	
	Encoder = LabelEncoder()
	labels = Encoder.fit_transform(Corpus['label'])

	X_train, X_test, y_train, y_test = train_test_split(Corpus['text'], labels, test_size=0.25) #il faut split les données avant faire l'embedding
	X_train = spacy_word2vec_features(X_train, nlp) #word embedding pour les données d'entraînement
	X_test = spacy_word2vec_features(X_test, nlp) #word embedding pour les données d'évaluation
	clf = SVC(C=1, kernel="linear").fit(X_train, y_train) #ajouter l'algoritheme svm

	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour wordEmbedding_svm : ")
	print(classification_report(y_test, y_pred))

def wordEmbedding_logisticRegression(file):
	Corpus = pd.read_csv(file, sep='\t')
	
	Encoder = LabelEncoder()
	labels = Encoder.fit_transform(Corpus['label'])

	X_train, X_test, y_train, y_test = train_test_split(Corpus['text'], labels, test_size=0.25)
	X_train = spacy_word2vec_features(X_train, nlp)
	X_test = spacy_word2vec_features(X_test, nlp)
	clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train) #ajouter l'algoritheme logistic regression

	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour wordEmbedding_logisticRegression : ")
	print(classification_report(y_test, y_pred))

def wordEmbedding_randomForest(file):

	Corpus = pd.read_csv(file, sep='\t')
	
	Encoder = LabelEncoder()
	labels = Encoder.fit_transform(Corpus['label'])

	X_train, X_test, y_train, y_test = train_test_split(Corpus['text'], labels, test_size=0.25)
	X_train = spacy_word2vec_features(X_train, nlp)
	X_test = spacy_word2vec_features(X_test, nlp)

	clf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train) #ajouter l'algoritheme random forest
	y_pred = clf.predict(X_test)
	print("le rapport de la classification pour wordEmbedding_randomForest : ")
	print(classification_report(y_test, y_pred))	

if __name__ == '__main__':

	file = "/Users/shenxinyi/Documents/python2/projet/programme/monde.csv"
	wordEmbedding_randomForest(file)

