#!/usr/bin/python
from gensim import corpora, models, similarities
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc,mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import datetime
import gensim, logging, os, sys, re
import numpy as np
import pandas as pd
import string
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


#colors = ['b', 'r', 'g', 'y','c','m','k']
dirname = './'
casesno = (6,7,8,10,12,13)
cases = list(['case_' + str(t) + '_d_1' for t in casesno] + ['case_' + str(t) + '_d_2' for t in casesno])
lbls = ('high','medium','low')

trainscores = np.zeros((len(cases),1))
testscores = np.zeros((len(cases),1))
bestC = np.zeros((len(cases),1))
errors = np.zeros((len(cases),1))
errors2 = np.zeros((len(cases),1))
nummovies = np.zeros((len(cases),1))

argList = list(sys.argv) #give meta_movie_data.csv or meta_album_data.csv
file1 = argList[1]

moviesFile = pd.read_csv(dirname + file1,sep=',', encoding = "ISO-8859-1")
movies = moviesFile['DBpedia_URI'].tolist()
movies = list(np.unique([x.strip().strip('/').replace("http://dbpedia.org/resource/", "dbr:") for x in movies]))
labels = moviesFile['label'].tolist()
ratings = moviesFile['rating'].tolist()

datastring = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

results = './'
resultsFileAll = open(results + 'all_' + datastring + '_' + file1.replace('.csv','_lin.txt'), 'w')

#select movies common to all models
movies_changed = set(movies)
for i,case in enumerate(cases):
    print('Case: {}'.format(case))
    model = models.Word2Vec.load(dirname + 'd1/' + case + '_scheme_d1')
    dictList = model.wv.vocab.items()
    dictDBpedia = [s[0] for s in dictList if 'dbr:' in s[0]]
    moviesX = [s for s in movies if s in dictDBpedia]
    movies_changed = movies_changed.intersection(set(moviesX))
    del model

moviesX = list(movies_changed)

for i,case in enumerate(cases):
    model = models.Word2Vec.load(dirname + 'd1/' + case + '_scheme_d1')
    nummovies[i] = len(moviesX)
    indices = np.where(np.in1d(movies,moviesX))[0]
    Y =  np.take(labels,indices)
    Yratings =  np.take(ratings,indices)
    X = model[moviesX]
    pca = PCA(n_components=50)
    X = pca.fit_transform(X)
    scaler = StandardScaler()  
    scaler.fit(X)  
    X = scaler.transform(X)  
    print('Case: {}'.format(case))
    lr = SVC()
    trainscores[i] = np.mean(cross_val_score(lr, X, Y, cv=10, scoring='accuracy'))
    lr = LinearRegression()
    errors[i] = abs(np.mean(cross_val_score(lr, X, Yratings, cv=10, scoring="neg_mean_squared_error")))**0.5
    del model
    del lr
    #del clf


for _c, _tr, _movieno, _err in zip(cases, trainscores, nummovies, errors):
    resultsFileAll.write(_c  + '\t' + str(_movieno) + '\t' + str(_tr) + '\t' + str(_err) + '\n')

resultsFileAll.close()
