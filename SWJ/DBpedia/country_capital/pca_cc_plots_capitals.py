import numpy as np
import gensim, logging, os, sys, re
from gensim import corpora, models, similarities
from sklearn.decomposition import PCA
import datetime
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from adjustText import adjust_text
import matplotlib.pyplot as plt

colors = ['b', 'r', 'k', 'w', 'g', 'y','c','m','k']
markers = ['o', '^', 'v', 's','+','x','D']

dirname = "./"
entitiesFile = pd.read_csv(dirname + 'capitals_short.txt',sep='\t', nrows=14)#, skiprows = 1)

entities = [s.replace("http://dbpedia.org/resource/","dbr:") for s in entitiesFile['entity'].tolist() if len(s) > 0]
labels1 = [s.lower() for s in entitiesFile['label'].tolist() if len(s) > 0]
labels = [s.replace("http://dbpedia.org/resource/","") for s in entitiesFile['entity'].tolist() if len(s) > 0]

ulabels = list(np.unique(labels1))
c = [colors[ulabels.index(label)] for label in labels1]
m = [markers[ulabels.index(label)] for label in labels1]

plotWidth = 8
plotHeight = 8

casesno = (6,7,8,10) #6: uniform, 7: pagerank, 8: freq, 10: spec
cases = list(['case_' + str(t) + '_d_2' for t in casesno])

datastr = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

for caseidx, case in enumerate(cases):
    fig, ax = plt.subplots(figsize=(plotWidth,plotHeight))
    model = models.Word2Vec.load(dirname + 'd1/' + case + '_scheme_d1')
    X = model[entities]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    x = result[:,0]
    y = result[:,1]
    #for _m, _c, _x, _y, _label in zip(marker, c, x, y, labels1):
    firstcity = True
    firstcountry = True
    #firstcity = False
    #firstcountry = False
    for _c, _m, _x, _y, _label in zip(c, m, x, y, labels1):
        if _label == 'city' and firstcity:
            plt.scatter(_x, _y, marker=_m, c=_c, label = _label, s=70)
            firstcity = False
        elif _label == 'country' and firstcountry:
            plt.scatter(_x, _y, marker=_m, c=_c, label = _label, s=70)
            firstcountry = False
        else:
            plt.scatter(_x, _y, c=_c, marker=_m, label = None, s=70)
        #print (_c, _label)
    words = list(labels)
    plt.ylim(-4,6)
    plt.xlim(-4,5)
    texts = [plt.text(x[i], y[i], labels[i], color = c[i], fontsize=12) for i in range(len(x))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('./' + datastr + '_plot_' + case + '_cities.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()
    del model

