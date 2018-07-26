import numpy as np
import gensim, logging, os, sys, re, datetime
from gensim import corpora, models, similarities

dirlabels = ('film','book', 'music')#, 'film')
oldlabel = 'http://dbpedia.org/resource/'
datalabel = 'dbr:'

modeldir = '/models/'
datastring = '{:%Y%m%d-%H%M}'.format(datetime.datetime.now())
results = modeldir + '/results/'
resultsFileAll = open(results + 'all_' + datastring + '_' + 'combined.txt', 'w')
fnames = os.listdir(modeldir)
fnames = np.sort([f for f in fnames if '.' not in f])

for dirlabel in dirlabels:
    f_entities  = './DBpedia_' + dirlabel + 'entities.txt'
    movielists = './' + dirlabel + '/' + dirlabel + '_lists/'
    with open(f_entities) as f:
        entitiesList = f.readlines()
    entitiesList = np.unique([x.strip().strip('/').replace(oldlabel, datalabel) for x in entitiesList])
    searchKeys = os.listdir(movielists)
    for fname in fnames:
        model = models.Word2Vec.load(modeldir + fname)
        depth = fname[fname.index('_d_')+3:fname.index('_scheme')]
        for searchKey2 in searchKeys:
            expname = searchKey2.replace('.txt','')
            with open(movielists + searchKey2) as f:
                content = f.readlines()
            content = [c.replace(oldlabel, datalabel) for c in content]
            content = np.unique([c.strip() for c in content if datalabel in c])
            print(expname)
            movk = 0
            avgPre = 0.0
            avgRec = 0.0
            for searchKey in content:
                print(searchKey)
                try:
                    relevantRecs = [c for c in content if searchKey != c]
                    kvals = np.arange(1,len(relevantRecs)+1)
                    matches = model.wv.most_similar(positive=[searchKey], topn=20000)
                    matchinga = [s[0] for s in matches if s[0] in entitiesList]
                except KeyError:
                    for k in np.arange(0,len(relevantRecs)):
                        resultsFileAll.write(fname + '\t' + dirlabel + '\t' + expname + '\t' + searchKey + '\t' + str(k+1) + '\t' + str(0.0) + '\t' + str(0.0) + '\t' + str(np.max(kvals)) + '\t' + depth + '\tKeyError' + '\n')
                    continue
                for k in kvals:
                    matching  = matchinga[0:k]
                    print(k)
                    recall = 0.0
                    precision = 0.0
                    if len(relevantRecs) > 0:
                        recall = 100.0 * len(set(relevantRecs).intersection(set(matching)))/len(relevantRecs)
                    if len(matching) > 0:
                        precision = 100.0 * len(set(relevantRecs).intersection(set(matching)))/len(matching)
                    resultsFileAll.write(fname + '\t' + dirlabel + '\t' + expname + '\t' + searchKey + '\t' + str(k) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(np.max(kvals)) + '\t' + depth + '\tNormal' + '\n')
    del model
resultsFileAll.close()
