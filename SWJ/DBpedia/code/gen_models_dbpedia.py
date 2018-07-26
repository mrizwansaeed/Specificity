import numpy as np
import gensim, logging, os, sys, re
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',filename='experiment08222017.out', level=logging.INFO)
#suffix = 'd1'

class MySentences(object):
    def __init__(self, basedir, filenames):
        self.filenames = filenames
        self.basedirectory = basedir
    def __iter__(self):
        for filename in self.filenames:
            for line in open(self.basedirectory + filename):
                line = line.rstrip('\n').replace("\""," ")
                words = line.split(' ')
                w = [x.strip() for x in words if len(x) > 2]
                yield w
            #filename.close()

#pathlbls = ['city','country','film','book']
pathlbls = ['input/']
for pathlbl in pathlbls:
    dirname = './' + pathlbl
    allFileList = os.listdir(dirname)

    suffixes = ['d1', 'exactD', 'uptoD']
    suffix = suffixes[0]

    depths = [1,2,3]
    cases = [6,7,8,9,10,11,12,13]


    for i in cases:
        fileList = [s for s in allFileList if ('case_' + str(i) + '_') in s]
        for d in range(0,len(depths)):
            uptoDepths = ['1']
            if depths[d] > 1:
                if suffix == 'd1':
                    uptoDepths.append(str(depths[d]))
                elif suffix == 'exactD':
                    uptoDepths = [str(depths[d])]
                elif suffix == 'uptoD':
                    uptoDepths = map(str,list(np.arange(1,depths[d]+1)))
                #simply get the depth out of each file name and check in uptoDepths array
            uptoDepthFiles = [s for s in fileList if s[s.index('_d')+2:s.index('_bn')] in uptoDepths]
            print(uptoDepthFiles)
            sentences = MySentences(dirname, uptoDepthFiles)
            savedFile = './' + pathlbl + '/' + suffix + '/case_' + str(i) + '_d_' + str(depths[d]) + '_scheme_' + suffix
            print 'Computing Model:' + savedFile
            try:
                #model = gensim.models.Word2Vec(sentences, size=500, workers=24, window=12, sg=1, negative=5, iter=8, min_count=5)#, sample=1e-5)
                model = gensim.models.Word2Vec(sentences, size=500, workers=24, window=10, sg=1, negative=15, iter=5)#, min_count=5)#, sample=1e-5)
                model.save(savedFile)
                del model
            except MemoryError:
                print 'Memory Error'
                continue 
            except RuntimeError:
                print 'Runtime Error for case/depth ' + str(i) + '/' + str(depths[d]) 
                continue
            print 'Saving Model'+ savedFile
            #model.save(savedFile)
            #del model

