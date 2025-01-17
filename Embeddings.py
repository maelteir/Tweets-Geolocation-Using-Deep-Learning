from gensim.models import KeyedVectors
from gensim.models import FastText, Word2Vec
from tqdm import tqdm
import numpy as np
import os

class Embeddings:
    def __init__(self,_vocabulary):
        self.vocabulary = _vocabulary
        
    def load_fasttext(self, Arabic = 1):
        print('loading word embeddings...')
        self.embeddings_index = {}
        if (Arabic == 1):
	        f = open('~/Embeddings/cc.ar.300.vec',encoding='utf-8') #Adjust the location
        else:
	        f = open('~/Embeddings/cc.en.300.vec',encoding='utf-8') #Adjust the location
        for line in tqdm(f):
	        values = line.strip().rsplit(' ')
	        word = values[0]
	        coefs = np.asarray(values[1:], dtype='float32')
	        self.embeddings_index[word] = coefs
        f.close()
        print('found %s word vectors' % len(self.embeddings_index))
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_word2vec(self):
        from fse import Vectors, Average, IndexedList
        vecs = Vectors.from_pretrained("word2vec-google-news-300")
        word2vecDict = Average(vecs)
        print("Word2vec vocab: "+str(len(word2vecDict.wv.vectors)))
        print(word2vecDict.wv.vectors.shape)
        self.embeddings_index = word2vecDict.wv
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_aravec( self,MODEL_NAME, dim=300):
       if ("cbow" in MODEL_NAME):
          model = Word2Vec.load('~/Embeddings/tweets_cbow_'+str(dim)) #Adjust the location
       else:
          model = Word2Vec.load('~/Embeddings/tweets_sg_'+str(dim)) #Adjust the location
       self.embeddings_index = model.wv
       return model.wv
       
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_glove(self):
        glove_dir = '~/Embeddings/glove.6B' #Adjust the location
        self.embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(self.embeddings_index))
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def create_embeddings_matrix(self, embedding_dim=100, w2v=1):
        count =0
        self.embeddings_matrix = np.random.rand(len(self.vocabulary)+1, embedding_dim)
        for word, i in self.vocabulary.items():
            try:
               if (w2v == 1): 
                   embedding_vector = self.embeddings_index.get_vector(word) 
               else: 
                   embedding_vector= self.embeddings_index.get(word)
            except KeyError:
                   #print(word)
                   count = count +1
                   embedding_vector= np.zeros(embedding_dim, dtype=np.float32) 
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector
        print('Matrix shape: {}'.format(self.embeddings_matrix.shape))
        print('unknown words or OOV'+str(count))
        return self.embeddings_matrix
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_embeddings_layer(self,embeddings_matrix, name, max_len, trainable=False):
        embedding_layer = Embedding(
            input_dim=self.embeddings_matrix.shape[0],
            output_dim=self.embeddings_matrix.shape[1],
            input_length=max_len,
            weights=[self.embeddings_matrix],
            trainable=trainable,
            name=name)
        return embedding_layer

