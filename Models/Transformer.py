#References: 
#https://keras.io/examples/nlp/text_classification_with_transformer/
#https://keras.io/guides/keras_nlp/transformer_pretraining/

import numpy as np 
import tensorflow as tf 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import tensorflow as tf
from Embeddings import Embeddings

import pandas as pd 
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
tqdm.pandas()

import random
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K

import gc


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # self-attention layer
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # layer norm
        ffn_output = self.ffn(out1)  #feed-forward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # layer norm

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embed_dim, trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim , trainable=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class Transformer:
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, _x_train, _y_train, _x_val, _y_val, _STATES_NUM, _MODEL_NAME, _MAX_LEN, _FILTERS, _TruncatingPost, _PaddingPost, _DROPRATE=0.5, _LSTMLayers = 1,  _Arabic =1):
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_val = _x_val
        self.y_val = _y_val
        self.STATES_NUM = _STATES_NUM
        self.tokenizer = None
        self.MODEL_NAME = _MODEL_NAME
        self.Arabic = _Arabic
        self.MAX_LEN = _MAX_LEN 
        self.Filters = _FILTERS
        self.TruncatingPost = _TruncatingPost
        self.PaddingPost = _PaddingPost
        self.LSTMLayers = _LSTMLayers
        self.DropRate = _DROPRATE
                
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    # fit a tokenizer
    def create_tokenizer(self,lines):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(lines)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    # encode a list of lines
    def encode_text(self, lines):
       # integer encode
        encoded = self.tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        if (self.TruncatingPost == 1):
            if (self.PaddingPost == 1):
                padded = pad_sequences(encoded, maxlen=self.MAX_LEN , truncating='post' , padding = 'post')
            else:
                padded = pad_sequences(encoded, maxlen=self.MAX_LEN , truncating='post' , padding = 'pre') 
 
        else:
            if (self.PaddingPost == 1):
                padded = pad_sequences(encoded, maxlen=self.MAX_LEN , truncating='pre' , padding = 'post')
            else:
                padded = pad_sequences(encoded, maxlen=self.MAX_LEN , truncating='pre' , padding = 'pre') 
        return padded  
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def create_label_encoder(self, y_train):
        self._label_encoder = preprocessing.LabelEncoder()
        self._label_encoder.fit(y_train)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def construct_dataset(self):
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.", self.x_train.shape[0],
                             self.y_train.shape[0])

        if self.x_val.shape[0] != self.y_val.shape[0]:
            raise ValueError("x_val and y_val must have the same number of samples.", self.x_val.shape[0],
                             self.y_val.shape[0])

        if self.tokenizer is None: self.create_tokenizer(list(self.x_train) + list(self.x_val))
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Max document length: %d' % self.MAX_LEN)
        print('Vocabulary size: %d' % self.vocab_size)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(self.x_train.shape[0] + self.x_val.shape[0]))
        self.x_val = self.encode_text(self.x_val)
        self.x_train = self.encode_text(self.x_train)
        

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_model(self):
        if ("LSTM" in self.MODEL_NAME):
           if ("Keras" in self.MODEL_NAME):
              self.keras = 1              
           else:
              self.keras = 0             
                     
        import re
        result = re.search('dim(.*)_', self.MODEL_NAME)
        vector_size = int(result.group(1))
        print ("vector_size: " + str(vector_size))
      
        trainable=True
        if ("trainable0" in self.MODEL_NAME):
                 trainable=False
       
        num_heads = 2 # Number of attention heads
        ff_dim = self.Filters #64 # Hidden layer size in feed forward network inside transformer
        
        self.model = Sequential()   
        self.model.add(layers.Input(shape=(self.MAX_LEN, )))
        self.model.add(TokenAndPositionEmbedding(self.MAX_LEN, self.vocab_size, vector_size))

        z = 0
        while z < (self.LSTMLayers - 1):
             self.model.add(TransformerBlock(vector_size, num_heads, ff_dim))  
             z += 1
        self.model.add(TransformerBlock(vector_size, num_heads, ff_dim))  
        self.model.add(layers.GlobalAveragePooling1D())
        self.model.add(layers.Dropout(self.DropRate))
        self.model.add(layers.Dense(ff_dim, activation='relu'))
        self.model.add(layers.Dropout(self.DropRate))
        self.model.add(layers.Dense( self.STATES_NUM, activation='softmax'))
        
        print(self.model.summary())
        plot_model(self.model, show_shapes=True, to_file='Transformer.png')
        return self.model

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fit_model(self, es, callbacks, N_EPOCHS, BATCH_SIZE, optimizerfile, modelfile = None):
         if (modelfile is None):  
          MODEL = self.model
         else:
          MODEL = self.model
          self.model.load_weights(modelfile)

         history = MODEL.fit(self.x_train, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.x_val, self.y_val), callbacks=[es,callbacks])
         self.model = MODEL
         
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def predict_model(self,x_test):
        testX = self.encode_text(list(x_test))
        print(testX.shape)
        tf_output = self.model.predict(testX, verbose = 1)
        predictions = np.argmax(tf_output, axis=1)
        return predictions

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def save_model(self, checkpointfile, modelfile, optimizerfile):
      #The first step is needed since the last chcekpoint is 
      #the best model not the last saved model
      self.model.load_weights(checkpointfile)
      self.model.save(modelfile)
    
    



