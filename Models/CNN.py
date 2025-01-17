from pickle import NONE
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import concatenate
import tensorflow as tf 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from Embeddings import Embeddings
import numpy as np
import keras as Keras
import pickle

class CNN:
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, _x_train, _y_train, _x_val, _y_val, _STATES_NUM, _MODEL_NAME, _MAX_LEN, _TruncatingPost, _PaddingPost, _FILTERS=256, _DROPRATE=0.5, _Arabic =1):
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_val = _x_val
        self.y_val = _y_val
        self.STATES_NUM = _STATES_NUM
        self.MODEL_NAME = _MODEL_NAME
        self.MAX_LEN = _MAX_LEN 
        self.Arabic=_Arabic
        self.oneembedding = 1
        self.keras = 1
        self.TruncatingPost = _TruncatingPost
        self.PaddingPost = _PaddingPost
        self.Filters = _FILTERS
        self.DropRate = _DROPRATE
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    # fit a tokenizer
    def create_tokenizer(self,lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    # encode a list of lines
    def encode_text(self, tokenizer, lines):
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
    def construct_dataset(self):
        self.tokenizer = self.create_tokenizer(list(self.x_train)) 
        
        # calculate vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary size: %d' % self.vocab_size)
        
        # encode data
        self.trainX = self.encode_text(self.tokenizer, list(self.x_train))        
        self.valX = self.encode_text(self.tokenizer, list(self.x_val))

    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_model(self):
        if ("CNN1" in self.MODEL_NAME):
           if ("Keras" in self.MODEL_NAME):
              self.oneembedding = 1
              self.keras = 1              
           else:
              self.oneembedding = 1
              self.keras = 0             
        else:
           if ("CNN3" in self.MODEL_NAME ):
            if ("Keras" in self.MODEL_NAME):
                self.oneembedding = 0
                self.keras = 1                
            else:
                self.oneembedding = 0
                self.keras = 0                
        import re
        result = re.search('dim(.*)_', self.MODEL_NAME)
        self.vector_size = int(result.group(1))
        print ("vector_size: " + str(self.vector_size))

        if (self.keras == 0):
          embed = Embeddings(self.tokenizer.word_index)
          w2v = 0
          if ("glove" in self.MODEL_NAME):
              embed.load_glove() 

          if ("word2vec" in self.MODEL_NAME):
              embed.load_word2vec() 
              w2v = 1

          if ("fasttext" in self.MODEL_NAME):
              embed.load_fasttext(self.Arabic) 

          if ("aravec" in self.MODEL_NAME):
              embed.load_aravec(self.MODEL_NAME, self.vector_size) 
              w2v = 1

          embeddings_matrix = embed.create_embeddings_matrix(self.vector_size, w2v) #1 should be 0 for other than w2v models #self.vector_size was 300

        trainable=True
        if ("trainable0" in self.MODEL_NAME):
                 trainable=False

        if (self.oneembedding == 1 ):
           inputs = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding = Embedding(self.vocab_size, self.vector_size , trainable=trainable)(inputs)
           else:
              embedding= Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=trainable)(inputs)
           #drop_embed = Dropout(0.2)(embedding)
           
        # channel 1
        if (self.oneembedding == 0 ):
           inputs1 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding1 = Embedding(self.vocab_size, self.vector_size, trainable=trainable)(inputs1)
           else:
              embedding1 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=trainable)(inputs1)
           #drop_embed1 = Dropout(0.2)(embedding1)     
           conv1 = Conv1D(filters=self.Filters, kernel_size=4, activation='relu')(embedding1) #4
        else:
           conv1 = Conv1D(filters=self.Filters, kernel_size=4, activation='relu')(embedding) #4
        drop1 = Dropout(self.DropRate)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        
        # channel 2
        if (self.oneembedding == 0 ):
           inputs2 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):   
               embedding2 = Embedding(self.vocab_size, self.vector_size, trainable=trainable)(inputs2)
           else:
               embedding2 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=trainable)(inputs2)
           #drop_embed2= Dropout(0.2)(embedding2)    
           conv2 = Conv1D(filters=self.Filters, kernel_size=6, activation='relu')(embedding2) #6
        else:
           conv2 = Conv1D(filters=self.Filters, kernel_size=6, activation='relu')(embedding) #6
        drop2 = Dropout(self.DropRate)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        
        # channel 3
        if (self.oneembedding == 0 ):
           inputs3 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
               embedding3 = Embedding(self.vocab_size, self.vector_size, trainable=trainable)(inputs3)
           else:
               embedding3 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=trainable)(inputs3)
           #drop_embed3 = Dropout(0.2)(embedding3)    
           conv3 = Conv1D(filters=self.Filters, kernel_size=8, activation='relu')(embedding3) #8
        else:
           conv3 = Conv1D(filters=self.Filters, kernel_size=8, activation='relu')(embedding) #8
        drop3 = Dropout(self.DropRate)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
       
        #channel 4
        if ("Channels4" in self.MODEL_NAME):
          if (self.oneembedding == 0 ):
              inputs4 = Input(shape=(self.MAX_LEN,))
              if (self.keras == 1 ):
                   embedding4 = Embedding(self.vocab_size, self.vector_size, trainable=trainable)(inputs4)
              else:
                   embedding4 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=trainable)(inputs4)
              #drop_embed4 = Dropout(0.2)(embedding4)     
              conv4 = Conv1D(filters=self.Filters, kernel_size=10, activation='relu')(embedding4) #10
          else:
              conv4 = Conv1D(filters=self.Filters, kernel_size=10, activation='relu')(embedding) #10
          drop4 = Dropout(self.DropRate)(conv4)
          pool4 = MaxPooling1D(pool_size=2)(drop4)
          flat4 = Flatten()(pool4)
     
        # merge
        if ("Channels4" in self.MODEL_NAME):
            merged = concatenate([flat1, flat2, flat3, flat4])
        else:
            merged = concatenate([flat1, flat2, flat3])
        # interpretation
        #dense1 = Dense(256, activation='relu')(merged)
        #drop = Dropout(0.4)(merged)
        outputs = Dense(self.STATES_NUM, activation='softmax')(merged)
        if (self.oneembedding == 0 ):
           if ("Channels4" in self.MODEL_NAME):
                self._model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
           else:
                self._model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        else:
           self._model = Model(inputs=inputs, outputs=outputs)

        print(self._model.summary())
        plot_model(self._model, show_shapes=True, to_file='CNN.png')
        return self._model

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fit_model(self, es, callbacks,N_EPOCHS,BATCH_SIZE, optimizerfile, modelfile = None):
        if (modelfile is None):  
          MODEL = self._model
        else:
          MODEL = self._model
          self._model.load_weights(modelfile)
      
       
        if ( "CNN3" in self.MODEL_NAME):
           if("Channels4" in self.MODEL_NAME):
                MODEL.fit([self.trainX, self.trainX, self.trainX, self.trainX], self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=([self.valX, self.valX, self.valX, self.valX], self.y_val), callbacks=[es,callbacks])
           else:
                MODEL.fit( [self.trainX, self.trainX, self.trainX], self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=([self.valX, self.valX, self.valX], self.y_val), callbacks=[es,callbacks])
        else: 
           if("CNN4" in self.MODEL_NAME):
               MODEL.fit([self.trainX, self.trainX, self.trainX, self.trainX], self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=([self.valX, self.valX, self.valX, self.valX], self.y_val) , callbacks=[es,callbacks])
           else:
               if ("CNN1" in self.MODEL_NAME): 
                   MODEL.fit(self.trainX, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.valX, self.y_val), callbacks=[es,callbacks])
        self._model = MODEL
        
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def predict_model(self,x_test):
        testX = self.encode_text(self.tokenizer, list(x_test))
        print(testX.shape)
        if ( "CNN3" in self.MODEL_NAME): 
          if("Channels4" in self.MODEL_NAME):
             tf_output = self._model.predict([testX,testX,testX,testX], verbose = 1)
          else:
             tf_output =self._model.predict([testX,testX,testX], verbose = 1)
        else:
             tf_output = self._model.predict(testX, verbose = 1)
        predictions = np.argmax(tf_output,axis=1)
        print("tf_output: "+str(len(tf_output))+" "+str(tf_output))
        print("predictions: "+str(predictions))
        return predictions
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def save_model(self, checkpointfile, modelfile, optimizerfile):
      #The first step is needed since the last chcekpoint is 
      #the best model not the last saved model 
         
      self._model.load_weights(checkpointfile)
      self._model.save(modelfile)



