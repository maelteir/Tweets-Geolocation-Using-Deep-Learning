from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Bidirectional,Dense, Dropout, LSTM, Embedding ,GlobalMaxPooling1D, Flatten ,Conv1D,Input,concatenate, MaxPooling1D
from keras.constraints import max_norm
import matplotlib.pyplot as plt
import tensorflow as tf
from Embeddings import Embeddings
import numpy as np
from sklearn import preprocessing
from keras.models import Model
import keras as Keras

class LSTMModel:
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, _x_train, _y_train, _x_val, _y_val, _STATES_NUM, _MODEL_NAME, _MAX_LEN, _HIDDEN_LAYER_SIZE, _FILTERS, _TruncatingPost, _PaddingPost, _DROPRATE = 0.5, _LSTMLayers = 2, _Arabic =1):
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_val = _x_val
        self.y_val = _y_val
        self.STATES_NUM = _STATES_NUM
        self.MODEL_NAME = _MODEL_NAME
        self.Arabic = _Arabic
        self.MAX_LEN = _MAX_LEN 
        self.tokenizer = None
        self.keras = 1
        self.HIDDEN_LAYER_SIZE=_HIDDEN_LAYER_SIZE
        self.label_encoder= None
        self.Arabic=_Arabic
        self.oneembedding = 0
        self.keras = 1      
        self.filters =_FILTERS
        self.TruncatingPost = _TruncatingPost
        self.PaddingPost = _PaddingPost
        self.LSTMLayers = _LSTMLayers
        self.DropRate = _DROPRATE
        print("Class init - self.filters self.LSTMLayers"+str(self.filters)+" " +str(self.LSTMLayers))        
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
    # encode a list of lines
    def encode_text_forCNN(self, lines):
        # integer encode
        encoded = self.tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=self.MAX_LEN , truncating='post' , padding = 'post')
        return padded  

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def create_label_encoder(self, y_train):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(y_train)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def construct_dataset(self):
        if self.x_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.", self.x_train.shape[0],
                             self.y_train.shape[0])

        if self.x_val.shape[0] != self.y_val.shape[0]:
            raise ValueError("x_val and y_val must have the same number of samples.", self.x_val.shape[0],
                             self.y_val.shape[0])

        if self.tokenizer is None: self.create_tokenizer(list(self.x_train)) #+ list(self.x_val))
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Max document length: %d' % self.MAX_LEN)
        print('Vocabulary size: %d' % self.vocab_size)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(self.x_train.shape[0] + self.x_val.shape[0]))
        self.valX= self.encode_text(self.x_val)
        self.trainX = self.encode_text(self.x_train)

        
        # Prepare dataset for CNN with padding and truncating set to post
        self.trainX_forCNN = self.encode_text_forCNN(self.x_train)
        self.valX_forCNN= self.encode_text_forCNN(self.x_val)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_base_model(self):
        self.model = Sequential() 
        if (self.keras == 1 ):
              self.model.add(Embedding(self.vocab_size, self.vector_size , trainable=self.trainable))
        else:
              self.model.add(Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable))
        z = 0
        while z < (self.LSTMLayers - 1):
             print("inside lstm loop z:" + str(z))
             self.model.add(LSTM(self.filters, dropout= self.DropRate, recurrent_dropout=self.DropRate, return_sequences=True))
             z += 1
        self.model.add(LSTM(self.filters, dropout=self.DropRate, recurrent_dropout=self.DropRate))
        self.model.add(Dropout(self.DropRate))
        self.model.add(Dense(self.STATES_NUM, activation='softmax'))
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_bi_model(self):
        print("Class define_bi_model - self.filters"+str(self.filters)) 
        self.model = Sequential() 
        if (self.keras == 1 ):
              self.model.add(Embedding(self.vocab_size, self.vector_size , trainable=self.trainable))
        else:
              self.model.add(Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable))
        z = 0
        while z < self.LSTMLayers - 1:
             self.model.add(Bidirectional(LSTM(self.filters, dropout=self.DropRate, recurrent_dropout=self.DropRate, return_sequences=True)))
             z += 1
        self.model.add(Bidirectional(LSTM(self.filters, dropout=self.DropRate, recurrent_dropout=self.DropRate)))
        self.model.add(Dropout(self.DropRate))
        self.model.add(Dense(self.STATES_NUM, activation='softmax'))
 
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_complete_model(self):
        if (self.oneembedding == 1 ):
           inputs = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding = Embedding(self.vocab_size, 700 , trainable=self.trainable)(inputs)
           else:
              embedding= Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=self.trainable)(inputs)
           #drop_embed = Dropout(0.2)(embedding)
        # channel 1
        if (self.oneembedding == 0 ):
           inputs1 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding1 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs1)
           else:
              embedding1 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=self.trainable)(inputs1)
           #drop_embed1 = Dropout(0.2)(embedding1)
           conv1 = Conv1D(filters=256, kernel_size=4, activation='relu')(embedding1)
        else:
           conv1 = Conv1D(filters=256, kernel_size=4, activation='relu')(embedding)
        drop1 = Dropout(0.2)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1) #(drop1)
        
        # channel 2
        if (self.oneembedding == 0 ):
           inputs2 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):   
               embedding2 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs2)
           else:
               embedding2 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=self.trainable)(inputs2)
           #drop_embed2= Dropout(0.2)(embedding2)
           conv2 = Conv1D(filters=256, kernel_size=6, activation='relu')(embedding2)
        else:
           conv2 = Conv1D(filters=256, kernel_size=6, activation='relu')(embedding)
        drop2 = Dropout(0.2)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
       
        # channel 3
        if (self.oneembedding == 0 ):
           inputs3 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
               embedding3 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs3)
           else:
               embedding3 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=self.trainable)(inputs3)
           #drop_embed3 = Dropout(0.2)(embedding3)
           conv3 = Conv1D(filters=256, kernel_size=8, activation='relu')(embedding3)
        else:
           conv3 = Conv1D(filters=256, kernel_size=8, activation='relu')(embedding)
        drop3 = Dropout(0.2)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        
        #LSTM
        if (self.oneembedding == 0 ):
           inputs4 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
               embedding4 = Embedding(self.vocab_size, 300, trainable=self.trainable)(inputs4)
           else:
               embedding4 = Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[embeddings_matrix], trainable=self.trainable)(inputs4)
           #drop_embed4 = Dropout(0.2)(embedding4)
           lstm_1 = LSTM(256, dropout=self.DropRate, recurrent_dropout=self.DropRate)(embedding4) #, return_sequences=True
           #lstm_2 = LSTM(512, dropout=self.DropRate, recurrent_dropout=self.DropRate)(lstm_1) 
        else:     
           lstm_1 = LSTM(256, dropout=self.DropRate, recurrent_dropout=self.DropRate)(embedding)    #, return_sequences=True  
           #lstm_2 = LSTM(512, dropout=self.DropRate, recurrent_dropout=self.DropRate)(lstm_1)
        drop4 = Dropout(self.DropRate)(lstm_1)
        #pool4 = MaxPooling1D(pool_size=2)(drop4)
        flat4 = Flatten()(drop4)
        
        merged = concatenate([flat1, flat2, flat3 , flat4])
        print("merged.shape:"+str(merged.shape))
        #maxpool = GlobalMaxPooling1D()(merged)
        #drop = Dropout(0.4) (merged) # (maxpool) 
        outputs = Dense(self.STATES_NUM, activation='softmax')(merged)
        
        if (self.oneembedding == 0 ):
           self.model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
        else:
           self.model = Model(inputs=inputs, outputs=outputs)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_model(self):
        if ("Keras" in self.MODEL_NAME):
              self.keras = 1              
        else:
              self.keras = 0             
                     
        import re
        result = re.search('dim(.*)_', self.MODEL_NAME)
        self.vector_size = int(result.group(1))
        print ("vector_size: " + str(self.vector_size))

        if (self.keras == 0):
          vocabulary = []
          for sentence in list(self.x_train):
             words = sentence.split()
             for word in words:
                vocabulary.append(word)
          print("vocabulary length: "+str(len(vocabulary)))
          
          embed = Embeddings(vocabulary)
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

          embeddings_matrix = embed.create_embeddings_matrix(300, w2v) #1 should be 0 for other than w2v models
          
        self.trainable=True
        if ("trainable0" in self.MODEL_NAME):
            self.trainable=False

        if ("Base" in self.MODEL_NAME):
               self.define_base_model()
        else:
               if ("Bi" in self.MODEL_NAME):
                   self.define_bi_model()
               else:
                   self.define_complete_model()
        
        print(self.model.summary())
        plot_model(self.model, show_shapes=True, to_file='LSTM.png')
        return self.model

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fit_model(self, es,callbacks,N_EPOCHS,BATCH_SIZE, optimizerfile, modelfile = None):
        if (modelfile is None):  
          MODEL = self.model
        else:
          MODEL = self.model
          self.model.load_weights(modelfile)

        if ("Base" in self.MODEL_NAME):
               history = MODEL.fit(self.trainX, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.valX, self.y_val), callbacks=[es,callbacks])
        else:
               if ("Bi" in self.MODEL_NAME):
                   history = MODEL.fit(self.trainX, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.valX, self.y_val), callbacks=[es,callbacks])
               else:
                   if (self.oneembedding == 0 ):
                       history = MODEL.fit([self.trainX_forCNN, self.trainX_forCNN, self.trainX_forCNN, self.trainX], self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=([self.valX_forCNN, self.valX_forCNN, self.valX_forCNN, self.valX], self.y_val), callbacks=[es,callbacks])
                   else:
                       history = MODEL.fit(self.trainX, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.valX, self.y_val), callbacks=[es,callbacks])
        self.model = MODEL
   #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def predict_model(self,x_test):
        testX = self.encode_text(list(x_test))
        testX_forCNN = self.encode_text_forCNN(list(x_test))
        print(testX.shape)
        if ("Base" in self.MODEL_NAME):
               tf_output = self.model.predict(testX, verbose = 1)
        else:
               if ("Bi" in self.MODEL_NAME):
                   tf_output = self.model.predict(testX, verbose = 1)
               else:
                   tf_output = self.model.predict([testX_forCNN, testX_forCNN, testX_forCNN, testX], verbose = 1)
        predictions = np.argmax(tf_output, axis=1)
        print("tf_output: "+str(len(tf_output))+" "+str(tf_output))
        print("predictions: "+str(predictions))
        return predictions
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def save_model(self, checkpointfile, modelfile, optimizerfile):
      #The first step is needed since the last chcekpoint is 
      #the best model not the last saved model
      self.model.load_weights(checkpointfile)
      self.model.save(modelfile)
