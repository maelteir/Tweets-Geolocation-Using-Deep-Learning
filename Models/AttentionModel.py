import keras as Keras
from tensorflow.keras.layers import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dense, Embedding, SpatialDropout1D, Bidirectional, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten
import numpy as np
from keras import backend as K


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class AttentionLayer(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape = (input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, np.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
        

from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from Embeddings import Embeddings
import keras.layers as L
from keras.models import Model
import tensorflow as tf
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Attention:
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, _x_train, _y_train, _x_val, _y_val, _STATES_NUM, _MODEL_NAME, _MAX_LEN, _FILTERS, _TruncatingPost, _PaddingPost, _DROP_RATE, _Arabic =1):
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_val = _x_val
        self.y_val = _y_val
        self.STATES_NUM = _STATES_NUM
        self.MODEL_NAME = _MODEL_NAME
        self.Arabic = _Arabic
        self.MAX_LEN = _MAX_LEN 
        self.tokenizer = None
        self.filters = _FILTERS
        self.TruncatingPost = _TruncatingPost
        self.PaddingPost = _PaddingPost
        self.DropRate = _DROP_RATE
        #self.NUM_OUTPUTS = _NUM_OUTPUTS
                
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
            raise ValueError("x_val and y_dev must have the same number of samples.", self.x_val.shape[0],
                             self.y_val.shape[0])

        if self.tokenizer is None: self.create_tokenizer(list(self.x_train) + list(self.x_val))
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Max document length: %d' % self.MAX_LEN)
        print('Vocabulary size: %d' % self.vocab_size)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(self.x_train.shape[0] + self.x_val.shape[0]))
        self.valX = self.encode_text(self.x_val)
        self.trainX = self.encode_text(self.x_train)
        

         
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_complete_model(self):
        if (self.oneembedding == 1 ):
           inputs = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding = Embedding(self.vocab_size, 700 , trainable=self.trainable)(inputs)
           else:
              embedding= Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable)(inputs)
           #drop_embed = Dropout(0.2)(embedding)
        # channel 1
        if (self.oneembedding == 0 ):
           inputs1 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
              embedding1 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs1)
           else:
              embedding1 = Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable)(inputs1)
           #drop_embed1 = Dropout(0.2)(embedding1)
           conv1 = Conv1D(filters=256, kernel_size=4, activation='relu')(embedding1)
        else:
           conv1 = Conv1D(filters=256, kernel_size=4, activation='relu')(embedding)
        drop1 = Dropout(0.2)(conv1)
        #pool1 = MaxPooling1D(pool_size=2)(drop1)
        #flat1 = Flatten()(pool1) #(drop1)
        att1 = AttentionLayer(self.MAX_LEN-4+1)(drop1)
        avg_pool1 = GlobalAveragePooling1D()(drop1)
        max_pool1 = GlobalMaxPooling1D()(drop1)
        x1 = concatenate([att1,avg_pool1, max_pool1])
        
        # channel 2
        if (self.oneembedding == 0 ):
           inputs2 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):   
               embedding2 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs2)
           else:
               embedding2 = Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable)(inputs2)
           #drop_embed2= Dropout(0.2)(embedding2)
           conv2 = Conv1D(filters=256, kernel_size=6, activation='relu')(embedding2)
        else:
           conv2 = Conv1D(filters=256, kernel_size=6, activation='relu')(embedding)
        drop2 = Dropout(0.2)(conv2)
        #pool2 = MaxPooling1D(pool_size=2)(drop2)
        #flat2 = Flatten()(pool2)
        att2 = AttentionLayer(self.MAX_LEN-6+1)(drop2)
        avg_pool2 = GlobalAveragePooling1D()(drop2)
        max_pool2 = GlobalMaxPooling1D()(drop2)
        x2 = concatenate([att2,avg_pool2, max_pool2])
        
        # channel 3
        if (self.oneembedding == 0 ):
           inputs3 = Input(shape=(self.MAX_LEN,))
           if (self.keras == 1 ):
               embedding3 = Embedding(self.vocab_size, 700, trainable=self.trainable)(inputs3)
           else:
               embedding3 = Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable)(inputs3)
           #drop_embed3 = Dropout(0.2)(embedding3)
           conv3 = Conv1D(filters=256, kernel_size=8, activation='relu')(embedding3)
        else:
           conv3 = Conv1D(filters=256, kernel_size=8, activation='relu')(embedding)
        drop3 = Dropout(0.2)(conv3)
        att3 = AttentionLayer(self.MAX_LEN-8+1)(drop3)
        avg_pool3 = GlobalAveragePooling1D()(drop3)
        max_pool3 = GlobalMaxPooling1D()(drop3)
        x3 = concatenate([att3,avg_pool3, max_pool3])
        #pool3 = MaxPooling1D(pool_size=2)(drop3)
        #flat3 = Flatten()(pool3)
        
            
        #merged = concatenate([flat1, flat2, flat3])
        merged = concatenate([x1, x2, x3])
        #att = AttentionLayer(165)(merged) #self.MAX_LEN)(merged)
        #x = concatenate([merged,avg_pool1, max_pool1])
        preds = Dense(self.STATES_NUM, activation='softmax')(merged)
        if (self.oneembedding == 0 ):
           self.model = Model(inputs=[inputs1, inputs2, inputs3], outputs=preds)
        else:
           self.model = Model(inputs=inputs, outputs=preds)
        
                    
        print(self.model.summary())
        plot_model(self.model, show_shapes=True, to_file='Attention.png')
        return self.model
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_base_model(self):
        sequence_input = Input(shape=(self.MAX_LEN,))
        if (self.keras == 1 ):
               embedding_layer = Embedding(self.vocab_size, self.vector_size , trainable=self.trainable)
        else:
               embedding_layer = Embedding(input_dim=self.embeddings_matrix.shape[0], output_dim=self.embeddings_matrix.shape[1], input_length=self.MAX_LEN, weights=[self.embeddings_matrix], trainable=self.trainable)
        x = embedding_layer(sequence_input)
        x = SpatialDropout1D(self.DropRate)(x)
        x = Bidirectional(LSTM(self.filters, return_sequences=True))(x) #L.CuDNNLSTM
        att = AttentionLayer(self.MAX_LEN)(x)
        avg_pool1 = GlobalAveragePooling1D()(x)
        max_pool1 = GlobalMaxPooling1D()(x)
        x = concatenate([att,avg_pool1, max_pool1])
        preds = Dense(self.STATES_NUM, activation='softmax')(x)
        self.model = Model(sequence_input, preds)
                    
        print(self.model.summary())
        #plot_model(model, show_shapes=True, to_file='multichannel_multiembedding_trainable.png')
        return self.model
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_model(self):
        self.oneembedding = 1
        self.keras = 1      
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
       # print ("vector_size: " + str(self.vector_size))

        if (self.keras == 0):
          vocabulary = []
          for sentence in list(self.x_train):
             words = sentence.split()
             for word in words:
                vocabulary.append(word)
         # print("vocabulary length: "+str(len(vocabulary)))
          
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
              embed.load_aravec(self.MODEL_NAME, vector_size) 
              w2v = 1

        #vocabulary = sentences #embeddings_index.keys() # replace this by the vocabulary of the dataset you want to train
        if (self.keras == 0 ):
               self.embeddings_matrix = embed.create_embeddings_matrix(300, w2v) #1 should be 0 for other than w2v models

        self.trainable=True
        if ("trainable0" in self.MODEL_NAME):
                 self.trainable=False
       
        if ("CNN3" in self.MODEL_NAME):
               self.define_complete_model()
        else:
               self.define_base_model()
        return self.model

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fit_model(self, es, callbacks, N_EPOCHS, BATCH_SIZE, optimizerfile, modelfile = None):
         if (modelfile is None):  
          MODEL = self.model
         else:
          MODEL = self.model
          self.model.load_weights(modelfile)

         if ( "CNN3" in self.MODEL_NAME):
            MODEL.fit( [self.trainX, self.trainX, self.trainX], self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=([self.valX, self.valX, self.valX], self.y_val), callbacks=[es,callbacks])
         else: 
            MODEL.fit(self.trainX, self.y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(self.valX, self.y_val), callbacks=[es,callbacks])
         self.model = MODEL    
          
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def predict_model(self,x_test):
        testX = self.encode_text(list(x_test))
        if ( "CNN3" in self.MODEL_NAME): 
             tf_output =self.model.predict([testX,testX,testX], verbose = 1)
        else:
             tf_output = self.model.predict(testX, verbose = 1)
        predictions = np.argmax(tf_output, axis=1)
      
        return predictions

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def save_model(self, checkpointfile, modelfile, optimizerfile):
      #The first step is needed since the last chcekpoint is 
      #the best model not the last saved model
       
      self.model.load_weights(checkpointfile)
      self.model.save(modelfile)
  

