from arabert import ArabertPreprocessor
from transformers import AutoConfig, TFAutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, Trainer, TrainingArguments, TFAutoModelForMaskedLM, DistilBertTokenizerFast, TFBertForSequenceClassification, BertTokenizer, DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#For Marabert
from transformers import AutoTokenizer

class Bert:
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, _x_train, _y_train, _x_val, _y_val, _STATES_NUM, _MODEL_NAME, _MAX_LEN, _Arabic =1):
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_val = _x_val
        self.y_val = _y_val
        self.STATES_NUM = _STATES_NUM
        self.MODEL_NAME = _MODEL_NAME
        self.Arabic = _Arabic
        self.MAX_LEN = _MAX_LEN                        
      
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def preprocess_input_data(self, texts, tokenizer, max_len=80):
        # Tokenize the text data using the tokenizer
        tokenized_data = [tokenizer.encode_plus(
            t,
            max_length=max_len,
            pad_to_max_length=True,
            add_special_tokens=True) for t in texts]

        # Extract tokenized input IDs and attention masks
        input_ids = [data['input_ids'] for data in tokenized_data]
        attention_mask = [data['attention_mask'] for data in tokenized_data]

        return input_ids, attention_mask
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def construct_dataset(self,_BATCH_SIZE):
      self.BATCH_SIZE = _BATCH_SIZE
      if (self.Arabic == 1):
          """
            ArabertPreprocessor(
                   model_name= "", keep_emojis = False, remove_html_markup = True, replace_urls_emails_mentions = True, strip_tashkeel = True,
                   strip_tatweel = True, insert_white_spaces = True, remove_non_digit_repetition = True, replace_slash_with_dash = None,
                   map_hindi_numbers_to_arabic = None, apply_farasa_segmentation = None)
          """
          arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-large-arabertv02-twitter")
          x_train = [arabert_prep.preprocess(item) for item in self.x_train]
          x_val = [arabert_prep.preprocess(item) for item in self.x_val]  
          if ("Marabert" in self.MODEL_NAME):
               self.MODEL_NAME = "UBC-NLP/MARBERTv2"
               self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERTv2")
          elif ("Arabicbert" in self.MODEL_NAME):
               self.MODEL_NAME = "asafaya/bert-base-arabic"
               self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
          else:
               self.MODEL_NAME = "aubmindlab/bert-base-arabertv2" 
               self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
      else:
          if (self.MODEL_NAME == "distilbert-base-uncased"):
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_NAME)
          else:
            self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
      print(self.MODEL_NAME)
      
      '''
      print("Training Sentence Lengths: ")
      plt.hist([ len(self.tokenizer.tokenize(sentence)) for sentence in self.x_train],bins=range(0,300,2))
      plt.show()
      plt.hist([ len(self.tokenizer.tokenize(sentence)) for sentence in self.x_val],bins=range(0,300,2))
      plt.show()
      '''
      # Preprocess the training data
      self.X_train_input_ids, X_train_attention_mask = self.preprocess_input_data(x_train, self.tokenizer,self.MAX_LEN)
      self.X_val_input_ids, X_train_attention_mask = self.preprocess_input_data(x_val, self.tokenizer,self.MAX_LEN)    
      
       

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def define_model(self):
        if (self.Arabic == 1):
            bert_model = TFBertForSequenceClassification.from_pretrained(self.MODEL_NAME, return_dict=True, num_labels=self.STATES_NUM) 
        else:
            if (self.MODEL_NAME == "distilbert-base-uncased"):
                bert_model = TFDistilBertForSequenceClassification.from_pretrained(self.MODEL_NAME, return_dict=True, num_labels=self.STATES_NUM)
            else:
                bert_model = TFBertForSequenceClassification.from_pretrained(self.MODEL_NAME, return_dict=True, num_labels=self.STATES_NUM)
                
        input_ids =  tf.keras.Input(shape=(self.MAX_LEN,), name='word_inputs', dtype='int32')
        attention_mask = tf.where(input_ids != 0, 1, 0)  # Creating attention mask
        bert_encodings = bert_model(input_ids, attention_mask=attention_mask)[0]

        outputs = tf.keras.layers.Dense(self.STATES_NUM, activation='softmax', name='outputs')(bert_encodings)#(doc_encoding)
        self.model = tf.keras.Model(inputs=[input_ids], outputs=[outputs])

        print(self.model.summary())
        plot_model(self.model, show_shapes=True, to_file='Bert.png')
        return self.model

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def fit_model(self, es, callbacks,N_EPOCHS,filename, optimizerfile, modelfile = None):
        self.model.fit(x=self.X_train_input_ids, y=self.y_train.tolist(), epochs=N_EPOCHS, batch_size=self.BATCH_SIZE, validation_data=(self.X_val_input_ids, self.y_val.tolist()), callbacks=[es,callbacks])
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def predict_item(self, item):         
          input_ids, attention_mask = self.preprocess_input_data(item, self.tokenizer,self.MAX_LEN)
          
          tf_output = self.model.predict(input_ids, verbose = 0)
          predictions = np.argmax(tf_output,axis=1)
          return predictions
          
    def predict_model(self, x_test):
      print('Evaluating the model on test data...')
      X_test_input_ids, X_test_attention_mask = self.preprocess_input_data(x_test, self.tokenizer,self.MAX_LEN)
      y_prob_test = self.model.predict(X_test_input_ids, verbose=True)
      prediction =  [np.argmax(x) for x in list(y_prob_test)] 

      return prediction

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    def save_model(self, checkpointfile, modelfile, optimizerfile):
      self.model.load_weights(checkpointfile)
      self.model.save(modelfile)

