#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Print system configurations
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#import tensorflow as tf
import transformers as trans
import keras
import torch
import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import tensorflow as tf

print ("Keras version", keras.__version__)
print ("Tensorflow version", tf.__version__)
print ("Torch version", torch.__version__)
print ("Transformers version", trans.__version__)

device_name = tf.test.gpu_device_name() #"/device:CPU:0" #tf.test.gpu_device_name() 
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read command line arguments
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--states_num", type=int, help="Number of states/provinces of the studied country.", default=13)
parser.add_argument("--max_words", type=int, help="Max number of words to analyze per tweet.", default=50)
parser.add_argument("--features_num", type=int, help="Number of features to use. default tweet and meta data", default=3)
parser.add_argument("--epochs", type=int, help="Number of epochs.", default=5)
parser.add_argument("--batch", type=int, help="Batch size.", default=32)
parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=5e-05)
parser.add_argument("--tweets_per_user", type=int, help="Number of tweets per user.", default=1000)
parser.add_argument("--lang", type=str, help="Tweet language.", default="Arabic")
parser.add_argument("--data_file", type=str, help="Input data file name.", default="")
parser.add_argument("--model_name", type=str, help="Model name.", default="bert-base-uncased")
parser.add_argument("--hidden_layer_size", type=int, help="Hidden layer size", default=128)
parser.add_argument("--ASCII", type=int, help="Use ASCII fields.", default=0)
parser.add_argument("--load", type=int, help="Load previous weights.", default=0)
parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file to load.", default="")
parser.add_argument("--filters", type=int, help="Number of filters.", default=256)
parser.add_argument("--lstmlayers", type=int, help="Number of lstm layers.", default=2)
parser.add_argument("--truncatingpost", type=int, help="Tokenizer truncating post.", default=1)
parser.add_argument("--paddingpost", type=int, help="Tokenizer padding post.", default=1)
parser.add_argument("--droprate", type=float, help="LSTM drop rate.", default=0.5)
args = parser.parse_args()


STATES_NUM = args.states_num 
print("Number of states/provinces: " + str(args.states_num))

MAX_LEN = args.max_words 
print("maximum words: " + str(args.max_words))

FEATURES_NUM = args.features_num 
print("Number of features: " + str(args.features_num))

N_EPOCHS = args.epochs
print("Epochs: " + str(args.epochs))

BATCH_SIZE = args.batch
print("Batch size: " + str(args.batch))

LEARNING_RATE = args.learning_rate
print("Learning rate: " + str(args.learning_rate))

TWEETS_PER_USER = args.tweets_per_user
print("Tweets per user: " + str(args.tweets_per_user))

LANGUAGE = args.lang
print("Tweets language: " + str(args.lang))

FILE_NAME = args.data_file
print("Data file name: " + str(args.data_file))

MODEL_NAME = args.model_name
print("The model name: " + str(args.model_name))

HIDDEN_LAYER_SIZE = args.hidden_layer_size
print("The LSTM model hidden layer size: " + str(args.hidden_layer_size))

ASCII = args.ASCII
print("Use ASCII Fields: " + str(args.ASCII))

LOAD = args.load
print("Load previous weights: " + str(args.load))

CHECKPOINT_FILE = args.checkpoint_file
print("Checkpoint file: " + CHECKPOINT_FILE)

FILTERS = args.filters
print("Number of filters: " + str(args.filters))

LSTMLayers = args.lstmlayers
print("Number of layers: " + str(args.lstmlayers))

TruncatingPost = args.truncatingpost
print("Tokenizer tuncating post: " + str(args.truncatingpost))

PaddingPost = args.paddingpost
print("Tokenizer padding post: " + str(args.paddingpost))

DROP_RATE = args.droprate
print("LSTM drop rate: " + str(args.droprate))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prepare the dataset
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from Dataset import Dataset

s=str(TWEETS_PER_USER)
if (TWEETS_PER_USER>=1000):
    s = "all"    
dataset = Dataset(_filename=FILE_NAME, _lang=LANGUAGE, _tweets_per_user=s,_features_num=FEATURES_NUM, _ASCII=ASCII)
labels, features, x_train, x_val, x_test, y_train, y_val, y_test =  dataset.preprocess()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the model and construct the dataset
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from Models.CNN import CNN
from Models.Bert import Bert
from Models.LSTMModel import LSTMModel
from Models.AttentionModel import Attention
from Models.Transformer import Transformer

Arabic =1 if (LANGUAGE == "Arabic") else 0

# Construct dataset
if ("Attention" in MODEL_NAME):
    attention  = Attention( x_train, y_train, x_val, y_val, STATES_NUM, MODEL_NAME, MAX_LEN, FILTERS, TruncatingPost,PaddingPost, DROP_RATE, Arabic)
    attention.construct_dataset()
elif ("CNN" in MODEL_NAME):
    cnn = CNN(x_train, y_train, x_val, y_val, STATES_NUM, MODEL_NAME,MAX_LEN,TruncatingPost,PaddingPost, FILTERS, DROP_RATE, Arabic)
    cnn.construct_dataset()
elif ("LSTM" in MODEL_NAME):
    lstm = LSTMModel(x_train, y_train, x_val, y_val, STATES_NUM, MODEL_NAME, MAX_LEN, HIDDEN_LAYER_SIZE, FILTERS, TruncatingPost,PaddingPost, DROP_RATE, LSTMLayers, Arabic)
    lstm.construct_dataset()
elif ("Bert" in MODEL_NAME):
    bert  = Bert( x_train, y_train, x_val, y_val, STATES_NUM, MODEL_NAME, MAX_LEN, Arabic)
    bert.construct_dataset(BATCH_SIZE)
elif ("Transformer" in MODEL_NAME):
    transformer  = Transformer( x_train, y_train, x_val, y_val, STATES_NUM, MODEL_NAME, MAX_LEN, FILTERS, TruncatingPost,PaddingPost, DROP_RATE, LSTMLayers, Arabic)
    transformer.construct_dataset()
else:
    print("Undefined Model Type!!!!! The only known models are CNN, Bert, LSTM, Attention, and Transfomer. Please use any of them.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Compile the model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#from official.nlp import optimization
from keras.optimizers import Adam
#from tensorflow.keras.optimizers.legacy import Adam
import transformers
from transformers import AdamWeightDecay


tf.random.set_seed(42)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy()
loss_desc = str(loss)

#AdamW Optimizer Setup
init_lr = LEARNING_RATE #5e-05 #2e-5
train_data_size = len(x_train)
steps_per_epoch = train_data_size // BATCH_SIZE
num_train_steps = steps_per_epoch * N_EPOCHS
num_warmup_steps = num_train_steps // 10
if ("Transformer" in MODEL_NAME) :   
   #optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
   optimizer_type = Adam
   optimizer = optimizer_type(learning_rate=init_lr)   
elif ("Bert" in MODEL_NAME) or ("Marabert" in MODEL_NAME):
   #optimizer = AdamWeightDecay(learning_rate =init_lr)
   optimizer=tf.keras.optimizers.AdamW(learning_rate=init_lr)
else:
   optimizer_type = Adam
   optimizer = optimizer_type(learning_rate=init_lr)
   
optimizer_desc = str(optimizer)

with tf.device(device_name): #'/CPU:0'):
    if  ("Attention" in MODEL_NAME):
         model = attention.define_model()
    elif ("LSTM" in MODEL_NAME):
         model = lstm.define_model()
    elif ("Bert" in MODEL_NAME):
         model = bert.define_model()
    elif ("CNN" in MODEL_NAME):
         model = cnn.define_model()
    elif ("Transformer" in MODEL_NAME):
         model = transformer.define_model()
    else:
         print("Undefined Model Type!!!!! The only known models are CNN, Bert, LSTM, Attention, and Transfomer. Please use any of them.")

   
    model.compile(optimizer= optimizer, loss=loss, metrics=['accuracy'])
   
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Train and save the model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow.keras.backend as K
import pickle

filename=MODEL_NAME.replace('/','') +"_adamw_"+FILE_NAME.replace('/','')+"_"+str(FEATURES_NUM)+"features_"+str(TWEETS_PER_USER)+"tweetsperuser_"+str(MAX_LEN)+"words_"+str(BATCH_SIZE)+"batch_"+str(LEARNING_RATE)+"learningrate"

Model_path ="/kaggle/working/"
Model_file = Model_path+filename+".keras"
with open(Model_path+filename+".txt", "w") as f:
    f.write("Model:\n "+MODEL_NAME+"\n\n")
    f.write("Tokenizer:\n BertTokenizer\n\n")
    f.write("Tweets per user:"+str(TWEETS_PER_USER)+"\n\n")
    f.write("Dataset Train size:\n "+str(len(x_train))+ "  Validation size: "+str(len(x_val))+ " Test size: "+str(len(x_test)) +"\n\n")
    f.write("Optimizer:\n "+optimizer_desc+" Learning rate: "+str(init_lr)+" num_train_steps: "+str(num_train_steps)+" num_warmup_steps: "+str(num_warmup_steps)+"\n\n")
    f.write("Loss:\n "+loss_desc+"\n\n")
    f.write("Max words length per tweet:\n "+str(MAX_LEN)+"\n\n")
    f.write("Batch size:\n "+str(BATCH_SIZE)+"\n\n")
    f.write("Epochs:\n "+str(N_EPOCHS)+"\n\n")
    

if  (not CHECKPOINT_FILE):
	checkpoint_path = "/kaggle/working/"+filename+".keras" 
else:
	checkpoint_path = "/kaggle/working/"+CHECKPOINT_FILE+".keras" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, mode='min', save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

Optimizer_File = Model_path+filename+"optimizer.pkl"
if (LOAD == 1):
  MODEL_FILE = Model_file
else:
  MODEL_FILE = None

with tf.device(device_name): #'/CPU:0'):
    if  ("Attention" in MODEL_NAME):      
         attention.fit_model(es,cp_callback,N_EPOCHS, BATCH_SIZE,Optimizer_File, MODEL_FILE)
    elif ("LSTM" in MODEL_NAME):
         lstm.fit_model(es, cp_callback,N_EPOCHS,BATCH_SIZE,Optimizer_File, MODEL_FILE)
    elif ("Bert" in MODEL_NAME):
         bert.fit_model(es, cp_callback,N_EPOCHS, Model_path+filename, Optimizer_File, MODEL_FILE)
    elif ("CNN" in MODEL_NAME):
         cnn.fit_model(es, cp_callback,N_EPOCHS,BATCH_SIZE, Optimizer_File, MODEL_FILE)
    elif ("Transformer" in MODEL_NAME):
         transformer.fit_model(es,cp_callback,N_EPOCHS, BATCH_SIZE, Optimizer_File, MODEL_FILE)
    else:
         print("Undefined Model Type!!!!! The only known models are CNN, Bert, LSTM, Attention, and Transfomer. Please use any of them.")         

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate the model
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


with tf.device(device_name):#'/CPU:0'): 
    if  ("Attention" in MODEL_NAME):
         attention.save_model(checkpoint_path, Model_path+filename+".keras", Optimizer_File)
         x_test_predict = attention.predict_model(x_test)
    elif ("LSTM" in MODEL_NAME):
         lstm.save_model(checkpoint_path, Model_path+filename+".keras", Optimizer_File)
         x_test_predict = lstm.predict_model(x_test)
    elif ("Bert" in MODEL_NAME):
         bert.save_model(checkpoint_path, Model_path+filename+".keras", Optimizer_File)
         x_test_predict = bert.predict_model(x_test)
    elif ("CNN" in MODEL_NAME):
         cnn.save_model(checkpoint_path, Model_path+filename+".keras", Optimizer_File)
         x_test_predict = cnn.predict_model(x_test)
    elif ("Transformer" in MODEL_NAME):
         transformer.save_model(checkpoint_path, Model_path+filename+".keras", Optimizer_File)
         x_test_predict = transformer.predict_model(x_test)
    else:
         print("Undefined Model Type!!!!! The only known models are CNN, Bert, LSTM, Attention, and Transfomer. Please use any of them.")        
         

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, x_test_predict)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, x_test_predict, average='micro')
print('Precision: %f' % precision)
precision = precision_score(y_test, x_test_predict, average='weighted')
print('Precision weighted: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, x_test_predict, average='micro')
print('Recall: %f' % recall)
recall = recall_score(y_test, x_test_predict, average='weighted')
print('Recall weighted: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, x_test_predict, average='micro')
print('F1 score: %f' % f1)
f1 = f1_score(y_test, x_test_predict, average='weighted')
print('F1 score weighted: %f' % f1)

# kappa
kappa = cohen_kappa_score(y_test, x_test_predict)
print('Cohens kappa: %f' % kappa)

# confusion matrix
matrix = confusion_matrix(y_test, x_test_predict)
print(matrix)
from textwrap import wrap
labels = [ '\n'.join(wrap(l, 8)) for l in labels ]
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels= labels)
disp.plot()
plt.show()

print(classification_report(y_test, x_test_predict))

