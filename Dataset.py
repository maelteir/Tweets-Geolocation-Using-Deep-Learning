import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

class Dataset:

    def __init__(self,_filename,_lang="Arabic",_tweets_per_user="all",_features_num=3,_ASCII=0):
        self.filename = _filename
        self.lang = _lang
        self.tweets_per_user = _tweets_per_user
        self.features_num = _features_num
        self.ASCII=_ASCII
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def Concat_Fields_Arabic(self,text,location,name):
        if not location: 
            location = " "        
        if not name: 
            name = " "
            
        if self.features_num == 1:
            concattext = str(text)
        elif self.features_num == 2:
            concattext = " أنا أسمى "+str(name)+ " و رأى "+str(text)
        elif self.features_num == 3:
            concattext = " أنا أسكن فى "+str(location)+" أسمي "+str(name)+ " و رأى "+str(text)
            
        return concattext
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def Concat_Fields_English(self,text,location,name):
        if not location: 
            location = " "
            
        if not name: 
            name = " "
            
        if self.features_num == 1:
            concattext = str(text)
        elif self.features_num == 2:
            concattext = "I am from "+str(location)+ " My opinion is "+str(text)
        elif self.features_num == 3:
            concattext = "I am from "+str(location)+" My name is "+str(name)+ " My opinion is "+str(text)     
            
        return concattext
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def plotRowsPerState2(self,df):
        labels = []
        counts = []
        with open('/kaggle/input/tweets-geolocation-sourcecode/DeepLearningCode/States', 'r') as states_file: #Adjust the location
            	lines = states_file.readlines()
        for l in range(15):
            count = list(df['label']).count(l)
            labels.append(l)
            counts.append(count)
                        
        plt.bar(labels,counts)
        plt.ylim(0, 5000)
        plt.show()
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def plotRowsPerState(self,df,title):
        states = []
        counts = []
        with open('/kaggle/input/tweets-geolocation-sourcecode/DeepLearningCode/States', 'r') as states_file: #Adjust the location
            lines = states_file.readlines()
        for line in lines:    
            s= line.replace('\n','')
            count = df['state'].value_counts()[s]
            states.append(s)
            counts.append(count)
                       
        df2 = df.groupby(['AuthorId'])['AuthorId'].count()
        annotation2 = "Tweets per author mean: " + str(round(df2.mean(),2)) +" standard deviation: " + str(round(df2.std(),2))+" mode: " + str(round(df2.mode(),2))+" max: " + str(round(df2.max(),2))+" min: " + str(round(df2.min(),2))
        annotation ="Mean: " + str(round(np.mean(counts),2)) +" Standard Deviation: " + str(round(np.std(counts),2))+" Variance: " + str(round(np.var(counts),2))+'\n'
        plt.title(title)
        plt.suptitle(annotation+annotation2)
        plt.bar(states,counts)
        plt.ylim(0, 5000)
        plt.show()

    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def preprocess(self):
        data_path = self.filename+"_"+self.tweets_per_user+"tweetsperuser.csv" 
        print("The data file: "+data_path)
        all_data = pd.read_csv(data_path, on_bad_lines='skip',  delimiter="\t",engine='python')   
        all_df = pd.DataFrame(all_data)
        
        all_df.columns = all_df.columns.str.strip()
        self.plotRowsPerState(all_df,"All tweets (train and test)")
        LE = LabelEncoder()
        all_df['label'] = LE.fit_transform(all_df['state'])
        labels = list(LE.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,11,12]))
       

        if (self.ASCII == 0):
           all_df= all_df.loc[:, ['AuthorId','Preprocessed','Preprocessed_name','Preprocessed_location','label','state']]
           all_df['Preprocessed']= all_df['Preprocessed'].astype(str) #all_df['preprocessed'].astype(str)
           all_df['Preprocessed_name']= all_df['Preprocessed_name'].astype(str) #all_df['preprocessed'].astype(str)
           all_df['Preprocessed_location']= all_df['Preprocessed_location'].astype(str) #all_df['preprocessed'].astype(str)
        else:
           all_df= all_df.loc[:, ['AuthorId','ASCII','ASCII_name','ASCII_location','label','state']]
           all_df['ASCII']=all_df['ASCII'].astype(str) 
           all_df['ASCII_name']=all_df['ASCII_name'].astype(str)   
           all_df['ASCII_location']=all_df['ASCII_location'].astype(str)

        #count = all_df.AuthorId.unique().size
                      
        if (self.lang == "Arabic"):
            if (self.ASCII == 0):
                all_df['final'] = all_df.apply(lambda x: self.Concat_Fields_Arabic(x['Preprocessed'],x['Preprocessed_location'],x['Preprocessed_name']), axis=1)
            else:
                all_df['final'] = all_df.apply(lambda x: self.Concat_Fields_Arabic(x['ASCII'],x['ASCII_location'],x['ASCII_name']), axis=1)
        else:
            if (self.ASCII == 0):
                all_df['final'] = all_df.apply(lambda x: self.Concat_Fields_English(x['Preprocessed'],x['Preprocessed_location'],x['Preprocessed_name']), axis=1)
            else:
                all_df['final'] = all_df.apply(lambda x: self.Concat_Fields_English(x['ASCII'],x['ASCII_location'],x['ASCII_name']), axis=1)

        all_df['final']=all_df['final'].astype(str)
        all_df["Number of Words"] = all_df["final"].apply(lambda n: len(n.split()))
        max_words = all_df["Number of Words"].max()
        mean_words = all_df["Number of Words"].mean()
        print ("max words: "+str(max_words)+"  mean_words:"+str(mean_words) )
        self.plotRowsPerState2(all_df)
          
        x_all = all_df['final'].values
        y_all = all_df['label'].values
    
        x_train, x_rem, y_train, y_rem = train_test_split(x_all, y_all,test_size = 0.2, shuffle= True,random_state = 42, stratify = y_all) 
        x_test, x_val, y_test, y_val= train_test_split(x_rem, y_rem,test_size = 0.5, shuffle= True,random_state = 42, stratify = y_rem)
        print ("Train: Length of x:"+str(len(x_train)) + "  Length of y:"+str(len(y_train)) )
        print ("Val: Length of x:"+str(len(x_val)) + "  Length of y:"+str(len(y_val)) )
        print ("Test: Length of x:"+str(len(x_test)) + "  Length of y:"+str(len(y_test)) )

        
        return  labels, np.array2string(all_df.columns.values), x_train, x_val, x_test, y_train, y_val, y_test
        
        

