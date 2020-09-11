import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec,KeyedVectors
from sklearn.model_selection import train_test_split


class Tweet_Sentiment_Analyzer:
    def __init__(self):
        pass

    def read_data(self):
        """
            Data Input
        """
        data = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1',header=None)
        data.columns = ['Sentiment','id','Date','flag','user','tweet']
        data = data.sample(frac = 1)
        return data.head(800000)   

    def remove_pattern(self,input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    def clean_tweets(self,data):
        """
        Cleaning the tweets data.
        """
        tweet_3 = pd.DataFrame(data,columns=['tweet'])
        tweet_3['Real_tweet'] = tweet_3['tweet']
        print("convert to lowercase")
        tweet_3['tweet'] = tweet_3['tweet'].str.lower()
        print("removing the tweets username.")
        tweet_3['tweet'] = np.vectorize(self.remove_pattern)(tweet_3['tweet'], "@[\w]*")
        print("removing all the RT: text from the Tweets")
        tweet_3['tweet'] = np.vectorize(self.remove_pattern)(tweet_3['tweet'], "RT :")
        tweet_3['tweet'] = tweet_3['tweet'].str.replace('[^a-zA-Z#]',' ')
        tweet_3['tweet'] = np.vectorize(self.remove_pattern)(tweet_3['tweet'], "\n")
        print("removing links")
        tweet_3['tweet'] = np.vectorize(self.remove_pattern)(tweet_3['tweet'], "r'^https?:\/\/.*[\r\n]*'")
        #removing the emoticons
        #tweet_3['tweet'] = tweet_3['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        tweet_3['Tweets_len'] = tweet_3['tweet'].apply(len)
        #removing tweets totally in different languages
        drop_tweets = tweet_3[tweet_3['Tweets_len'] == 0].index
        tweet_3.drop(index=drop_tweets,inplace=True)

        return tweet_3

    def token_check(self,x,model):
        """
        1.Check if the token exists in the word2vec model vocab. 
        2.Check if the length of the token is greater than 3 
        """
        my_list=[]
        for i in x:
            if len(i) > 3 and i in model.vocab:
                my_list.append(i)
            else:
                continue
        return my_list


    def word_vector(self,tokens, size,model):
        """
        Averaging the word vectors
        """
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokens:
            vec += model[word].reshape((1, size))
            count += 1.
        if count != 0:
            vec /= count
        return vec


tsa = Tweet_Sentiment_Analyzer()

data = tsa.read_data()

print(data.Sentiment.value_counts())

data.loc[data['Sentiment'] == 4,'Sentiment'] = 1

#Cleaning tweet data
clean_data = tsa.clean_tweets(data)

merged_data = pd.merge(data,clean_data.tweet,left_on=clean_data.index,right_on=clean_data.index)
merged_data = merged_data.loc[:,merged_data.columns.isin(['Sentiment','tweet_x','tweet_y'])]

# Tokenization
merged_data['tweet_y_tokens'] = merged_data['tweet_y'].apply(lambda x: x.split())


"""
#########################################
--------Word Embedding - Word2Vec
#########################################
"""
#Download the Google Word Embeddings
#!wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True,limit=100000)

merged_data['tweet_y_tokens'] = merged_data['tweet_y_tokens'].apply(lambda x:tsa.token_check(x,model))

print(merged_data.info())

wordvec_arrays = np.zeros((len(merged_data.tweet_y_tokens), 300))

print(wordvec_arrays.shape)

#Averaging the word vectors in a tweet before feeding the data to a model.
for i in range(len(merged_data.tweet_y_tokens)):
    wordvec_arrays[i,:] = tsa.word_vector(merged_data.tweet_y_tokens[i], 300,model)
    if i % 100000 == 0:
        print("Processed",i,"records..!!")
wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df['sentiment'] = merged_data['Sentiment']

#Changing the datatype of the columns in the dataframe.
for cols in wordvec_df.columns:
    wordvec_df[cols] = wordvec_df[cols].astype('float16')

print(wordvec_df.info())

print(wordvec_df.sentiment.value_counts())

"""
###################################################
#------------Data Spliting---------------
###################################################
"""
#splitting of data
X = wordvec_df.loc[:,~wordvec_df.columns.isin(['sentiment'])]
y = wordvec_df.loc[:,wordvec_df.columns.isin(['sentiment'])]

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25,random_state=101)

"""
###################################################
#------------Artificial Neural Netork---------------
###################################################
"""
from keras import models
from keras import layers
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras import backend as K
from matplotlib import pyplot as plt


def NN_arch1(lrate=0.0001):
    #one input one output #zero hidden layers
    model = models.Sequential()
    model.add(layers.Dense(300,input_dim = 300, activation='relu'))
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(7,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=lrate)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

def model_fit(model,epoch_val=50):
    callbacks = EarlyStopping(monitor='val_loss',mode='min',patience=3)
    model.fit(X_train, y_train, epochs=epoch_val,batch_size=32)
    val_loss, val_acc = model.evaluate(X_test,y_test)
    print(val_loss, val_acc)
    return val_loss,val_acc

def predicted(prediction):
    for i in prediction:
        print("")
        for j in i:
            if j > 0.5:
                print(1," ",end="")
            else:
                print(0," ",end="")

def history_plot(history):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    training_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.figure(figsize=(5,3))

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    #Visualize accuracy history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training acc', 'Test acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show();


model_NN = NN_arch1(0.0001)

val_loss,val_acc = model_fit(model_NN,10)

history = model_NN.fit(X_train,y_train,epochs=10,verbose=0,validation_data=(X_test, y_test)) 
history_plot(history)

model_NN.save('/sTwitter_Sentiment_NN_model.h5')
"""
##################################
#---------LightGBM---------------
##################################
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate = 0.125, metric = 'l1', n_estimators = 20, num_leaves = 38)

param_grid = {
    'n_estimators': [x for x in [150]],
    'learning_rate': [0.25],
    'num_leaves': [32],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501]}

gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train.values.ravel(),eval_set = [(X_test, y_test)],eval_metric = ['auc', 'binary_logloss'],early_stopping_rounds = 10)

print('Best parameters found by grid search are:', gridsearch.best_params_)

gbm = lgb.LGBMClassifier(boosting_type= 'gbdt', learning_rate=0.25, n_estimators=150, num_leaves= 32,\
                         objective= 'binary', random_state= 501)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=20)

#Metric Validation
y_pred_prob = gbm.predict_proba(X_test)[:, 1]
auc_roc_0 = str(roc_auc_score(y_test, y_pred_prob))
print('AUC: \n' + auc_roc_0)

##############################################
#---------------Sample Prediction:
##############################################

def clean_tweet_content(tweet):
    print("convert to lowercase")
    tweet = tweet.lower()
    print("removing the tweets username.")
    tweet = np.vectorize(tsa.remove_pattern)(tweet, "@[\w]*")
    print("removing all the RT: text from the Tweets")
    tweet = np.vectorize(tsa.remove_pattern)(tweet, "RT :")
    print("removing links")
    tweet = np.vectorize(tsa.remove_pattern)(tweet, "r'^https?:\/\/.*[\r\n]*'")
    tweet = np.vectorize(tsa.remove_pattern)(tweet,"\n")
    tweet = np.array_str(tweet) 
    print(type(tweet))
    tweet = tweet.replace('[^a-zA-Z#]',' ')
    return tweet


clean_tweet = clean_tweet_content(tweet)
tweet_tokens = list(clean_tweet.split(" "))
tweet_tokens_filtered = tsa.token_check(tweet_tokens,model)

_arrays = np.zeros((1, 300))
_arrays[0,:] = tsa.word_vector(tweet_tokens_filtered,300,model)
vectorized_array = pd.DataFrame(_arrays)

print(vectorized_array)

print(vectorized_array.iloc[:,0:300])

pred = model_NN.predict([vectorized_array.iloc[:,0:300]])

print(predicted(pred))



