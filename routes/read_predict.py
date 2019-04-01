import pandas as pd
import keras
import json
import sys

from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


import nltk
import nltk.classify.util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd 
import pickle
import re
import sys
import os

cwd = os.getcwd()

f = open(cwd+"/routes/input.json")

data = json.load(f) 

f.close()
continu=True
for tweet in data['statuses']:
    if continu:
        df=pd.DataFrame.from_dict(tweet,orient="index").T
        dfi=dict(df["user"])
        df['user_id']=dfi[0]['id']
        df['name']=dfi[0]['name']
        continu=False
    else:
        dft=pd.DataFrame.from_dict(tweet,orient="index").T
        dfi=dict(dft["user"])
        dft['user_id']=dfi[0]['id']
        dft['name']=dfi[0]['name']
        df=pd.concat([df,dft],axis=0)


df.reset_index(drop=True,inplace=True)

good_columns=["created_at","user_id","name","text","geo"]
df=df[good_columns]
model_input=df['text']


def build_model(output_emb = 32, lstm_out_dim = 128, dense_size = 64):
    
    voc_len = 10000
    max_len_sentence = 64
    output_emb = output_emb

    lstm_out_dim = lstm_out_dim
    dropout_lstm = 0.3
    recurrent_dropout = 0.2
    dense_size = dense_size
    dense_dropout = 0.3

    model = Sequential()

    model.add(Embedding(voc_len, output_emb, input_length = max_len_sentence))
    model.add(Bidirectional(LSTM(lstm_out_dim, return_sequences=True, dropout = dropout_lstm, recurrent_dropout = recurrent_dropout)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(dense_size, activation="relu"))
    model.add(Dropout(dense_dropout))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model




Dictionnary = np.loadtxt(cwd+"/routes/Dict.txt", dtype='str')

def Tokenisation(text): 
    '''sentence = "The Quick brown fox, Jumps over the lazy little dog. Hello World. Good thing"'''
    sentence = text

    sentence = sentence.replace(".", " ")
    sentence = sentence.replace(",", " ")
    sentence.split(" ")
    sentence = re.sub('[^a-zA-Z]',' ', sentence)

    stop_words = set(stopwords.words('english'))
    lemma = nltk.wordnet.WordNetLemmatizer()

    TokenisedWords = word_tokenize(sentence)
    Tags = nltk.pos_tag(TokenisedWords)

    filtered_sentence = []

    for w in TokenisedWords:
        if w not in stop_words:
            filtered_sentence.append(lemma.lemmatize(w).lower())


    return filtered_sentence


def GetIndexFromWord(word):

    for line in range(len(Dictionnary)):
        if Dictionnary[line] == word:
            return line+1
        if(line == len(Dictionnary)-1):
            return 0
            
def GetNormalizedArray(TwitterText):

    Results = []
    if(len(TwitterText) < 64):
        for i in range(len(TwitterText)):
            Results.append(GetIndexFromWord(TwitterText[i]))
        while (len(Results) != 64):
            Results.append(0)
    
    return Results
            

model = build_model()

model.summary()


model.load_weights(cwd+"/routes/model_best.hdf5")

Total = []
for i in range(0,len(model_input)):

    ResultArray = GetNormalizedArray(Tokenisation(model_input[i]))
    Total.append(ResultArray)
    
model_input=np.array(Total)


label_predict=model.predict_classes(model_input)
df['label']=label_predict

df[["text","label"]]
df.to_csv(cwd+"/routes/output.csv",sep=",")

sys.stdout.flush()