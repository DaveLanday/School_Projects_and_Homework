#!/usr/bin/env python
# -*- coding: utf-8 -*-

# jarvis.py
# dlanday

import websocket
import pickle
import json
import urllib
import requests
import sqlite3
import sklearn # you can import other sklearn stuff too!
# FILL IN ANY OTHER SKLEARN IMPORTS ONLY
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

import botsettings # local .py, do not share!!
TOKEN = botsettings.API_TOKEN
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print(*args)


try:
    conn = sqlite3.connect("jarvis.db")
except:
    debug_print("Can't connect to sqlite3 database...")

#Create a connection cursor object:
c = conn.cursor()

def post_message(message_text, channel_id):
    requests.post("https://slack.com/api/chat.postMessage?token={}&channel={}&text={}&as_user=true".format(TOKEN,channel_id,message_text))

#Here is a function that adds training data to the db file:
def insert_train_data(msg_txt,action):
    """INSERT_TRAIN_DATA: While in training mode, this function takes user 
    messages and the given action NAME and stores them into the training database"""
    with conn:
        c.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", (msg_txt, action,))

class Jarvis():
    
    def __init__(self): # initialize Jarvis
        self.JARVIS_MODE = None
        self.ACTION_NAME = None
        
        
    
    
        # SKLEARN STUFF HERE:
        #self.BRAIN = None # FILL THIS IN
        
        #Create a pipeline to help build B.O.W:
        self.BRAIN = Pipeline([('vect', CountVectorizer()),#Essentially encodes each word in the document by its frequency
                               ('tfidf', TfidfTransformer()),#larger documents may have higher freqs. This acts as a normalizer
                               ('clf', MultinomialNB()),])#This is the classifier model. We are using Naive Bayes
        
    def on_message(self, ws, message):
        m = json.loads(message)
        debug_print(m, self.JARVIS_MODE, self.ACTION_NAME)
        
        
        
        #Define Action Names:
        actionNames = ['TIME','PIZZA','GREET','WEATHER','JOKE']
        
        #Define TRAIN mode and TEST mode as strings:
        TRAIN = 'TRAIN'
        TEST = 'TEST'
        
#       #Make lists to store training features and training target data:
        x_train = []
        y_train = []
        
        for row in c.execute("SELECT * from training_data"):#make sure new data is being entered correctly to the table
                #print(row)
                x_train.append(''.join(list(map(type('U35'),row[1]))))
                y_train.append(''.join(list(map(type('U35'),row[2]))))
                
        # only react to Slack "messages" not from bots (me):
        if m['type'] == 'message' and 'bot_id' not in m:
            
            #Define Training Mode:
            if m['text'] == 'training time':
                self.JARVIS_MODE = TRAIN #set JARVIS_MODE to training
                post_message('OK, I\'m ready for training. What NAME should this ACTION be?',m['channel'])
            
            if m['text'] in actionNames:
                self.ACTION_NAME = m['text']
                post_message('OK, let\'s call this action ' + self.ACTION_NAME + '. Now give me some training text!', m['channel'])
                # The conditions are met for the next loop, make sure they are not so that user can ask a question:
                m['type'] = 'user_typing'
                   
            if self.JARVIS_MODE == TRAIN and self.ACTION_NAME != None and m['type'] != 'user_typing':
                #Store training data:
                if m['text'] != 'done':
                    post_message('Ok, I\'ve got it. What else?',m['channel'])
                    #Store training data:
                    insert_train_data(m['text'],self.ACTION_NAME)
                
                else:
                    self.JARVIS_MODE = None
                    self.ACTION_NAME = None
                    post_message('OK, I\'m finished training.',m['channel']) #Indicate escape from training mode
            
 ######################################################################################################################           
            # Define Testing Mode:        
            if m['text'] == 'testing time':#set JARVIS_MODE to testing
                self.JARVIS_MODE = TEST
                post_message('I\'m training my brain with the data you\'ve given me...',m['channel'])
                # Run the learning algorithm on the training data:
                self.BRAIN.fit(x_train,y_train)
                #Pickle (serialize) Jarvis' Brain fit to the training data:
                brain_out = open('jarvis_brain.pkl','wb')
                pickle.dump(self.BRAIN,brain_out)
                brain_out.close()
                post_message('OK, I\'m ready for testing. Write me something and I\'ll try to figure it out.',m['channel'])
                m['type'] = 'user_typing'
                #Ask Jarvis a question:
                
            if self.JARVIS_MODE == TEST and  m['type'] != 'user_typing':
                if m['text'] != 'done':
                    test_document = [] #list of user input you want to test (foces correct dtype)
                    #test_document.append(m['text'])# append user input for testing
                    test_document.append(''.join(list(map(type('U35'),m['text'])))) #ensures dtype('<U35') array is being passed
                    post_message('Ok, I think the action you mean is ' + self.BRAIN.predict(test_document)[0], m['channel'])
                    post_message('Write me something else and I\'ll try to figure it out.',m['channel'])
                else:
                    self.JARVIS_MODE = None
                    post_message('OK, I\'m finished testing.',m['channel']) #Indicate escape from testing mode
            pass

        #print(type(m['text']))
        #print(self.BRAIN.predict(m['text']))
        #Print the mode and the action name as a sanity check:
        #print('\n',self.JARVIS_MODE,self.ACTION_NAME)

def start_rtm():
    """Connect to Slack and initiate websocket handshake"""
    r = requests.get("https://slack.com/api/rtm.start?token={}".format(TOKEN), verify=False)
    r = r.json()
    r = r["url"]
    return r


def on_error(ws, error):
    print("SOME ERROR HAS HAPPENED", error)


def on_close(ws):
    conn.close()
    print("Web and Database connections closed")


def on_open(ws):
    print("Connection Started - Ready to have fun on Slack!")



r = start_rtm()
jarvis = Jarvis()
ws = websocket.WebSocketApp(r, on_message=jarvis.on_message, on_error=on_error, on_close=on_close)
ws.run_forever()


