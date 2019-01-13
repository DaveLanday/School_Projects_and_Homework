#!/usr/bin/env python3
# encoding: utf-8

import json 
import sys
import tweepy

import Get_News_Tweets #Local .py do not share!

#Define twitter app credentials:
consumer_key        = Get_News_Tweets.consumer_key
consumer_secret     = Get_News_Tweets.consumer_secret
access_token_key    = Get_News_Tweets.API_Token
access_token_secret = Get_News_Tweets.API_SECRET
 
#Define handler:
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

#Build API:
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True) #handles API rate limits for us!

#error handling:
if not api:
    print('problem connecting to API')
    sys.exit(-1)
    
#Rest of Code#

#names of outlets:
users = ['GuardianUS','nytpolitics','foxnewspolitics','BreitbartNews']

#dates of interest  (to be specified):
dates_of_interest = [('2017-11-11','2017-12-11')]

#Define a query:
#query = "Trump%20from%3Anytpolitics%2C%20OR%20from%3ABreitbartNews%2C%20OR%20from%3Afoxnewspolitics%2C%20OR%20from%3AGuardianUS%20since%3A2017-11-11%20until%3A2017-12-11&src=typd&lang=en"
query2 = "Trump"

for name in users:
    with open(name+'_Trump_Tweets_'+dates_of_interest[0][0]+'_'+dates_of_interest[0][1]+'.json', 'w', encoding='utf-8') as outfile:
        for tweet in tweepy.Cursor(api.user_timeline,q=query2,id = name, since=dates_of_interest[0][0], until = dates_of_interest[0][1]).items():
            json.dump(tweet._json,outfile)
            outfile.write('\n')

            


