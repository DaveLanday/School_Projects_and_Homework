#!/usr/bin/env python3
# encoding: utf-8

import json 
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

 
#Which News outlets are we taking from?
users = ['GuardianUS','nytpolitics','foxnewspolitics','BreitbartNews']

#Define a variable that stores the query information (if limit is exceeded, we will want to stop scraping and wait until we can scrape again):
queries_left = api.rate_limit_status()['resources']['search']


#how many tweets can we gather per query?
#premium API limit: tweepy restricts us to 500 tweets per query. With standard, it is 100 only

#dates_of_interest = [('2017-10-7','2017-11-7'),('2017-9-7','2017-10-6'),('2017-8-7','2017-9-6'),('2017-7-7','2017-8-6'),
#                     ('2017-6-7','2017-7-6'),('2017-5-7','2017-6-6'),('2017-4-7','2017-5-6'),('2017-3-7','2017-4-6'),
#                     ('2017-2-7','2017-3-6'),('2017-1-7','2017-2-6'),('2016-12-7','2017-1-6'),('2016-11-7','2017-12-6')]
dates_of_interest = [('2017-11-7','2017-12-9')]
#get tweets store to a .txt file called 'user_name'

for name in users:
    outfile = open(name+'_tweets_Nov2017_Dec2017.json','w', encoding='utf-8')
    for date in dates_of_interest:
        for tweets in tweepy.Cursor(api.user_timeline,id = name, since=date[0], until = date[1], tweet_mode = 'extended').items(): 
            json.dump(tweets._json, outfile)
            outfile.write('\n')
    outfile.close()
    
 