import tweepy
import sys
import pandas as pd
#import numpy as np
import time
import re
#from TweetMiner import TweetMiner
from tweepy.utils import import_simplejson
json = import_simplejson()
import time
from threading import Thread
import random
from queue  import Queue
thread_processed=0
stream_count=0

from tweepy.models import Status


queue = Queue()
consumer_key= 'BIfCgx5b74RJtkfVwxYCerVab'
consumer_secret= 'jZIhjW6ohl6QFqgXMWSES7piYF8MmAlHGkOLIJcb5p0k1WszaT'

#Setting connection of app to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)





class MyStreamListener(tweepy.StreamListener):
    
    def __init__(self, api=None):
        self.api = api or tweepy.API()
        self.t_end = time.time()+ 60*7
        global queue
        self.normal_count=1
        #self.tweet_miner = TweetMiner()

        #self.begin_index = pd.read_csv('collection.csv', index_col = 0 ,encoding = 'ISO-8859-1',header=None,skiprows=3)
        #self.stream_count= self.begin_index.index.max()
        self.stream_count=1

        
        self.all_of_them= pd.DataFrame(columns=('id','tweetText','userName','date','hash_str','user_url','rt_count','reply_id'))

    def initialize(self):
        self.m=""
    def on_data(self,data):
        full_text=""

        data2 = json.loads(data)

        if 'extended_tweet' in data2 :
            if('full_text' in data2["extended_tweet"]):
            
                full_text=bytes(str(data2["extended_tweet"]["full_text"]).encode("utf-8"))
                full_text=full_text.decode('utf-8')
                print('FUL TEXT *******************************************************************************')
                print(full_text)
 
            #print(self.find_between( data, '"extended_tweet":{"full_text":"','",'))
            #print(data)
        if("retweeted_status" in data2):
            if('full_text' in data2["retweeted_status"]):
                full_text=bytes(str(data2["retweeted_status"]["full_text"]).encode("utf-8"))
                full_text=full_text.decode('utf-8')
                print('FUL TEXT *******************************************************************************')
                print(full_text)
        #print(full_text)


        data = json.loads(data)

        if 'in_reply_to_status_id' in data:
            status = Status.parse(self.api, data)
            if self.on_status(status,full_text) is False:
                return False
        elif 'delete' in data:
            delete = data['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'event' in data:
            status = Status.parse(self.api, data)
            if self.on_event(status) is False:
                return False
        elif 'direct_message' in data:
            status = Status.parse(self.api, data)
            if self.on_direct_message(status) is False:
                return False
        elif 'friends' in data:
            if self.on_friends(data['friends']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(data['limit']['track']) is False:
                return False
        elif 'disconnect' in data:
            if self.on_disconnect(data['disconnect']) is False:
                return False
        elif 'warning' in data:
            if self.on_warning(data['warning']) is False:
                return False
        else:
            logging.error("Unknown message type: " + str(raw_data))

    def on_status(self, status,full_text):
        if(status.lang=="en"):
            if(full_text==""):
                tweetText=bytes(str(status.text).encode("utf-8"))
                tweetText = tweetText.decode('utf-8')
            else:
                tweetText=bytes(str(full_text).encode("utf-8"))
                tweetText = tweetText.decode('utf-8')
            #parsing urls.
            tweetTexts = re.split('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweetText)
            tweetText = ''.join(tweetTexts)
                
            #parsing hashtags.
            tweetTextss = re.split(r"#\w+", tweetText)
            tweetText = ''.join(tweetTextss)



            hash_ts=[]
            for hashtag in status.entities ['hashtags']:
                hash_t=hashtag['text']
                if hash_t:
                    hash_ts.append(hash_t)


            user_urls=[]
            for user_url in status.entities ['urls']:
                user_u=user_url['expanded_url']
                if user_u:
                    user_urls.append(user_u)    
                    

                      
            userName=status.user.screen_name

            id = status.id
            id_str = status.id_str
            date = status.created_at
            user_url = status.user.url
            rt_count= status.retweet_count
            reply_id= status.in_reply_to_status_id
            repiled_id_list= None


            
            # if tweet is a retweet.
            if hasattr(status,'retweeted_status'):

                retweeted_status= status.retweeted_status
                if(full_text==""):
                    tweetText=bytes(str(retweeted_status.text).encode("utf-8"))
                    tweetText = tweetText.decode('utf-8')
                else:
                    tweetText=bytes(str(full_text).encode("utf-8"))
                    tweetText = tweetText.decode('utf-8')

                #parsing urls.
                tweetTexts = re.split('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweetText)
                tweetText = ''.join(tweetTexts)


                #parsing hashtags.
                tweetTextss = re.split(r"#\w+", tweetText)
                tweetText = ''.join(tweetTextss)

                


                
                id_str = retweeted_status.id_str
                date = retweeted_status.created_at
                user_url = retweeted_status.user.url
                rt_count= retweeted_status.retweet_count
                reply_id= retweeted_status.in_reply_to_status_id
                repiled_id_list= status.user.id
                
                
                hash_ts=[]
                for hashtag in status.entities ['hashtags']:
                    hash_t=hashtag['text']
                    if hash_t:
                        hash_ts.append(hash_t)


                user_urls=[]
                for user_url in status.entities ['urls']:
                    user_u=user_url['expanded_url']
                    if user_u:
                        user_urls.append(user_u) 
                        

            
            
            hash_str = ', '.join(hash_ts)

            url_str = ', '.join(user_urls)
        


            if(len(hash_ts) > 0):
                if(len(user_urls)>0):
                    
                    
                    d= {'ID':id,'Tweet':tweetText,'UserName':userName,'Publication Time':date,'Hashtags':hash_str,'User URLS':url_str,'# RT':rt_count,'Replier Id Strs':reply_id,'Retweeted Tweet User ID':repiled_id_list}
                    df= pd.DataFrame(d,index=[self.stream_count+1])

                else:
                    
                    d= {'ID':id,'Tweet':tweetText,'UserName':userName,'Publication Time':date,'Hashtags':hash_str,'User URLS':None,'# RT':rt_count,'Replier Id Strs':reply_id,'Retweeted Tweet User ID':repiled_id_list}
                    df= pd.DataFrame(d,index=[self.stream_count+1])

                self.all_of_them=self.all_of_them.append(df)


            else:
                if(len(user_urls)>0):
                
                    d= {'ID':id,'Tweet':tweetText,'UserName':userName,'Publication Time':date,'Hashtags':None,'User URLS':url_str,'# RT':rt_count,'Replier Id Strs':reply_id,'Retweeted Tweet User ID':repiled_id_list}
                    df= pd.DataFrame(d,index=[self.stream_count+1])

                else:
                    d= {'ID':id,'Tweet':tweetText,'UserName':userName,'Publication Time':date,'Hashtags':None,'User URLS':None,'# RT':rt_count,'Replier Id Strs':reply_id,'Retweeted Tweet User ID':repiled_id_list}
                    df= pd.DataFrame(d,index=[self.stream_count+1])
                    

               
                # adding tweets into dataframe container.
                #self.all_of_them=self.all_of_them.append(df)
           # queue.put(df)

            #print("mert")
            print(self.normal_count)
            print(tweetText)

            
                #self.all_of_them.to_csv('company_ve_journal_politics5.csv',header=False,mode= 'w')
            df.to_csv('new_set4.csv',header=False,mode= 'a',encoding='utf-8')

            self.stream_count+=1
            self.normal_count+=1

            
            #self.tweet_miner.do_process(df,self.stream_count)



            return True
        
    def on_error(self, status_code):
        print (status_code)
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

    def next_tuple(self):
        if( queue.empty()):
            return
        else:
            df= queue.get()
            queue.task_done()
            self.emit([df])

        
#Dynamically fetch access token
#dbg
print ('1!!')

access_token= '629518185-dsQxbBTIAvtD4QJZemMXJHLZfrg1Csle6gwwE1eX'
access_token_secret= 'AmnM8KRcmdU7llZGuVHg3T9mBxUZIKzBNiEohATWELNdg'

auth.set_access_token(access_token, access_token_secret)
auth.secure = True
#dbg
print ('2!!')

api = tweepy.API(auth)
#dbg
print ('3!!')



myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, language="en", listener=myStreamListener,tweet_mode="extended")
print ('4!!')

myStream.filter(track=['italy', 'justin', 'horan', 'australia', 'karma', 'katy', 'asia', 'lou', 'alexander rybak', 'bill', 'titanium', 'uk', 'dem', 'red carpet', 'cyrus', 'rocky balboa', 'vegas', 'sc', 'camila', 'adam', 'israel', 'liam', 'representatives', 'white supremacists', 'sean', 'hilary', 'pantera', 'armenia', 'potus', 'oakley', 'nate', 'morgan freeman', 'president', 'turkey', 'republican', 'evanescence', 'bbc', 'european', 'cat', 'game of thrones', 'kkk', 'apocalypse', 'turkish', 'chuck', 'tariq', 'house representatives', 'uber', 'white privilege', 'kabuki', 'lp', 'cub swanson', 'earth', 'commonwealth games', 'polonium tea syndrome', 'healthcare', 'healthcare bill', 'american people', 'cold war', 'jesus', 'bananies', 'gorilla', '2016 presidential campaign', 'des', 'libs', 'health bill', 'death panel', 'mayo', 'electoral college', 'chameleon', 'townhall', 'reps', 'racism', 'insulin', 'watergate', 'liberals', 'stoke', 'gender studies', 'genocide', 'halsey', 'reichert', 'civil rights movement', 'pro socialism', 'senators', 'phil', 'tomahawks', 'planned parenthood', 'cantando', 'republicans', 'institutional racism', 'tories', 'belgium', 'health plan', 'portuguese', 'moon', 'starving', 'liberal', 'ces', 'socialist', 'trumpanzees', 'computer networking', 'kylo ren', 'trump/russia', 'tomahawk', 'military', "we can't stop", 'trumps', 'white supremacist', 'spicy', 'shashlik rap', 'la', 'jim comey', 'nial', 'yoongi', 'avocadies', 'ibs', 'ed balls', 'cali', 'daniel', 'democratic', 'sun', 'uranium', 'vodka', 'sessions', 'penicillin', 'us', 'abortion', 'whites', 'unicorns', 'canonization', 'the death panel'],async=True)
print ('5!!')
#ConsumerThread().start()
#ConsumerThread().start()
#ConsumerThread().start()
#ConsumerThread().start()
#ConsumerThread().start()
