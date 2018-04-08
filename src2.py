# Libraries used
import pandas as pd
import numpy as np
import string
import time

# Function for Cleansing and obtaining the bag of feature words

def Cleanse(tweet_data):
    cleanse_t=time.time() 
    bowl=[]
    print 'Cleansing Started!'
    for i in range(len(tweet_data)):
        f = tweet_data[i,1].lower()
       
        inter_links = [word for word in f.split() if (word.startswith('http://') or word.startswith('www.'))]
        for j in inter_links:
            f = f.replace(j,' ')    
        
        at_user= [word for word in f.split() if (word.startswith('@'))]
        for u in at_user:
            f=f.replace(u,'')
        
        f=f.translate(None,string.punctuation)
        f=" ".join(f.split())
        
        
        for k in f.split():
            if k not in stopwords:
                bowl.append(k)
                d_cleanse[i,1]= k
        
    seen = set()
    result = []
    for item in bowl:
        if item not in seen:
            seen.add(item)
            result.append(item)

    print 'Cleansing Complete! '
    print "Total time taken to cleanse the data is", time.time() - cleanse_t, "seconds!"
    print " "
    return result

load_t=time.time()

# Impoting the twitter feed with their sentiment and other features. 

df = pd.read_csv("training.1600000.processed.noemoticon.csv",usecols=[5,0],header = None)

# Processing of the data to obtain tweets and sentiments.

d = np.array(df)
d[d==0] = -1
d[d==4] = 1
d_cleanse=d

print "The Tweets and Sentiments are put in the desired form!"
print ""
print "Total time taken to load the data is", time.time() - load_t, "seconds!"
print ""


# Importing the Stop words. 

with open('stopwords.txt') as m:
    stopwords = m.readlines()
stopwords = [x.strip() for x in stopwords]

print "The Stop words are loaded!"
print ""

# Total bag of feature words

total_bag = Cleanse(d)

print "Converted all letters into lowercase in the tweets."
print ""
print "Converted all occurences of 'www.' or 'https://' to URL."
print ""
print "Removed additional white spaces."
print ""
print "Removed all punctuation."
print ""
print "Replaced duplicate words"
print ""
print "Also, obtained all the feature words!!"
print ""
print "The total bag of words have"
print len(total_bag)
print "words"

