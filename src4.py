# Libraries used
import pandas as pd
import numpy as np
import string
import time
import random
from nltk.probability import FreqDist
from math import sqrt
from numpy import linalg as LA

# Function for Cleansing and obtaining the bag of feature words
def test_Cleanse(tweet_data):
    #start_t=time.time() 
    #bowl=[]
    print 'Cleansing test data! '
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
                #bowl.append(k)
                #d_cleanse[i,1]= k
                test_cleanse[i,1]=k
    print "Data Cleansed"
    return test_cleanse

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


# Function for obtaining the feature matrix

def getF(d_batch): 
    featmat_t=time.time()
    Q=np.zeros([batch,len(total_bag)])
    senti=np.zeros(batch)
    for r in range(d_batch.shape[0]):     
        k = FreqDist(d_batch[r][1].split())
        k = dict(k)
        senti[r]=d_batch[r][0]
        for key in k:    
            #print key
            if key in total_bag:            
                Q[r,total_bag.index(key)] = k[key]
    print "Total time taken to create the feature matrix is", time.time() - featmat_t, "seconds!"
    print " "
    return Q,senti



def smooth(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



# Function for performing PEGASOS

def pegasos(feature,senti,w,N):
    #lambu=0.01
    #Sum=np.zeros(len(total_bag))
    for j in range(feature.shape[0]):
        u=np.dot(w.T,feature[j])
        cond=(senti[j]*u)
        Sum = 0
        if cond<1:
            Sum=Sum+(senti[j]*feature[j])
        grad=(lambu*w).T-((N/len(senti))*Sum)
        wt=w.T-grad
        w=min(1,(1/sqrt(lambu))/LA.norm(wt))*wt
        #print w
        w=w.T
    return w




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

# Total bag of feature words.

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
print ""

print "The batch size of the feature matrix is"
batch=80
print batch 
print "" 


R=random.sample(d,batch)
R=np.asarray(R)

# Feature Matrix and the correspoding Sentiment vector.

feature,S=getF(R)
print "The Feature matrix is obtained!"
print ""
print feature

# Training using PEGASOS

condi=np.zeros(batch)
t=[]
w=np.zeros(len(total_bag))
grad=np.zeros(len(total_bag))

wt=np.ones(len(total_bag))

lambu = 0.001

print "PEGASOS IS STARTING!"

Err_iter = []
W_store = []

pega_t=time.time()    

for v in range(1,2001):
    
    R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    R=random.sample(d_cleanse,batch)
    R=np.asarray(R)
    feature,senti=getF(R)
    N=1/(lambu*v)
    w=pegasos(feature,senti,w,N)
    print v

        
    S = []
    P = np.zeros([batch,len(total_bag)])
    
    print "Computing accuracy for ", v ,"iteration"
    
    R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    R=random.sample(d,batch)
    R=np.asarray(R)
    feature,S=getF(R)   
    count = 0
    
    for i in range(batch):
        score = np.dot(w.T,feature[i])
        if score>=0:
            predY = 1
        else:
            predY = -1
        if predY!=S[i]:
            count+=1
    
    err_per = count        
    print "Percentage error with training data=", count
    
    print ""
    print "Done for", v , "iteration"
        
        
    Err_iter.append(err_per)
    W_store.append(w)

print "Total time taken is", time.time() - pega_t, "seconds!"

import matplotlib.pyplot as plt


New_error = np.asarray(Err_iter)
y = smooth(New_error, window_len=20, window = 'hanning')
plt.figure(1)
plt.title('PEGAOS-Traning Data')
plt.grid(True)
plt.plot(y)
plt.xlabel('Number of Iterations')
plt.ylabel('Error %')


# Testing using Pegasos

df2 = pd.read_csv("testdata.manual.2009.06.14.csv",usecols=[5,0],header = None)
test = np.array(df2)
test_cleanse=test
test[test==0] = -1
test[test==4] = 1
test[test==2]=1
error_list_pega=[]
test_cleanse=test_Cleanse(test)
#test_bag=Cleanse(test)
for i in range(len(W_store)):    
    test_R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    test_R=random.sample(test,batch)
    test_R=np.asarray(test_R)
    test_feature,Senti=getF(test_R) 
    
    count_pega = 0
    
    for i in range(batch):
        score_grad = np.dot(W_store[i].T,test_feature[i])
        #print score
        if score_grad>=0:
            predY = 1
        else:
            predY = -1
        if predY!=Senti[i]:
            count_pega+=1
        #print "the number ",i
    error_list_pega.append(count_pega)
    print "\nPercentage ERROR with testing data for PEGASOS=",count_pega

New_error2 = np.asarray(error_list_pega)
y1 = smooth(New_error2, window_len=20, window = 'hanning')
plt.figure(2)
plt.title('PEGAOS-Testing Data')
plt.grid(True)
plt.plot(y1)
plt.xlabel('Number of Iterations')
plt.ylabel('Error %')





