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

# Function for performing ADAGRAD

def adagrad(feature,senti,w_ada,N):
    G=np.zeros(len(total_bag))
    G_tot=np.ones(len(total_bag))
    for j in range(feature.shape[0]):
        u=np.dot(w_ada.T,feature[j])
        cond=(senti[j]*u)
        Sum = 0
        if cond<1:
            Sum=Sum+(senti[j]*feature[j])
        grad_ada=lambu*feature[j]-(N/len(senti))*(Sum)
    G_tot=G_tot+np.square(grad_ada)
    G=np.sqrt(G_tot)
    Ginv=(1./G)
    wt_ada=w_ada-N*np.multiply(Ginv,grad_ada)
    w_ada=min(1,((1/sqrt(lambu))/LA.norm(np.multiply(G,wt_ada))))*wt_ada
    return w_ada


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

# Training using ADAGRAD

condi=np.zeros(batch)

grad_ada=np.zeros(len(total_bag))

w_ada=np.zeros([len(total_bag)])
wt_ada=np.zeros([len(total_bag)])


lambu = 0.001


Err_v = []

W_store2 = []

print "ADAGRAD IS STARTING!"

start_tada=time.time()
for v in range(1,2001):
    
    R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    R=random.sample(d_cleanse,batch)
    R=np.asarray(R)
    feature,senti=getF(R)
    N=1/(lambu*v)
    #N = 1
    w_ada=adagrad(feature,senti,w_ada,N)
    w_ada.shape
    S = []
    P = np.zeros([batch,len(total_bag)])
    
    print v 
  
    print "Computing accuracy for ", v ,"iteration"
    R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    R=random.sample(d,batch)
    R=np.asarray(R)
    feature,S=getF(R)
    count_ada=0
    for i in range(batch):
        score_ada = np.dot(w_ada.T,feature[i])
        if score_ada>=0:
            predY = 1
        else:
            predY = -1
        if predY!=S[i]:
            count_ada+=1
    
    err_per = count_ada 
    print "Percentage error with training data=", err_per
    
    print ""
    print "Done for", v , "iteration"
    
    Err_v.append(err_per)
    
    
    W_store2.append(w_ada)
        
print "Total time taken is", time.time() - start_tada, "seconds!"

import matplotlib.pyplot as plt
'''plt.title('PEGAOS')
plt.plot(Err_iter)
plt.xlabel('Jaccard similarity')
plt.ylabel('No.of users')'''


New_error = np.asarray(Err_v)
y = smooth(New_error, window_len=20, window = 'hanning')
plt.figure(1)
plt.title('ADAGRAD-Traning Data')
plt.grid(True)
plt.plot(y)
plt.xlabel('Number of Iterations')
plt.ylabel('Error %')



# Testing using ADAGRAD


df2 = pd.read_csv("testdata.manual.2009.06.14.csv",usecols=[5,0],header = None)
test = np.array(df2)
test_cleanse=test
test[test==0] = -1
test[test==4] = 1
error_list=[]
test[test==2]=1
test_cleanse=test_Cleanse(test)
#test_bag=Cleanse(test)
for i in range(len(W_store2)):    
    test_R=np.zeros([batch,len(total_bag)])
    feature = np.zeros([batch,len(total_bag)])
    test_R=random.sample(test,batch)
    test_R=np.asarray(test_R)
    test_feature,Senti=getF(test_R) 
    
    count_ada = 0
    
    for i in range(batch):
        score_grad = np.dot(W_store2[i].T,test_feature[i])
        #print score
        if score_grad>=0:
            predY = 1
        else:
            predY = -1
        if predY!=Senti[i]:
            count_ada+=1
    error_list.append(count_ada)
    print "\nPercentage ERROR with testing data for ADAGRAD=",count_ada
    

New_error2 = np.asarray(error_list)
y = smooth(New_error2, window_len=20, window = 'hanning')
plt.figure(2)
plt.title('ADAGRAD-Testing Data')
plt.grid(True)
plt.plot(y)
plt.xlabel('Number of Iterations')
plt.ylabel('Error %')
    