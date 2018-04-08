# Libraries used
import pandas as pd
import numpy as np
import time

start_t=time.time()

# Impoting the twitter feed with their sentiment and other features. 

df = pd.read_csv("training.1600000.processed.noemoticon.csv",usecols=[5,0],header = None)


# Processing of the data to obtain tweets and sentiments.

d = np.array(df)
d[d==0] = -1
d[d==4] = 1
tweet_data=d

print "The Tweets and Sentiments are put in the desired form!"
print ""
print "Total time taken to load the data is", time.time() - start_t, "seconds!"