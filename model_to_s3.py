import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import sys
df = pd.read_csv('./CC.csv')

no_frauds = len(df[df['Class'] == 1])
non_fraud_indices = df[df.Class == 0].index
non_fraud_indices = df[df.Class == 0].index
random_indices = np.random.choice(non_fraud_indices, no_frauds, replace=False)
fraud_indices = df[df.Class == 1].index
under_sample_indices = np.concatenate([fraud_indices,random_indices])
under_sample = df.loc[under_sample_indices]
X_under = under_sample[['V3','V4','V5','V7','V8','V10','V11','V12','V14','V17','V19','V20','V21','Amount']]
y_under = under_sample['Class']
rfc = RandomForestClassifier(criterion= 'entropy', max_depth= 4, max_features= 'auto', n_estimators= 200)
rfc.fit(X_under,y_under)
filename = 'finalized_model.sav'
pickle.dump(rfc, open(filename, 'wb'))

from boto.s3.connection import S3Connection
from boto.s3.key import Key

aws_id = sys.argv[1]
aws_secret = sys.argv[2]

#Creating S3 Connection To upload the pickel file
conn = S3Connection(aws_id, aws_secret)
# Connecting to specified bucket
b = conn.get_bucket('teamsrkcc')
#Initializing Key
k = Key(b)
i = 'finalized_model.sav'
k.key = i
k.set_contents_from_filename(i)
k.set_acl('public-read')





