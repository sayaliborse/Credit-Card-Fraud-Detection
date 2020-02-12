from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from  sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import boto3
import io
from io import StringIO
import urllib.request
import sys
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


def timeModification(df):
    df['day'] = (df['Time'] // 86400) + 1
    df['period'] = df ['Time'] // 21600
    df['period'] = (df['period'] % 4) + 1
    return df

def extractday(df,day):
    if(day == "1"):
        data = df[df['day']==1]
        return data
    elif(day == "2"):
        data = df[df['day']==2]
        return data
    else:
        return df

def extractperiod(df,period):
    if(period == "1"):
        data = df[df['period']==1]
        return data
    elif(period == "2"):
        data = df[df['period']==2]
        return data
    elif(period == "3"):
        data = df[df['period']==3]
        return data
    elif (period == '4'):
        data = df[df['period']==4]
        return data
    else:
        return df
def getDatafromS3():
	session = boto3.Session(profile_name ='default')
	client = boto3.client('s3', aws_access_key_id=aws_id,
    aws_secret_access_key=aws_secret)
	bucket_name = 'teamsrkcc'
	object_key = 'CC.csv'
	model_name = 'finalized_model.sav'
	csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
	body = csv_obj['Body']
	csv_string = body.read().decode('utf-8')
	df = pd.read_csv(StringIO(csv_string))
	urllib.request.urlretrieve("https://s3.amazonaws.com/teamsrkcc/finalized_model.sav", filename= 'finalized_model.sav')
	model = pickle.load(open("finalized_model.sav", "rb"))
	return df,model
	
@app.route('/')
def hello():
	return render_template('home.html')
	
@app.route('/result', methods = ['POST']) 
def result():
	dataset,model= getDatafromS3(k1,k2)
	timeModification(dataset)
	day = str(request.form.get('Day'))
	period = str(request.form.get('Period'))
	data = extractday(dataset,day)
	df = extractperiod(data,period)
	df.drop(['Unnamed: 0','Time'], inplace = True, axis=1)
	no_frauds = len(df[df['Class'] == 1])
	non_fraud_indices = df[df.Class == 0].index
	non_fraud_indices = df[df.Class == 0].index
	random_indices = np.random.choice(non_fraud_indices, no_frauds, replace=False)
	fraud_indices = df[df.Class == 1].index
	under_sample_indices = np.concatenate([fraud_indices,random_indices])
	under_sample = df.loc[under_sample_indices]
	X = under_sample[['V3','V4','V5','V7','V8','V10','V11','V12','V14','V17','V19','V20','V21','Amount']]
	Y = under_sample['Class'].values
	my_prediction = model.predict(X)
	#result_values = []
	#result_values.append(cm[0][0])
	#result_values.append((cm[0][0]+cm[0][1]))
	#result_values.append(cm[1][1])
	#result_values.append((cm[1][0] + cm[1][1]))
	f1 = f1_score(Y, my_prediction)
	p = precision_score(Y, my_prediction)
	r = recall_score(Y, my_prediction)
	a = accuracy_score(Y, my_prediction)
	
	if request.method == 'POST':
		return render_template('result.html', predicted_one = (my_prediction==1).sum() , actual_one = (Y==1).sum(), predicted_zero = (my_prediction==0).sum(),\
								actual_zero = (Y==0).sum(),\
								f1Score = f1, pScore = p, recallScore = r , accuracy = a)

if __name__ == "__main__":
	app.run(debug=True)