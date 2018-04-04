from sklearn.metrics import f1_score
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import os
cwd = os.getcwd()

# lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]

def whiten(series):
  import scipy
  EigenValues, EigenVectors = np.linalg.eig(series.cov())
  D = [[0.0 for i in range(0, len(EigenValues))] for j in range(0, len(EigenValues))]
  for i in range(0, len(EigenValues)):
    D[i][i] = EigenValues[i]
  DInverse = np.linalg.matrix_power(D, -1)
  DInverseSqRoot = scipy.linalg.sqrtm(D)
  V = np.dot(np.dot(EigenVectors, DInverseSqRoot), EigenVectors.T)
  series = series.apply(lambda row: np.dot(V, row.T).T, axis=1)
  return series

def whiten_series_list(list):
	for i in range(0,len(list)):
		mean = list[i].mean()
		list[i] -= mean
	temp = pd.DataFrame()
	for i in range(0,len(list)):
		temp[i] = list[i]
	temp = whiten(temp)
	newlist = [temp[i] for i in range(0,len(list))]
	return newlist

from retail import getcenter
retailpriceserieslucknow = getcenter('LUCKNOW')

from mandi import getmandi
from mandi import mandipriceseries
from mandi import mandiarrivalseries 
mandipriceserieslucknow = getmandi('Lucknow',True)
mandiarrivalserieslucknow = getmandi('Lucknow',False)


def Normalise(arr):
  m = arr.mean()
  am = arr.min()
  aM = arr.max()
  arr -= m
  if(am == aM):
  	arr /= aM
  else:
  	arr /= (aM - am)
  return arr

def adjust_anomaly_window(anomalies,series):
	for i in range(0,len(anomalies)):
		anomaly_period = series[anomalies[0][i]:anomalies[1][i]]
		mid_date_index = anomaly_period[10:31].argmax()
		anomalies[0][i] = mid_date_index - timedelta(days=21)
		anomalies[1][i] = mid_date_index + timedelta(days=21)
		anomalies[0][i] = datetime.strftime(anomalies[0][i],'%Y-%m-%d')
		anomalies[1][i] = datetime.strftime(anomalies[1][i],'%Y-%m-%d')
	return anomalies

def get_anomalies(path,series):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%Y-%m-%d'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %Y-%m-%d'),'%Y-%m-%d') for date in anomalies[1]]
	anomalies = adjust_anomaly_window(anomalies,series)
	return anomalies

def get_anomalies_year(anomalies):
	mid_date_labels=[]
	for i in range(0,len(anomalies[0])):
		mid_date_labels.append(datetime.strftime(datetime.strptime(anomalies[0][i],'%Y-%m-%d')+timedelta(days=21),'%Y-%m-%d'))
	return mid_date_labels

# def newlabels(anomalies,oldlabels):
#   labels = []
#   k=0
#   for i in range(0,len(anomalies)):
#     if(anomalies[2][i] != ' Normal_train'):
#       labels.append(oldlabels[k])
#       #print k,oldlabels[k]
#       k = k+1
#     else:
#       labels.append(8)
#   return labels

def newlabels(anomalies):
  lucknowlabels = []
  for i in range(len(anomalies)):
    somestring = anomalies[2][i].strip().lower()
    if(somestring == 'transport' or somestring == 'fuel' or somestring == 'fuel hike' or somestring == 'strike'): #transport and fuel are related to each other
      lucknowlabels.append(1)
    elif(somestring == 'weather'):
      lucknowlabels.append(2)
    elif(somestring == 'inflation'):
      lucknowlabels.append(3)
    elif(somestring == 'hoarding'):
      lucknowlabels.append(4)
    elif(somestring == 'navratra'):
      lucknowlabels.append(5)
    else:
      lucknowlabels.append(8) #this is for the periods when tag is Normal_train
  return lucknowlabels

def prepare(anomalies,labels,priceserieslist):
	x = []
	for i in range(0,len(anomalies)):		
		p=[]
		for j in range(0,len(priceserieslist)):
			start = anomalies[0][i]
			end = datetime.strptime(anomalies[0][i],'%Y-%m-%d') + timedelta(days = 34)
			end = datetime.strftime(end,'%Y-%m-%d')
			# print priceserieslist[j][start:end]
			# p += (Normalise(np.array(priceserieslist[j][start:end].tolist()))).tolist()
			p += ((np.array(priceserieslist[j][start:anomalies[1][i]].tolist()))).tolist()
		x.append(np.array(p))
	return np.array(x),np.array(labels)		

def getKey(item):
	return item[0]

def partition(xseries,yseries,year,months):
	# min_month = datetime.strptime(min(year),'%Y-%m-%d')
	# max_month = datetime.strptime(max(year),'%Y-%m-%d')
	combined_series = zip(year,xseries,yseries)
	combined_series = sorted(combined_series,key=getKey)
	train = []
	train_labels = []
	fixed = datetime.strptime('2006-01-01','%Y-%m-%d')
	i=0
	while(fixed < datetime.strptime('2017-11-01','%Y-%m-%d')):
		currx=[]
		curry=[]
		for anomaly in combined_series:
			i += 1
			if(datetime.strptime(anomaly[0],'%Y-%m-%d') > fixed and datetime.strptime(anomaly[0],'%Y-%m-%d')- fixed <= timedelta(days=months*30)):
				currx.append(anomaly[1])
				curry.append(anomaly[2])
		train.append(currx)
		train_labels.append(curry)
		fixed = fixed +timedelta(days = months*30)
	
	return np.array(train),np.array(train_labels)

def get_score(xtrain,xtest,ytrain,ytest):
	scaler = preprocessing.StandardScaler().fit(xtrain)
	xtrain = scaler.transform(xtrain)
	xtest = scaler.transform(xtest)
	model = RandomForestClassifier(max_depth=2, random_state=0)
	# model = SVC(kernel='rbf', C=0.8)
	model.fit(xtrain,ytrain)
	test_pred = np.array(model.predict(xtest))
	# ytest = np.array(ytest)
	# if(test_pred[0] == ytest[0]):
	# 	return 1
	# else:
	# 	return 0
	return test_pred

def train_test_function(align_l, data_l):
	# align = [1,2,3]
	# anomaliesmumbai = get_anomalies('data/anomaly/normal_h_w_mumbai.csv',align_m)
	# anomaliesdelhi = get_anomalies('data/anomaly/normal_h_w_delhi.csv',align_d)
	anomalieslucknow = get_anomalies('data/anomaly/normal_h_w_lucknow.csv',align_l)
	# delhilabelsnew = newlabels(anomaliesdelhi,delhilabels)
	lucknowlabelsnew = newlabels(anomalieslucknow)
	# mumbailabelsnew = newlabels(anomaliesmumbai,mumbailabels)
	# print "Delhi --------------------------"
	# x1,y1 = prepare(anomaliesdelhi,delhilabelsnew,data_d)
	# print "Mumbai --------------------------"

	# x2,y2 = prepare(anomaliesmumbai,mumbailabelsnew,data_m)
	# print "Lucknow --------------------------"

	x3,y3 = prepare(anomalieslucknow,lucknowlabelsnew,data_l)
	# temp = 0
	# for y in y1:
	# 	if(y == 2 or y==3 or y==5):
	# 		temp = temp + 1 
	# print temp

	# temp = 0
	# for y in y2:
	# 	if(y == 2 or y==3 or y==5):
	# 		temp = temp + 1 
	# print temp
	
	# temp = 0
	# for y in y3:
	# 	if(y == 2 or y==3 or y==5):
	# 		temp = temp + 1 
	# print temp
	
	lucknow_anomalies_year = get_anomalies_year(anomalieslucknow)
	xall = np.array(x3.tolist())
	yall = np.array(y3.tolist())
	xall_new =[]
	yall_new = []
	yearall_new = []
	yearall = np.array(lucknow_anomalies_year)
	for y in range(0,len(yall)):
		if( yall[y] == 2 or yall[y]== 4):
			xall_new.append(xall[y])
			yall_new.append(yall[y])
			yearall_new.append(yearall[y])
		elif(yall[y] == 3):
			xall_new.append(xall[y])
			yall_new.append(0)
			yearall_new.append(yearall[y])

	assert(len(xall_new) == len(yearall_new))
	# print len(xall_new)
	total_data, total_labels = partition(xall_new,yall_new,yearall_new,6)
	predicted = []
	actual_labels = []
	# for temp in range(0,len(total_data)):
	# 	print len(total_data[temp])
	for i in range(0,len(total_data)):
		if( len(total_data[i]) != 0):
			test_split = total_data[i]
			test_labels = total_labels[i]
			actual_labels = actual_labels + test_labels
			train_split = []
			train_labels_split = []
			for j in range(0,len(total_data)):
				if( j != i):
					train_split = train_split + total_data[j]
					train_labels_split = train_labels_split+total_labels[j]
			# print i, train_split
			pred_test = get_score(train_split,test_split,train_labels_split,test_labels)	
			predicted = predicted + pred_test.tolist()
	predicted = np.array(predicted)
	actual_labels= np.array(actual_labels)
	# print len(actual_labels)
	print (sum(predicted == actual_labels) * 100.0)/len(predicted)
	from sklearn.metrics import confusion_matrix
	print confusion_matrix(actual_labels,predicted)

	# train_data = []
	# train_labels = []
	# actual_train_labels = []
	# for i in range(0,len(total_data)):
	# 	if(len(total_data[i])!= 0):
	# 		train_data = train_data + total_data[i]
	# 		train_labels = train_labels + total_labels[i]
	# 		actual_train_labels = actual_train_labels + total_labels[i]
	# pred_test = get_score(train_data,train_data,train_labels,train_labels)	
	# print sum(pred_test == actual_train_labels)/124.0

	# print actual_labels
	# print predicted
	# print f1_score(actual_labels,predicted,labels=[2,3,5],average="macro")


train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow])
train_test_function(mandipriceserieslucknow,[mandipriceserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow,mandipriceserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow-mandipriceserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
train_test_function(mandipriceserieslucknow,[retailpriceserieslucknow/mandipriceserieslucknow])


# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandipriceseriesmumbai,mandipriceseriesdelhi,mandipriceserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])


# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(retailpriceseriesmumbai-mandipriceseriesmumbai,retailpriceseriesdelhi-mandipriceseriesdelhi,retailpriceserieslucknow-mandipriceserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])


#Change the argmax to idxmin for running the part below

# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai],[retailpriceseriesdelhi],[retailpriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[mandipriceseriesmumbai],[mandipriceseriesdelhi],[mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai-mandipriceseriesmumbai],[retailpriceseriesdelhi-mandipriceseriesdelhi],[retailpriceserieslucknow-mandipriceserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai,mandipriceseriesmumbai,mandiarrivalseriesmumbai],[retailpriceseriesdelhi,mandipriceseriesdelhi,mandiarrivalseriesdelhi],[retailpriceserieslucknow,mandipriceserieslucknow,mandiarrivalserieslucknow])
# train_test_function(mandiarrivalseriesmumbai,mandiarrivalseriesdelhi,mandiarrivalserieslucknow,[retailpriceseriesmumbai/mandipriceseriesmumbai],[retailpriceseriesdelhi/mandipriceseriesdelhi],[retailpriceserieslucknow/mandipriceserieslucknow])

