from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing
import math

from retail import getcenter
retailpriceserieslucknow = getcenter('LUCKNOW')

def get_anomalies(path):
  anomalies = pd.read_csv(path, header=None, index_col=None)
  anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
  anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
  return anomalies

anomalieslucknow = get_anomalies('data/anomaly/lucknow.csv')
lucknowlabels = []

# Labelling 
# Transport:  1
# Weather:  2
# Inflation:  3
# Hoarding:   4
# Navratra: 5

def get_labels(anomalies):
  global lucknowlabels
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
      lucknowlabels.append(-1)
  return

get_labels(anomalieslucknow)
print lucknowlabels
# lucknowlabels = [2,1,1,2,2,2,5,4,3,1,5,5,5,3,2,2,5,5,4,3,4,5,4,2,5,5,5,5,2,2,3,2,2,5,3,2,5,2]

def display_anomalies(anomalieslist, anomaly, labels):
  count = {'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,'07':0,'08':0,'09':0,'10':0,'11':0,'12':0}
  for i in range(0,len(anomalieslist)):
    if( labels[i] == anomaly):
      count[anomalieslist[0][i][5:7]] = count[anomalieslist[0][i][5:7]] + 1
  return count


def overlapping(anomalies,s,e,labels):
  startdate = datetime.strptime(s,'%Y-%m-%d')
  # if(startdate.month >=2 and startdate.month <= 7):
  #   return True
  for i in range(0,len(anomalies)):
    if(labels[i] == 2 or labels[i] == 5 or labels[i] == 3):  
      if((anomalies[0][i]<=s and s<=anomalies[1][i]) or  (anomalies[0][i]<=e and e<=anomalies[1][i])):
        return True
  return False

def findnormal_restricted(anomalies,series,labels):
  sdate = []
  edate = []
  date = CONSTANTS['STARTDATE']
  enddate = CONSTANTS['ENDDATE']
  from datetime import timedelta
  date = datetime.strptime(date,'%Y-%m-%d')+timedelta(days=21)
  enddate = datetime.strptime(enddate,'%Y-%m-%d')
  window = 42
  duration = timedelta(days=window)
  count  = 0
  while(duration <= enddate-date):
    s = datetime.strftime(date,'%Y-%m-%d')
    e = datetime.strftime(date+timedelta(days=window),'%Y-%m-%d') 
    x1 = (series.rolling(window=14,center=True).mean())[s:e]
    # x1 = series[s:e]
    date = date+timedelta(days=5)
    if not overlapping(anomalies,s,e,labels):
      a = x1.min()
      b = x1.max()
      if(math.isnan(a) == False and math.isnan(b) == False and b-a > 0 and b-a <300):
        sdate.append(s)
        edate.append(e)
        print b-a
        count = count + 1
        date = date+timedelta(days=40)
  print count
  return sdate,edate


def createnormalfile(path,anomaliesmumbai,retailpriceseriesmumbai,labels):
  a,b = findnormal_restricted(anomaliesmumbai,retailpriceseriesmumbai,labels)
  newdf = anomaliesmumbai
  newdf[1] = ' '+newdf[1]
  for i in range(len(a)):
    newdf.loc[i+len(anomaliesmumbai)] = [a[i],' '+b[i],' Normal_train']
  result = newdf.sort_values([0])
  result.to_csv(path, header=None,index=None)

createnormalfile('data/anomaly/normal_h_w_lucknow.csv',anomalieslucknow,retailpriceserieslucknow,lucknowlabels)
