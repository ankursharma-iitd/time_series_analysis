from datetime import datetime
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from os import listdir
import datetime as datetime

#Process Wholesale Data

code = -1
def process_wholesale():
    files = listdir('wholesaleData')
    #print files
    #print(files)
    newfile = open('wholesale_processed.csv', 'w')
    #files = [files[0]]
    for file in files:
        with open('wholesaleData/'+file) as f:
            content = f.readlines()

        for i in range(1,len(content)):
            temp = content[i].strip().split(',')
            #print temp
            if(len(temp) > 8):
                temp[0:2] = [''.join(temp[0:2])]
            #print temp
            mandi = temp[0]
            #print mandi
            date = temp[1]
            #print 1,mandi
            if date != '':
                date = datetime.datetime.strptime(date,'%d/%m/%Y').strftime('%Y-%m-%d')
                #print date
                arrival = temp[2]
                variety = temp[3]
                minp = temp[4]
                maxp = temp[5]
                modalp = temp[6]
                if mandi != '':
                    if mandi == 'Lucknow':
                        code = 87
                    else:
                        code = -1
                if code == 87:
                    mystr = date+','+str(1)+','+arrival+',NR,'+variety+','+minp+','+maxp+','+modalp+'\n'
                    newfile.write(mystr)

process_wholesale()

def load_data(csvfile):
    