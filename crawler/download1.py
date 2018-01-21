import time
import datetime
import os
import csv
from selenium import webdriver
#from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from datetime import timedelta, date

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)


def givedates(start_dt,end_dt):
	return [dt.strftime("%Y-%m-%d") for dt in daterange(start_dt, end_dt) ]

def monthtodays(month,year):
	if month in ["November","April","June","September"]:
		return 30
	elif month in ["February"]:
		if year%4 ==0:
			return 29
		else:
			return 28
	else:
		return 31	

months1 = ["January","February","March","April","May","June","July","August","September","October","November","December"]
months2 = ["January","February","March","April","May","June","July","August","September","October","November"]
months3 = ["June","July","August","September","October","November","December"]

def month_name_to_number(number):
    if number == "January":
        return 1
    elif number == "February":
        return 2
    elif number == "March":
        return 3
    elif number == "April":
        return 4
    elif number == "May":
        return 5
    elif number == "June":
        return 6
    elif number == "July":
        return 7
    elif number == "August":
        return 8
    elif number == "September":
        return 9
    elif number == "October":
        return 10
    elif number == "November":
        return 11
    elif number == "December":
        return 12

centernames = ["Maharashtra"]

def extractdata():

	path_to_chromedriver = 'C:\selenium' 
	browser = webdriver.Chrome()
	url = 'http://nhb.gov.in/OnlineClient/MonthlyPriceAndArrivalReport.aspx'
	print "1"
	myfile= open('mynewdata.csv','a')
	for center in centernames:
		start_year = 2015
		end_year = 2017
		for year in range(start_year,end_year+1):
			months = months1
			if(year == 2017):
				months = months2
			elif(year == 2015):
				months = months3		
			for month in months:
				print year,month
				browser.get(url)
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlyear\"]/option[contains(text(),\""+str(year)+"\")]").click()
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlmonth\"]/option[contains(text(),\""+month+"\")]").click()
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_drpCategoryName\"]/option[contains(text(),\""+"VEGETABLES"+"\")]").click()
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_drpCropName\"]/option[contains(text(),\""+"ONION"+"\")]").click()
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_ddlvariety\"]/option[contains(text(),\""+"ONION"+"\")]").click()
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_LsboxCenterList\"]/option[contains(text(),\""+"MUMBAI"+"\")]").click()
				browser.implicitly_wait(30)
				browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_btnSearch\"]").click()
				table = browser.find_element_by_xpath("//*[@id=\"ctl00_ContentPlaceHolder1_GridViewmonthlypriceandarrivalreport\"]")
				rows = table.find_elements_by_tag_name("tr")
				monthnum = month_name_to_number(month)
				start_dt = date(year,monthnum,1)
				end_dt = date(year, monthnum, monthtodays(month,year))
				dates = givedates(start_dt,end_dt)
				myfile= open('mynewretaildata.csv','a')
				cells = rows[1].find_elements_by_xpath(".//*[local-name(.)='td']")
				temp = [cell.text for cell in cells]
				for i in range(0,len(dates)):
					temp1 = temp[i+4].split()
					if temp1 != []:
						mystr=dates[i]+','+temp1[3]+'\n'
					else:
						mystr=dates[i]+',0'+'\n'
					myfile.write(mystr)
				myfile.close()


if __name__ == '__main__':
	extractdata()
	

'''
<span id="ctl00_ContentPlaceHolder1_GridViewmonthlypriceandarrivalreport_ctl02_lblresult" 
style="display:inline-block;width:98%;">
3000<br>4800<br>4000<br>5000<br>1596<br>
</span>

'''