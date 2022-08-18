import pandas as pd
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nifty50 contains ticker name and url suffix of the stock
nifty50 = pd.read_csv("nifty50.csv")
urlsymbol=""
outfile=""
# making sure valid input is taken in all times
while True:
  try:
    inp = (input("Enter ticker symbol: "))
    for i in nifty50.index:
        if (nifty50['TICKER'][i]==inp):
            urlsymbol=nifty50['Company Name'][i]
            outfile=nifty50['TICKER'][i]
    if urlsymbol!="":
      print(" Ticker entered successfully...")
      break;
    else:
      print("wrong input enter again ..")      
  except ValueError:
    print("Provide a valid ticker symbol...")
    continue

news_tables = {}
url = 'https://in.investing.com/equities/{urlsymbol}'.format(urlsymbol=urlsymbol)
req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
resp = urlopen(req)  
html = BeautifulSoup(resp, features="lxml")
news_table = html.find(class_="common-articles-list")

df=news_table.findAll('h3')
news_list=[]
count=0
for link in df:
    news_list.append(link.get("title"))
    count+=1

df1=news_table.findAll('time')
news_time=[]
j=1
for time in df1:
    if j%2==0:
        news_time.append(time.text)
        j=j+1
        continue
    else:
        j=j+1
        continue

class my_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value
  
# Main Function
dict_obj = my_dictionary()
i=1
for i in range(count):
    dict_obj.add(news_list[i], news_time[i])

data=pd.DataFrame(dict_obj.items(), columns=['News','Date'])
for i in data.index:
    data['ticker']=inp

# to change hours

data['Date'] = data['Date'].str.replace("hours ago","")

def newhours(x):
    d = datetime.today() - timedelta(hours=x)
    return d.strftime('%b %d, %Y %R')

for i in data.index:
    if len(data['Date'][i]) == 3 or len(data['Date'][i]) == 2 :
        x=int(data['Date'][i])
        y=newhours(x)
        data['Date'][i]=y

# to change minutes

data['Date'] = data['Date'].str.replace("minutes ago","")

def newminutes(x):
    d = datetime.today() - timedelta(minutes=x)
    return d.strftime('%b %d, %Y %R')

for i in data.index:
    if len(data['Date'][i]) == (3 or 2):
        x=int(data['Date'][i])
        y=newminutes(x)
        data['Date'][i]=y

data['Date']=pd.to_datetime(data.Date).dt.date
vader = SentimentIntensityAnalyzer()
scores = data['News'].apply(vader.polarity_scores).tolist()
scores_df = pd.DataFrame(scores)
data = data.join(scores_df, rsuffix='_right')
mean_scores=data.groupby(['Date']).mean()

mean_scores.to_csv("{outfile}.csv".format(outfile=outfile))