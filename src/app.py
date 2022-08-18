import numpy as np
import warnings
import pandas as pd
import yfinance as yf
from urllib.request import urlopen,Request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from numpy.lib.function_base import append
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from flask import Flask,render_template,request, redirect, url_for
# from forms import ContactForm
from datetime import datetime, timedelta
from sklearn import preprocessing
nltk.download('vader_lexicon')

#import fundamental
app = Flask(__name__)             # create an app instance
app.static_folder = 'static'
app.secret_key = 'secretKey'

@app.route("/", methods=['GET', 'POST'])
def mainpage():                      
    form = None
    if request.method == 'POST':
        name =  request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")
        res = pd.DataFrame({'name':name, 'email':email, 'subject':subject ,'message':message}, index=[0])
        res.to_csv('contactusMessage.csv',mode='a', header=False, index=False)
    else:
        return render_template("index.html", form=form) 

@app.route('/predictnow', methods=['GET', 'POST'])
def predictnow():
    if request.method == 'POST':
        return redirect(url_for('mainpage'))
    return render_template('inner-page.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        ticker = request.form.get("ticker")
        stock = yf.Ticker(ticker+".NS")

        
        ###Prediction Start
        regressor = load_model(ticker+".h5")
        # Last 10 days prices
        dt = stock.history(interval="1d", period="10d")
        new_dt = dt.filter(['Close'])
        sc=MinMaxScaler()
        DataScaler = sc.fit(new_dt)
        last10Days = new_dt[-10:].values
        Last10Days = []
        ##Append the past 10 days
        Last10Days.append(last10Days)

        ##Converting the X_test_data into a numpy array
        Last10Days = np.array(Last10Days)
 
        # Normalizing the data just like we did for training the model
        Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))
 
        # Changing the shape of the data to 3D
        # Choosing TimeSteps as 10 because we have used the same for training
        NumSamples=1
        TimeSteps=10
        NumFeatures=1
        Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

        # Making predictions on data
        predicted_Price = regressor.predict(Last10Days)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        tom_price=predicted_Price[0][0]
        changefromtoda = round(tom_price-currentPrice, 3)
        changefromtodaype = round(((tom_price-currentPrice)/currentPrice)*100,3)
        changefromtoday=""
        changefromtodayper=""
        predsign=""
        if(changefromtoda>0):
            changefromtoday= "+"+str(changefromtoda)
            changefromtodayper="+"+str(changefromtodaype)
            predsign="+"
        else :
            changefromtoday=changefromtoda
            changefromtodayper=changefromtodaype
            predsign="-"
        #print(predicted_Price)
        ####Prediction End   
        
        changefromyesterda = round(currentPrice-previousclose, 3)
        changefromyesterdaype = round(((currentPrice-previousclose)/previousclose)*100,3)
        changefromyesterday=""
        changefromyesterdayper=""
        sign=""
        if(changefromyesterda>0):
            changefromyesterday= "+"+str(changefromyesterda)
            changefromyesterdayper="+"+str(changefromyesterdaype)
            sign="+"
        else :
            changefromyesterday=changefromyesterda
            changefromyesterdayper=changefromyesterdaype
            sign="-"
        req = request.form
        print(req)

        d = stock.history(period="2y",interval="1d")
        d.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
        d['Returns']=((d['Close']-d['Open'])/d['Open'])*100
        d.to_csv('Stock_data.csv')
        df = pd.read_csv('Stock_data.csv')
        df.sort_values('Date')
        arr = df.to_numpy()

        stockDate=[]
        closePrice=[]
        highPrice=[]
        lowPrice=[]
        openPrice=[]   
        dfnorm=d
        x = dfnorm.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        dfnorm = pd.DataFrame(x_scaled)
        dfnorm['Date']=df['Date']
        dfnorm.to_csv('Stock_datanorm.csv')
        df2 = pd.read_csv('Stock_datanorm.csv')
        df2.sort_values('Date')
        arr2 = df2.to_numpy()
        volume=[]
        Returns=[]
        i=0
        for row in arr:
            stockDate.append(arr[i][0])
            closePrice.append(arr[i][4])
            highPrice.append(arr[i][2])
            lowPrice.append(arr[i][3])
            openPrice.append(arr[i][1])
            i=i+1
        j=0
        for row in arr2:
            volume.append(arr2[j][4])
            Returns.append(arr2[j][5])
            j=j+1
        
        #Recommendation

        temp = stock.history(period="2y",interval="1d")
        temp.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
        temp['Returns']=((temp['Close']-temp['Open'])/temp['Open'])*100
        temp.to_csv('Recommendation.csv')
        stock_data = pd.read_csv('Recommendation.csv')
        stock_data.sort_values('Date')

        #SMA
        stock_data['SMA_25'] = stock_data['Close'].rolling(25).mean()
        stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()
        stock_data['Signal_SMA'] = np.where(stock_data['SMA_25'] > stock_data['SMA_50'], 1.0, 0.0)
        stock_data['Position_SMA'] = stock_data['Signal_SMA'].diff()
        stock_data = stock_data.dropna()
        #SMA-backtest
        buyAmt = 0
        sellAmt = 0
        return_perSMA = 0
        buyDates = np.array([])
        for i in range(stock_data.shape[0]):
            if stock_data.iloc[i, 10]==1:
                buyAmt = buyAmt + stock_data.iloc[i, 4]*100
                buyDates = np.append(buyDates, i)
        for i in range(stock_data.shape[0]):
            for j in buyDates:
                if i == j:
                    if (int(j)+60 < stock_data.shape[0]):
                        sellAmt = sellAmt + stock_data.iloc[int(j+60), 4]*100
                    else:
                        sellAmt = sellAmt + stock_data.iloc[int(j+10), 4]*100
        buyAmt=round(buyAmt,3)
        total_SMA = round(sellAmt - buyAmt,3)
        return_perSMA = round((sellAmt/buyAmt)*100,3)

        #EMA
        stock_data['EMA_25'] = stock_data['Close'].ewm(span= 25, adjust=False).mean()
        stock_data['EMA_50'] = stock_data['Close'].ewm(span= 50, adjust=False).mean()
        stock_data['Signal_EMA'] = np.where(stock_data['EMA_25'] > stock_data['EMA_50'], 1.0, 0.0)
        stock_data['Position_EMA'] = stock_data['Signal_EMA'].diff()
        #EMA-backtest
        emabuyAmt = 0
        emasellAmt = 0
        emabuyDates = np.array([])

        for i in range(stock_data.shape[0]):
            if stock_data.iloc[i, 14]==1:
                emabuyAmt = emabuyAmt + stock_data.iloc[i, 4]*100
                emabuyDates = np.append(emabuyDates, i)
                
        emabuyAmt=round(emabuyAmt,3)
        for i in range(stock_data.shape[0]):
            for j in emabuyDates:
                if i == j:
                    if (int(j)+60 < stock_data.shape[0]):
                        emasellAmt = emasellAmt + stock_data.iloc[int(j+60), 4]*100
                    else:
                        emasellAmt = emasellAmt + stock_data.iloc[int(j+10), 4]*100            
                    
        total_EMA = round((emasellAmt - emabuyAmt),3) 
        return_perEMA=round(((emasellAmt/emabuyAmt)*100),3)

        #BB
        stock_data['Rolling Mean'] = stock_data['Close'].rolling(20).mean()
        stock_data['Rolling Std'] = stock_data['Close'].rolling(20).std()
        stock_data['Bollinger High'] = stock_data['Rolling Mean'] + (stock_data['Rolling Std']*2) 
        stock_data['Bollinger Low'] = stock_data['Rolling Mean'] - (stock_data['Rolling Std']*2)
        stock_data = stock_data.dropna()
        stock_data['Signal_BB'] = None
        stock_data['Position_BB'] = None
        for row in range(len(stock_data)):
            if (stock_data['Close'].iloc[row] > stock_data['Bollinger High'].iloc[row]) and (stock_data['Close'].iloc[row-1] < stock_data['Bollinger High'].iloc[row-1]):stock_data['Signal_BB'].iloc[row] = 0
            if (stock_data['Close'].iloc[row] < stock_data['Bollinger Low'].iloc[row]) and (stock_data['Close'].iloc[row-1] > stock_data['Bollinger Low'].iloc[row-1]):stock_data['Signal_BB'].iloc[row] = 1  

        stock_data['Signal_BB'].fillna(method='ffill',inplace=True)
        stock_data['Position_BB'] = stock_data['Signal_BB'].diff()
        #BB-backtest
        bbbuyAmt = 0
        bbsellAmt = 0
        bbbuyDates = np.array([])

        for i in range(stock_data.shape[0]):
            if stock_data.iloc[i, 20] == 1:
                bbbuyAmt = bbbuyAmt + stock_data.iloc[i, 4]*100
                bbbuyDates = np.append(bbbuyDates, i)

        bbbuyAmt=round(bbbuyAmt,3)
        for i in range(stock_data.shape[0]):
            for j in bbbuyDates:
                if i == j:
                    if (int(j)+60 < stock_data.shape[0]):
                        bbsellAmt = bbsellAmt + stock_data.iloc[int(j+60), 4]*100
                    else:
                        bbsellAmt = bbsellAmt + stock_data.iloc[int(j), 4]*100

        total_BB = round((bbsellAmt - bbbuyAmt),3)
        return_perBB = round(((bbsellAmt/bbbuyAmt)*100),3)

        #MACD
        stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust= False).mean() - stock_data['Close'].ewm(span=26, adjust= False).mean()
        stock_data['Signal_9'] = stock_data['MACD'].ewm(span=9, adjust= False).mean()
        stock_data['Signal_MACD'] = np.where(stock_data.loc[:, 'MACD'] > stock_data.loc[:, 'Signal_9'], 1.0, 0.0)
        stock_data['Position_MACD'] = stock_data['Signal_MACD'].diff()
        #MACD-backtest
        cdbuyAmt = 0
        cdsellAmt = 0
        cdbuyDates = np.array([])
        for i in range(stock_data.shape[0]):
            if stock_data.iloc[i, 24]==1:
                cdbuyAmt = buyAmt + stock_data.iloc[i, 4]*100
                cdbuyDates = np.append(buyDates, i)
                
        cdbuyAmt=round(cdbuyAmt,3)
        for i in range(stock_data.shape[0]):
            for j in cdbuyDates:
                if i == j:
                    if (int(j)+60 < stock_data.shape[0]):
                        cdsellAmt = cdsellAmt + stock_data.iloc[int(j+60), 4]*100
                    else:
                        cdsellAmt = cdsellAmt + stock_data.iloc[int(j+8), 4]*100

        total_MACD = round((cdsellAmt - cdbuyAmt),3)
        return_perCD=round(((cdsellAmt/cdbuyAmt)*100),3)
        
        #RSI
        stock_data['Diff'] = stock_data['Close'].diff()
        stock_data['Gain'] = stock_data['Diff'][stock_data['Diff']>0]
        stock_data['Loss'] = (-1)*stock_data["Diff"][stock_data["Diff"]<0]
        stock_data = stock_data.fillna(0)
        stock_data['AvgGain'] = stock_data['Gain'].rolling(window=14).mean()
        stock_data['AvgLoss'] = stock_data['Loss'].rolling(window=14).mean()
        stock_data['RSI'] = 100 - (100/(1+ stock_data['AvgGain']/stock_data['AvgLoss']))
        stock_data['RSI70'] = np.where(stock_data['RSI'] > 70, 1, 0)
        stock_data['RSI30'] = np.where(stock_data['RSI'] < 30, -1, 0)
        stock_data['Position_R70'] = stock_data['RSI70'].diff()
        stock_data['Position_R30'] = stock_data['RSI30'].diff()

        stock_data = stock_data.dropna()
        #stock_data.to_csv('smacheck.csv')
        arr3=stock_data.to_numpy()
        
        buyx=[]
        buyy=[]
        sellx=[]
        selly=[]
        emabuyx=[]
        emabuyy=[]
        emasellx=[]
        emaselly=[]
        bbhigh=[]
        bblow=[]
        bbmean=[]
        bbbuyx=[]
        bbbuyy=[]
        bbsellx=[]
        bbselly=[]
        sig9=[]
        macd=[]
        cdbuyx=[]
        cdbuyy=[]
        cdsellx=[]
        cdselly=[]
        rsi=[]
        r70=[]
        r30=[]
        rsbuyx=[]
        rsbuyy=[]
        rssellx=[]
        rsselly=[]

        s=0
        for row in arr3:
            if arr3[s][10]==1:
               buyx.append(arr3[s][0])
               buyy.append(arr3[s][8])
            elif arr3[s][10]==-1:
               sellx.append(arr3[s][0])
               selly.append(arr3[s][8])
            s=s+1
        showsmabuydate=buyx[-1]
        showsmabuyprice=round(buyy[-1],3)
        showsmaselldate=sellx[-1]
        showsmasellprice=round(selly[-1],3)

        t=0
        for row in arr3:
            if arr3[t][14]==1:
                emabuyx.append(arr3[t][0])
                emabuyy.append(arr3[t][12])
            elif arr3[t][14]==-1:
               emasellx.append(arr3[t][0])
               emaselly.append(arr3[t][12])
            t=t+1
        showemabuydate=emabuyx[-1]
        showemabuyprice=round(emabuyy[-1],3)
        showemaselldate=emasellx[-1]
        showemasellprice=round(emaselly[-1],3)

        x=0
        for row in arr3:
            if arr3[x][20]==1:
                bbbuyx.append(arr3[x][0])
                bbbuyy.append(arr3[x][18])
            elif arr3[x][20]==-1:
                bbsellx.append(arr3[x][0])
                bbselly.append(arr3[x][17])
            x=x+1
        showbbbuydate=bbbuyx[-1]
        showbbbuyprice=round(bbbuyy[-1],3)
        showbbselldate=bbsellx[-1]
        showbbsellprice=round(bbselly[-1],3)
        
        y=0
        for row in arr3:
            if arr3[y][24]==1:
                cdbuyx.append(arr3[y][0])
                cdbuyy.append(arr3[y][4])
            elif arr3[y][24]==-1:
                cdsellx.append(arr3[y][0])
                cdselly.append(arr3[y][4])
            y=y+1
        showcdbuydate=cdbuyx[-1]
        showcdbuyprice=round(cdbuyy[-1],3)
        showcdselldate=cdsellx[-1]
        showcdsellprice=round(cdselly[-1],3)
        z=0
        for row in arr3:
            if arr3[z][33]==1:
                rsbuyx.append(arr3[z][0])
                rsbuyy.append(arr3[z][30])
            elif arr3[z][34]==-1:
                rssellx.append(arr3[z][0])
                rsselly.append(arr3[z][30])
            z=z+1
        showrsibuydate=rsbuyx[-1]
        showrsibuyprice=round(rsbuyy[-1],3)
        showrsiselldate=rssellx[-1]
        showrsisellprice=round(rsselly[-1],3)

        sma25=[]
        sma50=[]
        possma=[]
        sigsma=[]
        madate=[]
        maclose=[]
        ema25=[]
        ema50=[]

        k=0
        for row in arr3:
            madate.append(arr3[k][0])
            maclose.append(arr3[k][4])
            sma25.append(arr3[k][7])
            sma50.append(arr3[k][8])
            sigsma.append(arr3[k][9])
            possma.append(arr3[k][10])
            ema25.append(arr3[k][11])
            ema50.append(arr3[k][12])
            bbmean.append(arr3[k][15])
            bbhigh.append(arr3[k][17])
            bblow.append(arr3[k][18])
            sig9.append(arr3[k][22])
            macd.append(arr3[k][21])
            rsi.append(arr3[k][30])
            r70.append(70)
            r30.append(30)
            k=k+1
        
        ###Sentiment Analysis
        nifty50 = pd.read_csv("nifty50.csv")
        urlsymbol=""
        outfile=""
        for i in nifty50.index:
            if (nifty50['TICKER'][i]==ticker):
                urlsymbol=nifty50['Company Name'][i]
                outfile=nifty50['TICKER'][i]
        
        url = 'https://in.investing.com/equities/{urlsymbol}'.format(urlsymbol=urlsymbol)
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        resp = urlopen(req)  
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(class_="common-articles-list")

        pagenumbers=6
        for i in range(2,pagenumbers):
            page = i
            #newdict={}
            url = 'https://in.investing.com/equities/{urlsymbol}/{page}'.format(urlsymbol=urlsymbol,page=page)
            req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
            resp = urlopen(req) 
            html = BeautifulSoup(resp, features="lxml")
            newdict = html.find(class_="common-articles-list")
            news_table.append(newdict)

        df=news_table.findAll('h3')
        news_list=[]
        count=0
        for link in df:
            news_list.append(link.get("title"))
            count=count+1
        
        df1=news_table.findAll('time')
        news_time=[]
        j=1
        count1=0
        for time in df1:
            if j%2==0:
                news_time.append(time.text)
                count1=count1+1
                j=j+1
                continue
            else:
                j=j+1
                continue
        
        class my_dictionary(dict):
            def __init__(self):
                self = dict()
            def add(self, key, value):
                self[key] = value
        dict_obj = my_dictionary()
        i=1
        for i in range(count1):
            dict_obj.add(news_list[i], news_time[i])
        
        data=pd.DataFrame(dict_obj.items(), columns=['News','date'])
        data['date'] = data['date'].str.replace("hours ago","")

        def newhours(x):
            d = datetime.today() - timedelta(hours=x)
            return d.strftime('%b %d, %Y %R')

        for i in data.index:
            if len(data['date'][i]) == 3 or len(data['date'][i]) == 2 :
                x=int(data['date'][i])
                y=newhours(x)
                data['date'][i]=y

        data['date'] = data['date'].str.replace("minutes ago","")

        def newminutes(x):
            d = datetime.today() - timedelta(minutes=x)
            return d.strftime('%b %d, %Y %R')

        for i in data.index:
            if len(data['date'][i]) == (3 or 2):
                x=int(data['date'][i])
                y=newminutes(x)
                data['date'][i]=y

        data['date']=pd.to_datetime(data.date).dt.date
        vader = SentimentIntensityAnalyzer()
        scores = data['News'].apply(vader.polarity_scores).tolist()
        scores_df = pd.DataFrame(scores)
        data = data.join(scores_df, rsuffix='_right')
        data['Date'] = pd.to_datetime(data.date).dt.date
        mean_scores=data.groupby(['Date']).mean()
        sentival=mean_scores['compound'][-1]
        mean_scores.to_csv('tempsenti.csv')
        df7=pd.read_csv("tempsenti.csv")
        dateval=df7['Date'].iloc[-1]
        sentiscore=""
        if sentival > 0:
            sentiscore="Positive Sentiment"
        elif sentival < 0:
            sentiscore="Negative Sentiment"
        else:
            sentiscore="Neutral Sentiment"
        ###Sentiment End

        return render_template("prediction_1.html",
        ticker=ticker, value  = value, fullname=fullname, website=website, bookvalue=bookvalue,
        fiftytwochange=fiftytwochange,beta=beta,currentPrice=currentPrice, dayHigh=dayHigh,
        dayLow=dayLow, dividendRate = dividendRate, fiftyTwoWeekHigh=fiftyTwoWeekHigh,
        fiftyTwoWeekLow=fiftyTwoWeekLow, forwardEps=forwardEps, forwardPE=forwardPE,
        marketCap=marketCap, priceToBook=priceToBook,returnOnAssets=returnOnAssets,
        revenueGrowth=revenueGrowth, targetHighPrice=targetHighPrice, targetLowPrice=targetLowPrice,
        trailingEps=trailingEps, trailingPE=trailingPE, twoHundredDayAverage=twoHundredDayAverage,
        returnOnEquity=returnOnEquity, changefromyesterday=changefromyesterday,
        changefromyesterdayper=changefromyesterdayper,sign=sign,lowPrice=lowPrice,stockDate=stockDate,
        highPrice=highPrice,closePrice=closePrice,volume=volume,Returns=Returns,openPrice=openPrice,
        sma25=sma25,sma50=sma50,possma=possma,sigsma=sigsma,madate=madate,maclose=maclose,
        buyx=buyx,buyy=buyy,sellx=sellx,selly=selly,ema50=ema50,ema25=ema25,emabuyx=emabuyx,emabuyy=emabuyy,
        emasellx=emasellx,emaselly=emaselly,bbmean=bbmean,bbhigh=bbhigh,bblow=bblow,bbbuyx=bbbuyx,
        bbbuyy=bbbuyy,bbsellx=bbsellx,bbselly=bbselly,sig9=sig9,macd=macd,cdbuyx=cdbuyx,cdbuyy=cdbuyy,
        cdsellx=cdsellx,cdselly=cdselly,rsi=rsi,r30=r30,r70=r70,rsbuyx=rsbuyx,rsbuyy=rsbuyy,
        rssellx=rssellx,rsselly=rsselly,showsmabuydate=showsmabuydate,showsmabuyprice=showsmabuyprice,
        showsmaselldate=showsmaselldate,showsmasellprice=showsmasellprice,showemabuydate=showemabuydate,
        showemabuyprice=showemabuyprice,showemaselldate=showemaselldate,showemasellprice=showemasellprice,
        showcdbuydate=showcdbuydate,showcdbuyprice=showcdbuyprice,showcdselldate=showcdselldate,
        showcdsellprice=showcdsellprice,showbbbuydate=showbbbuydate,showbbbuyprice=showbbbuyprice,
        showbbselldate=showbbselldate,showbbsellprice=showbbsellprice,showrsibuydate=showrsibuydate,
        showrsibuyprice=showrsibuyprice,showrsiselldate=showrsiselldate,showrsisellprice=showrsisellprice,
        tom_price=tom_price,changefromtoday=changefromtoday,changefromtodayper=changefromtodayper,
        predsign=predsign,sentiscore=sentiscore,dateval=dateval,buyAmt=buyAmt,total_SMA=total_SMA,
        return_perSMA=return_perSMA,emabuyAmt=emabuyAmt,total_EMA=total_EMA,return_perEMA=return_perEMA,
        bbbuyAmt=bbbuyAmt,total_BB=total_BB,return_perBB=return_perBB,cdbuyAmt=cdbuyAmt,total_MACD=total_MACD,
        return_perCD=return_perCD)

if __name__ == "__main__":        # on running python app.py
    app.run()                     # run the flask app 