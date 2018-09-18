from sklearn import tree, svm, linear_model
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import pandas_datareader.data as web

start = dt.datetime(2015, 1, 1)
end = dt.datetime(2016, 12, 31)

ticker = 'RBC'

# Loading Data
df = web.DataReader(ticker, 'iex', start, end)
df = df.reset_index()

closing_price = []
highest_price = []
lowest_price = []
opening_price = []
volume_traded = []
dates = []

labels = []

for i in df[['close']]:
    for j in df[i]:
        closing_price.append(round(j, 2))

for i in df[['high']]:
    for j in df[i]:
        highest_price.append(round(j, 2))

for i in df[['low']]:
    for j in df[i]:
        opening_price.append(round(j, 2))

for i in df[['open']]:
    for j in df[i]:
        lowest_price.append(round(j, 2))

for i in df[['volume']]:
    for j in df[i]:
        volume_traded.append(round(j/100000,4))

for i in df[['date']]:
    for j in df[i]:
        dates.append(j)

lag = 10

'''_______________________ALGORITHIM____________________'''

def algorithim(high,low,open,close,volume):

    features = []
    labels = []

    for i in range(len(high) - lag):

        if close[i] > close[i-1]:
            labels.append(1)
        else:
            labels.append(0)

        temp_h = high[i]
        temp_l = low[i]
        temp_o = open[i]
        temp_c = close[i]
        temp_v = volume[i]

        features.append([temp_h,temp_l,temp_o,temp_c,temp_v])

    model = svm.SVC()
    model = model.fit(features,labels)

    prediction = []

    for i in range(lag):

        temp_h = high[-i]
        temp_l = low[-i]
        temp_o = open[-i]
        temp_c = close[-i]
        temp_v = volume[-i]

        prediction.append([temp_h,temp_l,temp_o,temp_c,temp_v])

    BuySell = model.predict(prediction)[0]

    if BuySell == 1:
        return 1
    else:
        return 0

Cash = 100
Purchased = False
HistoricCash = []
day = 0
buy_or_sell = 0
# To not allow old trends in prices to affect more recent trends, window only limits us to look at ceratin periods
window = 100
count = 0
window_count = 0




'''____________________BUY SELL LOOP_________________'''

while day < len(closing_price):

    day += 1
    window_count += 1
    StockPrice = closing_price[day-1]

    if window_count == window:
        window_count = 0
        count += 80

    if Purchased==True:
        Cash = round(StockPrice*Cash/ closing_price[day-2],2)
        colour = 'green'
    else:
        colour = 'red'

    HistoricCash.append(Cash)

    if day > lag+3:

        high = highest_price[count:day]
        low = lowest_price[count:day]
        close = closing_price[count:day]
        open = opening_price[count:day]
        volume = volume_traded[count:day]

        buy_or_sell = algorithim(high,low,open,close,volume)
    if buy_or_sell == 1:
        if Purchased == False:
            Purchased = True
    else:
        if Purchased == True:
            Purchased = False

    plt.plot(dates[day - 2:day], closing_price[day - 2:day], color=colour)


print("Final Balance With Using ML Trading: " + str(HistoricCash[-1]))
print("Buy and Hold: " + str(round(HistoricCash[0] * closing_price[-1] / closing_price[0], 2)))
print("Performance: " + str(round(100 * HistoricCash[-1] * closing_price[0] / (closing_price[-1] * HistoricCash[0]), 2)) + "%")

plt.plot(dates, HistoricCash, color='purple')
plt.show()





