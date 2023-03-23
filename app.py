
# Libraries that are used.
import streamlit as st
from PIL import Image
import datetime
import math
import numpy as np
import pandas as pd 
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
# from yfinance import shared



# setting the page configuration of streamlit web app
st.set_page_config(
     page_title="Stock Visualization and forecasting",
     page_icon="🧊",
     layout="wide",
 )


col1,col2 = st.columns([3,1])
col1.title('Stock Visualization and Forecasting')

image= Image.open('./images/stocks.jpg')
new_img = image.resize((100,100))
col2.image(new_img)

#taking the input for the visualization
col1,col2,col3 = st.columns([2,2,2])
sd = col1.date_input(
    "Start Date",value= datetime.date(2012,1,1),max_value=datetime.datetime.now())

ed = col2.date_input(
    "End Date")

new_sd = sd.strftime("%Y-%m-%d")
new_ed = ed.strftime("%Y-%m-%d")

# stock_symbol =col3.text_input('Enter the stock name','GOOGL')
ticker_list = pd.read_csv('./TICKER_LIST.txt')
stock_symbol = col3.selectbox('Stock ticker', ticker_list) # Select ticker symbol
stock_symbol = stock_symbol.upper()


# stock function which actually handles all the part

def stock_visualization():

    # downloading the data of the selected ticker
    df = yf.download(tickers=stock_symbol, start=new_sd, end=new_ed)    
    c1 = yf.Ticker(stock_symbol)


    # THESE ARE NOT WORKING AFTER SOMETIME WHEN THE PROJECT BEEN MADE.

    # st.write(c1.info)
    # st.write(c1.info['longName'])
    # string_logo = '<img src= %s>' %c1.info['logo_url']
    # st.markdown(string_logo,unsafe_allow_html=True)
    # st.write(c1.info['longBusinessSummary'])

    st.subheader('1year Graph Trend')
    data =c1.history(period="12mo")
    st.line_chart(data.values)

    st.subheader('Descricption of stock')
    st.write(df.describe())

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(14,6))
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time chart with 100 MA')
    max100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(14,6))
    plt.plot(max100)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(df.Close)
    plt.legend(['rolling mean','actual'],loc='lower right')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 200 MA')
    ma200 = df.Close.rolling(200).mean()
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(14,6))
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD ($)',fontsize=18)
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close,'b')
    plt.legend(['MA 100','MA 200','actual'])
    st.pyplot(fig)


    # importing the model and scaling the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range =(0,1))

    model = load_model('keras_model.h5')

    data = pd.DataFrame(df.filter(['Close']))

    #convert the dataframe to numpy array
    dataset = data.values

    scaled_data = scaler.fit_transform(data)

    #get the number of rows to train the model on 
    training_data_len = math.ceil( len(dataset) * .7)
    

    test_data = scaled_data[training_data_len-60:,:]

    #create the data set x_test and y_test
    x_test=[]
    y_test = dataset[training_data_len:,:]  #unscaled 
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    # model predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)


    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    predictions = predictions*scale_factor
    y_test = y_test * scale_factor

    # final graph
    st.subheader('Prediction vs Original')
    fig2 = plt.figure(figsize=(14,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(predictions,'r',label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

if st.button('Visualize'):
    if new_sd == datetime.datetime.now().strftime("%Y-%m-%d"):
        st.write("Please select correct starting date")
    
    else:
        stock_visualization()
        
