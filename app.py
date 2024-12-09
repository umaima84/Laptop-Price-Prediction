import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# import the model
pipe = pickle.load(open('laptop_model.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',data['Company'].unique())

# type of laptop
product = st.selectbox('Product',data['Product'].unique())

# screen size
screen_size = st.selectbox('Screensize (Inches)', [13.3, 15.6, 15.4, 14.0 , 12.0 , 11.6, 17.3, 10.1, 13.5, 12.5, 13.0 ,18.4, 13.9, 12.3, 17.0 , 15.0 , 14.1, 11.3])

#cpu_frequency
cpu_frequency = st.selectbox('CPU Frequency', [2.3 , 1.8 , 2.5 , 2.7 , 3.1 , 3.  , 2.2 , 1.6 , 2.  , 2.8 , 1.2 , 2.9 , 2.4 , 1.44, 1.5 , 1.9 , 1.1 , 1.3 , 2.6 , 3.6 , 3.2 , 1.  ,2.1 , 0.9 , 1.92])

# Ram
ram = st.selectbox('RAM (GB)',[2,4,6,8,12,16,24,32,64])

#cpu
gpu = st.selectbox('GPU',data['GPU_Company'].unique())

#OS
os = st.selectbox('OS',data['OpSys'].unique())

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['Yes', "No"])

# IPS
ips = st.selectbox('IPS',['Yes', "No"])

# resolution
x_resolution = st.selectbox("X_Resolution", data["X_Resolution"].unique())
y_resolution = st.selectbox("Y_Resolution", data["Y_Resolution"].unique())

#cpu
cpu = st.selectbox('CPU',data['CPU'].unique())

#ssd
ssd = st.selectbox('SSD (GB)',[0,8,128,256,512,1024])

#hdd
hdd = st.selectbox('HDD (GB)',[0,128,256,500,1024,2048])

if st.button('Predict Price'):
    # query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    
    query = np.array([company, product, screen_size, cpu_frequency, ram, gpu, os, weight, touchscreen, ips, x_resolution, y_resolution, cpu, ssd, hdd])

    query = query.reshape(1,15)
    prediction = pipe.predict(query)[0]
    st.success(f"The predicted price of the laptop is: ${prediction:.2f}")

    



