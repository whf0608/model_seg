import streamlit as st
from glob import glob
import numpy as np 
import cv2
import importlib

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    
run_commed = st.sidebar.button('运行')

option = st.sidebar.selectbox(
        "select model",
        glob('*_st.py'),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

uploaded_file = st.file_uploader("Choose a file")

        
col11, col12, col13 , col14= st.columns(4)
with col11:
    emptycol11 = st.empty()
with col12:
    emptycol12 = st.empty()
with col13:
    emptycol13 = st.empty()
with col14:
    emptycol14 = st.empty()
col21, col22, col23, col24 = st.columns(4)

with col21:
    emptycol21 = st.empty()
with col22:
    emptycol22 = st.empty()
with col23:
    emptycol23 = st.empty()
with col24:
    emptycol24 = st.empty()
    
col31, col32, col33, col34 = st.columns(4)
with col31:
    emptycol31 = st.empty()
with col32:
    emptycol32 = st.empty()
with col33:
    emptycol33 = st.empty()
with col34:
    emptycol34 = st.empty()    
    
col41, col42, col43 , col44= st.columns(4)
with col41:
    emptycol42 = st.empty()
with col42:
    emptycol42 = st.empty()
with col43:
    emptycol43 = st.empty()

with col44:
    emptycol44 = st.empty()
    
cols=[emptycol34,emptycol33,emptycol32,emptycol31,emptycol24,emptycol23,emptycol22,emptycol21,emptycol14,emptycol13,emptycol12,emptycol11]

cache={}
def show(tag, img):
    if tag not in cache.keys():
        cache[tag] = cols.pop()
    cache[tag].image(img,caption=tag)

if uploaded_file is not None:
    # To read file as bytes:
    name = uploaded_file.name
    bytes_data = uploaded_file.getvalue()
    img = np.frombuffer(bytes_data,np.uint8)
    img = cv2.imdecode(img, 1)
    # st.image(img, caption='Sunrise by the mountains')
    cols[0].image(img)
    
    
if run_commed:
    model = importlib.import_module(option[:-3])
    model.train(show)
    
    
