import streamlit as st
from Image_Classification import *
import time
#import PIL.Image as Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")
st.title("Image Classification of Benign and Malignant skin cancer ")
st.header("We are going predict whether uploaded image of Melanoma is benign or Malignant")
st.write("Created on July'14 2020")
st.write("@Author: Arun Ramji Shanmugam")
st.write("________________________")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write("Upload an image to see if it is Benign or Malignant type of Melanoma")


uploaded_file = st.file_uploader("Choose an Image ...", type="jpg")
if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file)
    #uploaded_file = uploaded_file.get_values()
    st.image(0x7F6074075790, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write(type(uploaded_file))
    st.write(uploaded_file)
    label = machine_classification(uploaded_file,'model1.h5')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    if label == 0:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> benign</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
    else:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> Malignant</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
        
     
  
    
    
    st.write("______________________________________")
    st.write(" ")
    st.write("Disclaimer : What ever the prediction made by our App is purely for educational and training purpose!!")   


