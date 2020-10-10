import streamlit as st
from Image_Classification import *
import time
from PIL import Image, ImageOps
import base64
from tensorflow.keras.models import load_model
from keras import models
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

classifier = load_model('model1.h5')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")
st.title("Image Classification of Benign and Malignant skin cancer ")
st.subheader("We are going to predict whether uploaded image of Melanoma is benign or Malignant")
st.write("Created July'14 2020 by Arun Ramji Shanmugam")
st.write("________________________")
st.write("Click to seen an example or upload an image to see if it is a **Benign** or **Malignant** type of Melanoma")

example_button = st.button("See an example")
example_image = "Lentigo_maligna.jpeg"
if example_button:
    st.image(example_image, caption='Example Image', use_column_width=True)
    st.write("")
    #st.write("Classifying...")
    with open(example_image, "rb") as image:
      f = image.read()
      b = bytearray(f)
      uploaded_file = b[0]

    st.write("")
    st.write("Hi Doctor , Below are the sample images of how it looked like in some of my neural network layer...")
#     #Let's visualise all activation in the network
#     from keras.models import load_model
#     classifier = load_model('model1.h5')

    from keras.preprocessing import image
    import numpy as np

#   st.write(type(uploaded_file))
    import io
    test_image = Image.open(uploaded_file)
    test_image = test_image.convert('RGB')
    test_image = test_image.resize((150,150), Image.NEAREST)
    img_tensor = image.img_to_array(test_image)
    img_tensor = np.expand_dims(img_tensor,axis=0) #adding bias variable
    img_tensor /= 255.
#   st.write(img_tensor.shape)


    layer_outputs = [layer.output for layer in classifier.layers[:8]] #Extract the output of top 4 layer
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor) #Running model on predict mode

    layer_names = []
    for layer in classifier.layers[:4]:
        layer_names.append(layer.name) #Name of the layer , so that we can plot
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): #Displays the feature map
        n_features = layer_activation.shape[-1] #number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row #tiles the actiavtion channel in the matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                                  row * size : (row + 1) * size] = channel_image
            scale =1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            st.pyplot()

    #st.write(type(uploaded_file))
    #st.write(uploaded_file)
    st.write("Based on my analysis , Below is the result. keep in mind I am just 70% expert(accurate) now !!")
    st.write("Classifying...")
    label = machine_classification(uploaded_file,'model1.h5')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    if label[0] == 0:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> benign</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
        #st.write("With the probability of",label[1])
    else:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> Malignant</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
        #st.write("With the probability of",label[1])





    st.write("______________________________________")
    st.write(" ")
    st.write("Disclaimer : What ever the prediction made by our App is purely for educational and training purpose!!")


uploaded_file = st.file_uploader("Choose an Image ...", type=("jpg","png","jpeg"))
if uploaded_file is not None:
    #img  = Image.open(uploaded_file)
    #img  = base64.b64encode(uploaded_file.getvalue())
    #uploaded_file = uploaded_file.get_values()
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    #st.write("Classifying...")
    uploaded_file = uploaded_file.read()
    st.write("")
    st.write("Hi Doctor , Below are the sample images of how it looked like in some of my neural network layer...")

#     #Let's visualise all activation in the network
#     from keras.models import load_model
#     classifier = load_model('model1.h5')

    from keras.preprocessing import image
    import numpy as np

#   st.write(type(uploaded_file))
    import io
    test_image = Image.open(io.BytesIO(uploaded_file))
    test_image = test_image.convert('RGB')
    test_image = test_image.resize((150,150), Image.NEAREST)
    img_tensor = image.img_to_array(test_image)
    img_tensor = np.expand_dims(img_tensor,axis=0) #adding bias variable
    img_tensor /= 255.
#   st.write(img_tensor.shape)


    layer_outputs = [layer.output for layer in classifier.layers[:8]] #Extract the output of top 4 layer
    activation_model = Model(inputs=classifier.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor) #Running model on predict mode

    layer_names = []
    for layer in classifier.layers[:4]:
        layer_names.append(layer.name) #Name of the layer , so that we can plot
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): #Displays the feature map
        n_features = layer_activation.shape[-1] #number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row #tiles the actiavtion channel in the matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                                  row * size : (row + 1) * size] = channel_image
            scale =1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            st.pyplot()

    #st.write(type(uploaded_file))
    #st.write(uploaded_file)
    st.write("Based on my analysis , Below is the result. keep in mind I am just 70% expert(accurate) now !!")
    st.write("Classifying...")
    label = machine_classification(uploaded_file,'model1.h5')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    if label[0] == 0:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> benign</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
        #st.write("With the probability of",label[1])
    else:
        st.subheader('RESULT :')
        t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> Malignant</span> </span> melanoma!</div>"
        st.markdown(t, unsafe_allow_html=True)
        #st.write("With the probability of",label[1])





    st.write("______________________________________")
    st.write(" ")
    st.write("Disclaimer : What ever the prediction made by our App is purely for educational and training purpose!!")
