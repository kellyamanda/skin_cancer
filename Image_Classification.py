import keras
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model



def machine_classification(img, weights_file):
    #weights_file = '/Users/arunramji/Downloads/cats_and_dogs_small_1.h5'
    model = load_model(weights_file)
    from keras.preprocessing import image
    import io
    test_image = Image.open(img)
    test_image = test_image.convert('RGB')
    test_image = test_image.resize((150,150), Image.NEAREST)
    test_image = image.img_to_array(test_image)
    #test_image = image.load_img(img,target_size = (150,150))
    #test_image = image.img_to_array(test_image) #making it 3D array as input layer is 3D     
    test_image = np.expand_dims(test_image,axis=0) #adding bias variable

    #%timeit  
    result = model.predict(test_image)

    #print(train_generator.class_indices) #to check the value of outplut class assigned

    if result[0] <= 0.5:
        p = result[0] * 100
        return 1,p
    else:
        p = result[0] * 100  
        return 0,p
    
  
