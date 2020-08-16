# AI model to identify benign and Malignant Melanoma 

by @Arun Ramji Shanmugam - arunramji11@gmail.com

## Data sources :
1. https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery
2. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## Project Summary :
  *  An idea is to create an app with AI engine which would take the image of mole as an input and predict the probability of whether it is malignant or benign .
  *  Overall survival at 5 years depends on the thickness of the primary melanoma, whether the lymph nodes are involved, and whether there is spread of melanoma to             distant sites. Lymph nodes are small, bean-shaped organs that help fight infection. For people with "thin melanoma," defined as being less than 1 millimeter in maximal thickness, that has not spread to lymph nodes or other distant sites, the 5-year survival is 99%. 
   
   * hence sooner we figure out cancer in early stage, more probability of curing them , our prime objective is developing such tool that can identify it in early stage
   
## Libraries Used :
* Keras
* Numpy
* Scikit
* Matplotlib
### Processor 
* XLA_GPU on google colab
   
## Model Development 
1.  Classifcation - Binary
2.  Classes - Malignant(0) / Benign (1)
3.  Train/val/test :
    * Training set - 1329 Image beloning to 2 classes . Malignant : 699 , Benign : 629
    * Validation set - 200 Image beloning to 2 classes . Malignant : 200 , Benign : 200
    * Test set - 200 Image beloning to 2 classes . Malignant : 104 , Benign : 104 
4. Model Type : Basic CNN     
 Total params: 3,584,193
 Trainable params: 3,584,193
 Non-trainable params: 0

5. Early stopping : No
6. Drop Out : No
7. Regularisation : No

## Result :

*Validation Accuracy : 0.7225
*Test Accuracy : 0.6442307829856873



