from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import requires_csrf_token
from django.shortcuts import render
from .forms import CaseForm
import pandas as pd
import os
import numpy as np
import tensorflow 
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
import cv2
import math
#Classification imports

import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
np.random.seed(123)
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from keras.applications.xception import preprocess_input

from numpy.random import seed
seed(101)
import tensorflow as tf
tf.compat.v1.set_random_seed(101)
import pandas as pd
import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
import time
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
H = 256
W = 256

def index(request):
    return render(request, 'pages/index.html')

def about(request):
    return render(request, 'pages/about.html')

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)

def read_classification_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (299, 299))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = tf.expand_dims(x, 0) # Create a batch
    return x   

def save_results(ori_x, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

@requires_csrf_token
def segmentation(request):
    start_time = time.time()
    print()
    if request.method == 'POST':        
        form = CaseForm(data=request.POST ,files=request.FILES)
        
        if form.is_valid():
            form.save()
            file = request.FILES['image']

            img_path = os.path.join('./media/photos',file.name)
        

            path = 'my_model_16oct.h5'
   
            model = tf.keras.models.load_model(path)
        

            ori_x, x = read_image(img_path)


            """ Predicting the mask """
            y_pred = model.predict(x)[0] > 0.5
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred.astype(np.int32)

            """ Saving the predicted mask """
            save_mask_path = f"project/static/images/predections/{file.name}"
            save_results(ori_x, y_pred, save_mask_path)
            
            mask_path = f"images/predections/{file.name}"
            context = {'form':CaseForm(),'mask_path':mask_path}
            # Calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Elapsed time:',elapsed_time)
            return render(request, 'pages/predection.html',context)
        
        else:
            return render(request, 'pages/segmentation.html',{'form':CaseForm()})
    else:
        return render(request, 'pages/segmentation.html',{'form':CaseForm()})

def present_segmentation_classification(file):
    img_path = os.path.join('./media/photos',file.name)
    
    path = 'my_model_16oct.h5'

    model = tf.keras.models.load_model(path)
    
        

    ori_x, x = read_image(img_path)


    """ Predicting the mask """
    y_pred = model.predict(x)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)

    """ Saving the predicted mask """
    save_mask_path = f"project/static/images/predections/{file.name}"
    save_results(ori_x, y_pred, save_mask_path)
            
    mask_path = f"images/predections/{file.name}"
    return mask_path

@requires_csrf_token
def classification(request):
        custom_objects = {'top_2_accuracy': top_2_accuracy,'top_3_accuracy': top_3_accuracy}
        if request.method == 'POST':        
            form = CaseForm(data=request.POST ,files=request.FILES)
            if form.is_valid():
                form.save()
                file = request.FILES['image']

                img_path = './media/photos/'+file.name
                
                img = image.load_img(img_path, target_size=(299, 299)) 
                # Assuming Xception's default input size is 299x299

# Convert the image to a numpy array and preprocess it
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                path = 'xception.h5'
                model = tf.keras.models.load_model(path ,custom_objects=custom_objects)
        

                #x = read_classification_image(img_path)

                """   Classes """
                classes=[
                    'Actinic keratoses',
                    'Basal cell carcinoma',
                    'Benign keratosis-like lesions ',
                    'Dermatofibroma',
                     'Melanoma',
                    'Melanocytic nevi',
                    'Squamous cell carcinoma',
                    'None of the others',
                     'Vascular lesions'
                ]
                """ Predicting  """
                predictions = model.predict(img_array)
                # Get the top three predicted classes and their probabilities
                top_three_classes_indices = np.argsort(predictions[0])[-3:][::-1]
                top_three_probabilities = predictions[0][top_three_classes_indices]

                top3_list =[0,0,0]
                top3_prob =[0,0,0]
                # Print the top three predicted classes and their probabilities
                for i in range(3):
                    class_index = top_three_classes_indices[i] 
                    class_probability = top_three_probabilities[i]*100
                    top3_list[i] = class_index 
                    top3_prob[i] =  format(class_probability , '.6f')
                
                class1 = classes[top3_list[0]]
                class2 =  classes[top3_list[1]]
                class3 = classes[top3_list[2]]
                
                context = {'form':CaseForm(),'classification1':class1,'classification2':class2,
                           'classification3':class3,'Prob1':top3_prob[0],'Prob2':top3_prob[1],
                           'Prob3':top3_prob[2]}
                #class1 =='Melanoma' or class2 == 'Melanoma' or class3 == 'Melanoma'
                if(class1 =='Melanoma' or class2 == 'Melanoma' or class3 == 'Melanoma'):
                    maskPath = present_segmentation_classification(file)
                    context['mask_path'] = maskPath
                    return render(request, 'pages/classification_segmentation_result.html',context)
                else:    
                    return render(request, 'pages/classification_result.html',context)
        
            else:
                return render(request, 'pages/classification.html',{'form':CaseForm()})
        else:
            return render(request, 'pages/classification.html',{'form':CaseForm()})


