# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Aman Chauhan
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)

dog = ['Chihuahua', 'Japanese_Spaniel',
       'Maltese_Dog', 'Pekinese',
       'Shih-Tzu', 'Blenheim_Spaniel',
       'Papillon', 'Toy_Terrier',
       'Rhodesian_Ridgeback', 'Afghan_Hound',
       'Basset', 'Beagle', 'Bloodhound',
       'Bluetick', 'Black-And-Tan_Coonhound',
       'Walker_Hound', 'English_Foxhound',
       'Redbone', 'Borzoi',
       'Irish_Wolfhound', 'Italian_Greyhound',
       'Whippet', 'Ibizan_Hound',
       'Norwegian_Elkhound', 'Otterhound',
       'Saluki', 'Scottish_Deerhound',
       'Weimaraner', 'Staffordshire_Bullterrier',
       'American_Staffordshire_Terrier',
       'Bedlington_Terrier', 'Border_Terrier',
       'Kerry_Blue_Terrier', 'Irish_Terrier',
       'Norfolk_Terrier', 'Norwich_Terrier',
       'Yorkshire_Terrier', 'Wire-Haired_Fox_Terrier',
       'Lakeland_Terrier', 'Sealyham_Terrier',
       'Airedale', 'Cairn',
       'Australian_Terrier', 'Dandie_Dinmont',
       'Boston_Bull', 'Miniature_Schnauzer',
       'Giant_Schnauzer', 'Standard_Schnauzer',
       'Scotch_Terrier', 'Tibetan_Terrier',
       'Silky_Terrier', 'Soft-Coated_Wheaten_Terrier',
       'West_Highland_White_Terrier', 'Lhasa',
       'Flat-Coated_Retriever',
       'Curly-Coated_Retriever', 'Golden_Retriever',
       'Labrador_Retriever',
       'Chesapeake_Bay_Retriever',
       'German_Short-Haired_Pointer', 'Vizsla',
       'English_Setter', 'Irish_Setter',
       'Gordon_Setter', 'Brittany_Spaniel',
       'Clumber', 'English_Springer',
       'Welsh_Springer_Spaniel', 'Cocker_Spaniel',
       'Sussex_Spaniel', 'Irish_Water_Spaniel',
       'Kuvasz', 'Schipperke',
       'Groenendael', 'Malinois', 'Briard',
       'Kelpie', 'Komondor',
       'Old_English_Sheepdog', 'Shetland_Sheepdog',
       'Collie', 'Border_Collie',
       'Bouvier_Des_Flandres', 'Rottweiler',
       'German_Shepherd', 'Doberman',
       'Miniature_Pinscher',
       'Greater_Swiss_Mountain_Dog',
       'Bernese_Mountain_Dog', 'Appenzeller',
       'Entlebucher', 'Boxer',
       'Bull_Mastiff', 'Tibetan_Mastiff',
       'French_Bulldog', 'Great_Dane',
       'Saint_Bernard', 'Eskimo_Dog',
       'Malamute', 'Siberian_Husky',
       'Affenpinscher', 'Basenji', 'Pug',
       'Leonberg', 'Newfoundland',
       'Great_Pyrenees', 'Samoyed',
       'Pomeranian', 'Chow', 'Keeshond',
       'Brabancon_Griffon', 'Pembroke',
       'Cardigan', 'Toy_Poodle',
       'Miniature_Poodle', 'Standard_Poodle',
       'Mexican_Hairless', 'Dingo', 'Dhole',
       'African_Hunting_Dog']


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)

    preds=np.argmax(preds, axis=1)

    preds = dog[int(preds)]
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
