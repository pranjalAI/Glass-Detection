from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re, glob, os,cv2
import numpy as np
import pandas as pd
import glass_detection
from shutil import copyfile
import shutil
from distutils.dir_util import copy_tree

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

for f in os.listdir("D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"):
    os.remove("D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"+f)
# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
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
        similar_glass_details=glass_detection.getUrl(file_path)
        glass_details=similar_glass_details[0]
        suggested_url=similar_glass_details[1]
        parentclass=similar_glass_details[2]
        childclass=similar_glass_details[3]

        print("Checking for similar images.......")
        #getting similar images
        test_image = cv2.imread(file_path) 
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) 
        histogram_test = cv2.calcHist([gray_image], [0],  
                                 None, [256], [0, 256])

        hist_dict={}
        for image in os.listdir("D:\\My ML Projects\\lenskart task\\train\\"+str(parentclass)+"\\"):
            try:
                img_to_compare = cv2.imread("D:\\My ML Projects\\lenskart task\\train\\"+str(parentclass)+"\\"+image) 
                img_to_compare = cv2.cvtColor(img_to_compare, cv2.COLOR_BGR2GRAY) 
                img_to_compare_hist = cv2.calcHist([img_to_compare], [0],  
                                         None, [256], [0, 256])
                c=0
                i = 0
                while i<len(histogram_test) and i<len(img_to_compare_hist): 
                    c+=(histogram_test[i]-img_to_compare_hist[i])**2
                    i+= 1
                c = c**(1 / 2)
                hist_dict[image]=c[0]
            except:
                print(image)
        sort_dict = sorted(hist_dict.items(), key=lambda x: x[1], reverse=False)[1:11]
        similar_images=[]
        for i in sort_dict:
            similar_images.append("D:\\My ML Projects\\lenskart task\\train\\"+str(parentclass)+"\\"+str(i[0]))

        for f in os.listdir("D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"):
            os.remove("D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"+f)

        for count, image in enumerate(similar_images):
            dst=f"D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"
            #copyfile(image, dst)
            shutil.copy(image, dst, follow_symlinks=True)

        for count, filename in enumerate(os.listdir("D:\\My ML Projects\\Lenskart Final\\static\\similar_images")): 
            dst ="Glasses" + str(count+1) + ".jpg"
            src ="D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"+ filename 
            dst ="D:\\My ML Projects\\Lenskart Final\\static\\similar_images\\"+ dst 
            os.rename(src, dst) 

        print(glass_details)
        return glass_details
        #return suggested_url
        #return render_template("index.html",foobar=similar_images)
    return None


if __name__ == '__main__':
    app.run(debug=False)

