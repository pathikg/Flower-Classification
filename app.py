from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import cv2

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Defining a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './model.h5'

# Loading trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

print(model.optimizer.get_config())
# params = dict([(k, v) for k, v in model.optimizer.get_config().items()])
# config = {
#     'class_name': optimizer.__class__.__name__,
#      'config': params,
# }


def model_predict(img_path, model):
    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    index = np.argmax(res)
    classes = ["Sunflower", "Rose", "Daisy", "Dandelion", "Tulip"]
    if os.path.exists(img_path):
        os.remove(img_path)
    return classes[index]

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
        filedirpath = os.path.join(basepath, 'uploads')
        if not os.path.isdir(filedirpath):
            os.mkdir(filedirpath)
        file_path = os.path.join(
            filedirpath, secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None

# if __name__ == '__main__':
#     app.run(debug=True)