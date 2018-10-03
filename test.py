from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model
import numpy as np

# other imports
import json
import datetime
import time
import glob
import os

# load the user configs
with open('conf.json') as f:    
  config = json.load(f)

# config variables
test_path     = config["test_path"]
model_path    = config["model_path"]
model_name    = "save_model_stage1.h5"

print ("Loading model...")
filename = model_path + "/" + model_name
model = load_model(filename)

print ("Testing images...")
folders = [name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]
for folder in folders:
    #print("- class: {}".format(folder))
    success = 0
    average_confidence = 0
    files = glob.glob(test_path + "/" + folder + "/*.jpg")
    for file in files:
        img = image.load_img(file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y_prob = model.predict(x)
        y_class = y_prob.argmax(axis=-1)
        y_class = y_class[0]
        y_confidence = int(y_prob[0][y_class] * 100)
        #print("predicted label: {} (prob = {})".format(y_class, y_confidence))
        if y_class == int(folder):
            success += 1
        average_confidence += y_confidence
    success = int(100*success/len(files))
    average_confidence = int(average_confidence / len(files))
    print("class '{}': success rate = {}% with {}% avg confidence".format(folder, success, average_confidence))
