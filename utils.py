
import glob
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os

def generate_batches(path, batchSize):
    while True:
        files = glob.glob(path + '/*/*jpg')
        for f in range(0, len(files), batchSize):
            x = []
            y = []
            for i in range(f, f+batchSize):
                if i < len(files):  
                    img = cv2.imread(files[i])
                    x.append(cv2.resize(img, (224, 224)))
                    y.append(int(files[i].split('/')[1]))
            yield (np.array(x), to_categorical(y, num_classes=10))

def generate_batches_with_augmentation(train_path, batch_size, validation_split, augmented_data):
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=0.3,
            zoom_range=0.1,
            validation_split=validation_split)
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(224, 224),
            batch_size=batch_size, 
            save_to_dir=augmented_data)
        return train_generator

def create_folders(model_path, augmented_data):
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(augmented_data):
        os.mkdir(augmented_data)
    if not os.path.exists("logs"):
        os.mkdir("logs")
        