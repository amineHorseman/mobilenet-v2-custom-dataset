
import glob
import cv2
import numpy as np
from keras.utils import to_categorical

def generate_batches(path, batchSize):
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
