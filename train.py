# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.utils import to_categorical
from keras.optimizers import SGD

# other imports
import json
import datetime
import time

# load the user configs
with open('conf.json') as f:    
  config = json.load(f)

# config variables
weights     = config["weights"]
train_path    = config["train_path"]
test_path     = config["test_path"]
model_path    = config["model_path"]
batch_size    = config["batch_size"]
augmented_data     = config["augmented_data"]
validation_split   = config["validation_split"]


# create model
base_model = MobileNetV2(include_top=False, weights=weights, 
                          input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
top_layers = base_model.output
top_layers = GlobalAveragePooling2D()(top_layers)
top_layers = Dense(1024, activation='relu')(top_layers)
predictions = Dense(10, activation='softmax')(top_layers)
model = Model(inputs=base_model.input, outputs=predictions)
print ("[INFO] successfully loaded base model and model...")

# start time
start = time.time()

print ("Freezing the base layers. Unfreeze the top 2 layers...")
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print ("Start training...")
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=0.3,
        zoom_range=0.1,
        horizontal_flip=False, 
        validation_split=validation_split)
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=batch_size, 
        save_to_dir=augmented_data)
model.fit_generator(train_generator, verbose=1, epochs=30)

print ("Saving...")
model.save(model_path + "/save_model_stage1.h5") 

"""
print ("Visualization...")
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

print ("Unfreezing all layers...")
for i in range(len(model.layers)):
  model.layers[i].trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

print ("Start training - phase 2...")
model.fit_generator(train_generator, verbose=1, epochs=1)

print ("Saving...")
model.save(model_path + "/save_model_stage2.h5") 
"""
# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
print ("[STATUS] total duration: {}".format(end - start))