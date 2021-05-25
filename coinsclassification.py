
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import glob

input_path = Path(r'C:\Users\sneha\Downloads\archive\classification_dataset\all')

im_size = 320

image_files =list(glob.glob('*.jpg'))


def read_file(fname):
    # Read image
    print(fname)
    im = Image.open(fname)

    # Resize
    im.thumbnail((im_size, im_size))

    # Convert to numpy array
    im_array = np.asarray(im)

    # Get target
    target = int(fname.split('_')[0])

    return im_array, target

images = []
targets = []

for image_file  in image_files:

    
    image, target = read_file(image_file)
    
    images.append(image)
    targets.append(target)

X = (np.array(images).astype(np.float32) / 127.5) - 1
y_cls = np.array(targets)

X.shape, y_cls.shape

coins_ids = {
    5: 0,
    10: 1,
    25: 2,
    50: 3,
    100: 4
}

ids_coins = [5, 10, 25, 50, 100]

y = np.array([coins_ids[coin] for coin in y_cls])

X_train, X_valid, y_train, y_valid, fname_train, fname_valid = train_test_split(
    X, y, image_files, test_size=0.2, random_state=42)
print(X_train,X_valid, y_train, y_valid, fname_train, fname_valid)

im_width = X.shape[2]
im_height = X.shape[1]

im_width, im_height

from keras.layers import Conv2D, MaxPool2D, Flatten, GlobalAvgPool2D, GlobalMaxPool2D, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
model = Sequential()
model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(im_height, im_width, 3)) )
model.add( MaxPool2D(2) )

model.add( Conv2D(32, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(64, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(128, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(256, 3, activation='relu', padding='same') )

# Transition between CNN and MLP
model.add( GlobalAvgPool2D() )

# MLP network
model.add( Dense(256, activation='relu') )

model.add( Dense(5, activation='softmax') )

model.summary()
optim = Adam(learning_rate=1e-3)
model.compile(optim, 'sparse_categorical_crossentropy', metrics=['acc'])

callbacks = [
    ReduceLROnPlateau(patience=5, factor=0.1, verbose=True),
    ModelCheckpoint('best.model', save_best_only=True),
    EarlyStopping(patience=12)
]
history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_valid, y_valid), batch_size=32,callbacks=callbacks)
                   
model.evaluate(X_valid, y_valid)

y_pred = model.predict(X_valid)
y_pred_cls = y_pred.argmax(1)
print("y_valid",y_valid)
print("y_pred",y_pred)
print("y_pred_cls",y_pred_cls)


print(classification_report(y_valid,y_pred_cls))

