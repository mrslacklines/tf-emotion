
import cv2
import numpy as np
import os
import random
from imutils import paths
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from time import time
from trueface.recognition import FaceRecognizer

from models import generate_vgg16


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 0.0001
BS = 32
IMAGE_DIMS = (96, 96, 3)
TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcGlfa2V5Ijoia2RhTEJpUExUUzMzeG1iUUVVdzltNWl0UUluN2VMV3g2S3R4S1hrViIsImV4cGlyeV90aW1lX3N0YW1wIjoxNTM0MzQxNjAwLjB9.15RFd0hUEAo8TlBT1rfiedXZbEHwEcUzMASkbY8JhXU'


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
patience = 20
earlyStopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')
checkpoint = ModelCheckpoint(
    'model-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True,
    mode='auto')

reduce_lr = ReduceLROnPlateau(
    'val_loss', factor=0.00002, patience=int(patience / 4), verbose=1)


fr = FaceRecognizer(
    ctx='cpu',
    fd_model_path='model',
    fr_model_path='model.trueface',
    params_path='model.params',
    token=TOKEN)


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('/floyd/input/emotion')))
random.seed(42)
random.shuffle(imagePaths)


# initialize the data and labels
data = []
labels = []


# loop over the input images
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2].split("_")
    # load the image, pre-process it, and store it in the data list
    if label[0] in ['checkpoints', '.ipynb']:
        continue
    # extract chip and append face to dataset instead of full image
    # preprocessed -> emotion -> chip
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image) / 255.0
    data.append(image)
    print len(data)
    # extract set of class labels from the image path and update the
    # labels list
    print label
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42)


# construct the image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

model = generate_vgg16(len(mlb.classes_), in_shape=IMAGE_DIMS)

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


model.compile(
    loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1, callbacks=[checkpoint, earlyStopping])
