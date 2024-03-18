# File to store modularizable code to be imported

# all possible imports
import os
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# constants
data_dir = 'human_mask_data'
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

# function to get processed data
def load_data():
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    img_size = (224, 224)
    images = []
    masks = []
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        image = load_img(img_path, target_size=(224, 224))
        mask = load_img(mask_path, target_size=(224, 224), color_mode='grayscale')
        images.append(np.array(image))
        masks.append(np.array(mask))
    return np.array(images), np.array(masks)

# defined u-net model according to https://arxiv.org/pdf/1505.04597.pdf
def unet(input_size=(224, 224, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2D(128, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2D(64, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2D(32, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

def plot_3_best_worst(model, accuracy_scores, X, y, filename="my_3best.png"):
    sorted_accuracies = sorted(accuracy_scores)
    best_1_index = accuracy_scores.index(sorted_accuracies[-1])
    best_2_index = accuracy_scores.index(sorted_accuracies[-2])
    best_3_index = accuracy_scores.index(sorted_accuracies[-3])
    worst_1_index = accuracy_scores.index(sorted_accuracies[2])
    worst_2_index = accuracy_scores.index(sorted_accuracies[1])
    worst_3_index = accuracy_scores.index(sorted_accuracies[0])    
    
    plt.figure(figsize=(21, 12))
    for i, (best, worst) in enumerate(((best_1_index, worst_1_index), (best_2_index, worst_2_index), (best_3_index, worst_3_index)), 1):
        plt.subplot(3, 6, (i-1)*6+1)
        plt.imshow(X[best])
        plt.imshow(model.predict(X)[best], cmap='hot', interpolation='nearest', alpha=0.4)
        plt.title('Original Image')
        plt.axis('off')
    
        plt.subplot(3, 6, (i-1)*6+2)
        plt.imshow(y[best], cmap='gray')
        plt.title(f'\n*BEST #{i}*\n\nGiven Mask')
        plt.axis('off')

        plt.subplot(3, 6, (i-1)*6+3)
        plt.imshow((model.predict(X) > 0.33).astype(np.float32)[best], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        # --------

        plt.subplot(3, 6, (i-1)*6+4)
        plt.imshow(X[worst])
        plt.imshow(model.predict(X)[worst], cmap='hot', interpolation='nearest', alpha=0.4)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(3, 6, (i-1)*6+5)
        plt.imshow(y[worst], cmap='gray')
        plt.title(f'\n*WORST #{i}*\n\nGiven Mask')
        plt.axis('off')

        plt.subplot(3, 6, (i-1)*6+6)
        plt.imshow((model.predict(X) > 0.33).astype(np.float32)[worst], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.savefig(filename)
