import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

data_dir = 'human_mask_data'
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

# loading images and masks
image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])

images = []
masks = []

for img_path, mask_path in zip(image_paths, mask_paths):
    image = load_img(img_path, target_size=(224, 224))
    mask = load_img(mask_path, target_size=(224, 224), color_mode='grayscale')
    
    images.append(img_to_array(image))
    masks.append(img_to_array(mask))

images = np.array(images) / 255.0  
masks = np.array(masks) / 255.0    

# train/val/test split of 70/10/20
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.666666666, random_state=42)


# Model building/compiling/fitting
model: object # declared for sake of scope
model_file_name = 'my_model.h5'
try:
    model = load_model(model_file_name)
except:
    # Define U-Net model
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

    model = unet()

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

    model.save(model_file_name)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Train loss: {loss}, Train accuracy: {accuracy}')
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


iou_scores = []
threshold = 0.32
for i in range(len(y_test)):
    iou = jaccard_score(y_test[i].flatten(), 
                        (model.predict(X_test) > threshold).astype(np.float32)[i].flatten())
    iou_scores.append(iou)
    
# avg iou score
mean_iou = np.mean(iou_scores)
print(f'Mean IOU: {mean_iou}')

# ------------------------------------------------
# visual evaluation showing original, given mask, and pred mask

# 3 random images to evaluate
random_indices = random.sample(range(len(X_test)), 3)
random_images = X_test[random_indices]
random_masks = y_test[random_indices]
pred_random_masks = model.predict(random_images)
# thresholded to make binary mask from grayscale mask
pred_random_masks_binary = (pred_random_masks > threshold).astype(np.float32) 

# display OG image, given mask, and predicted mask for the random images
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 3, i*3+1)
    plt.imshow(random_images[i])
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 3, i*3+2)
    plt.imshow(random_masks[i].reshape(224, 224), cmap='gray')
    plt.title('Given Mask')
    plt.axis('off')

    plt.subplot(3, 3, i*3+3)
    plt.imshow(pred_random_masks_binary[i].reshape(224, 224), cmap='gray')
    plt.title('Predicted Binary Mask')
    plt.axis('off')

plt.tight_layout()
plt.savefig('my_cnn.png')
