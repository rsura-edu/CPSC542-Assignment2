from modules import *

images, masks = load_data()

images = np.array(images) / 255.0
masks = np.array(masks) / 255.0

# train/val/test split of 70/10/20
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


# Model building/compiling/fitting
model: object # declared for sake of scope
model_file_name = 'my_cnn_model.h5'
try:
    model = load_model(model_file_name)
    print("Existing CNN model exists")
except:
    model = unet()

    # compile/train
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=16, 
                        validation_data=(X_val, y_val), 
                        callbacks=[EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)])
    model.save(model_file_name)

    # plotting training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('my_cnn_history.png')
    
    print("CNN model trained and saved. View 'my_cnn_history.png' for training history")

