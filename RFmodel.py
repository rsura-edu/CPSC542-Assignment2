from modules import *

images, masks = load_data()

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
y_train = y_train.reshape(len(y_train), -1)
y_test = y_test.reshape(len(y_test), -1)

model_file_name = 'my_rf_model.pkl'
# Load or fit the Random Forest model
try:
    with open(model_file_name, 'rb') as f:
        rf_model = pickle.load(f)
except Exception as e:
    rf_model = RandomForestClassifier(n_estimators=1, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # serialize the model to a saved file
    with open(model_file_name, 'wb') as f:
        pickle.dump(rf_model, f)

# predict
train_pred_masks = rf_model.predict(X_train)
test_pred_masks = rf_model.predict(X_test)

# ------------------------------------------------
# text/numeric eval of the model

train_accuracy = accuracy_score(y_train.flatten(), train_pred_masks.flatten())
train_iou = jaccard_score(y_train.flatten(), train_pred_masks.flatten(), average='macro')
test_accuracy = accuracy_score(y_test.flatten(), test_pred_masks.flatten())
test_iou = jaccard_score(y_test.flatten(), test_pred_masks.flatten(), average='macro')

print("Training Accuracy: ", train_accuracy)
print("Testing Accuracy: ", test_accuracy)
print("Training IoU: ", train_iou)
print("Testing IoU: ", test_iou)

# ------------------------------------------------
# visual evaluation showing original, given mask, and pred mask

random_indices = random.sample(range(len(X_test)), 3) # random images to eval
test_pred_masks_unflat = unflatten_masks(test_pred_masks, (224, 224)) # unflatten

plt.figure(figsize=(12, 12))
for i, index in enumerate(random_indices):
    # original image
    plt.subplot(3, 3, i*3+1)
    plt.imshow(X_test[index].reshape(224, 224, 3))
    plt.axis('off')
    plt.title('Original Image')

    # given mask
    plt.subplot(3, 3, i*3+2)
    plt.imshow(y_test[index].reshape(224, 224), cmap='gray')
    plt.axis('off')
    plt.title('Given Mask')

    # predicted mask
    plt.subplot(3, 3, i*3+3)
    plt.imshow(test_pred_masks_unflat[index], cmap='gray')
    plt.axis('off')
    plt.title('Predicted Mask')

plt.tight_layout()
plt.savefig('my_rf.png')