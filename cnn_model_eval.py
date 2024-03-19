from modules import *
from cnn_model_build import model_file_name

images, masks = load_data()

images = np.array(images) / 255.0
masks = np.array(masks) / 255.0

# train/val/test split of 70/10/20
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)


model = load_model(model_file_name)

# ------------------------------------------------
# text/numeric eval of the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Train loss: {loss}, Train accuracy: {accuracy}')
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')


# Calculate IOU scores for train
for label, X, y in [("Train", X_train, y_train),("Test", X_test, y_test)]: 
    iou_scores = []
    for i in range(len(y)):
        iou = jaccard_score(y[i].flatten(), 
                            (model.predict(X) > 0.33).astype(np.float32)[i].flatten())
        iou_scores.append(iou)

    # avg iou score
    mean_iou = np.mean(iou_scores)
    print(f'Mean {label} IOU: {mean_iou}')

# ------------------------------------------------
# visual evaluation showing original, given mask, and pred mask

plot_3_best_worst(model,
                  [accuracy_score(y_test[i].flatten(), 
                        (model.predict(X_test) > 0.33).astype(np.float32)[i].flatten()) 
                   for i in range(len(y_test))], 
                  X_test, 
                  y_test, 
                  'my_cnn_3best3worst.png')