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
    print("Existing RF model exists")
except Exception as e:
    rf_model = RandomForestClassifier(n_estimators=1, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # serialize the model to a saved file
    with open(model_file_name, 'wb') as f:
        pickle.dump(rf_model, f)

    print("RF model trained and saved")