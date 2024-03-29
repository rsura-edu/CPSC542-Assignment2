# CPSC542-Assignment2

Link to [Github Repository](https://github.com/rsura-edu/CPSC542-Assignment2).
Link to [external dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset/tree/master) (reuploaded to my repository for convenience)

## 1) Personal Info

- a. Full name: Rahul Sura
- b. Student ID: 2371308
- c. Chapman email: sura@chapman.edu
- d. Course number and section: CPSC 542 - 01
- e. Assignment or exercise number: Assignment 2

## 2) Source Files:

- eda.py
- modules.py
- cnn_model_build.py
- cnn_model_eval.py
- rf_model_build.py
- rf_model_eval.py

## 3) A description of any known compile or runtime errors, code limitations, or deviations from the assignment specification (if applicable):

- Need to use the following versions to ensure no deprecated functions are used:
    - python version 3.8.10
    - sklearn version 0.24.2

## 4) A list of all references used to complete the assignment, including peers (if applicable):

- Discussed with Shree Murthy and Dylan Inafuku about hyperparameter tuning and grad cam implementation
- https://arxiv.org/pdf/1505.04597.pdf for U-net architecture
- https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img for loading images
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html for IOU score
- Old CPSC 393 code for model architecture and matplotlib
- Old CPSC 392 code for random forest

## 5) Instructions for running the assignment

- To build each model:
    - `python3 cnn_model_build.py`
    - `python3 rf_model_build.py`
- To evaluate each model:
    - `python3 cnn_model_eval.py`
    - `python3 rf_model_eval.py`
- For exploratory data analysis:
    - `python3 eda.py`