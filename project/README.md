# Final Project Report for CS 175, Spring 2020
**Project Title:** Water Filled Lung Detector

**Project Number:** Group 14

**Student Name(s)**

Allan Tran, 61735904. allannt@uci.edu

Jason Davis, 22336416, jasonbd@uci.edu

Eva Dai, 94015611, ydai8@uci.edu

### Files we have:

* [data](./data)
  - [stage_2_train_images](./data/stage_2_train_images): folder that contains the sample of our training images
  - [stage_2_test_images](./data/stage_2_test_images): folder that contains the sample of our testing images
  - [train.csv](./data/train.csv): contains labels of the sample training data
  - [test.csv](./data/test.csv): contains labels of the sample testing data
 
* [src](./src)
  - [dataset.py](./src/dataset.py): dataset class that loads the dataset from the data folder
  - [classifier.py](./src/classifier.py): classifier model for identifying pneumonia
  - [train.ipynb](./src/train.ipynb): contains the code for loading the data, the classifier model, training the model, and check accuracy on the model
  - [faster_training.ipynb](./src/faster_training.ipynb): contains the code for loading the training data and training the fasterRCNN model
  - [faster_testing.ipynb](./src/faster_testing.ipynb): contains the code for loading the testing data and testing the fasterRCNN model
  - [combining_predictions.ipynb](./src/combining_predictions.ipynb): contains the code for loading predictions for both classifier and combining them using AND operator

* [project.ipynb](./project.ipynb): final demonstration of the entire project using a small size dataset
* [project.html](./project.html): html version of the final demonstration of the project
