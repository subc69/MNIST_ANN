# MNIST_ANN
This project is for mnist digit classification (learning purpose) written in Python 3.6.
The ANN model is implemented at basic level. Back propagation are implemented by writing the chain rule for differentiation using numpy.
All calculations including forward and back propagation are implemented in complete object oriented manner.
Sigmoid function is used for activation in each layer.
The data used for training and testing are mnist image files (.jpg) derived from Kaggle dataset.
Data processing classes are implemented for first deriving the csv data from image files for training and testing and then retriving data for training and testing through data generators implemented using python, pandas, numpy..
Pandas dataframe is used for reading and writing data.
Since test data are unlabelled, to compare predicted values for test data with actual digits, a data visualization functionality is implemented in dataManager class.
A model built with this neural net is put into Model folder.
For training, set the global TEST variable to False
For testing with batch data, set global TEST variable to True and PREDICT_A_DIGIT variable to False
For predicting a digit by selecting and mnist image tile for digits, set PREDICT_A_DIGIT variable to True.
Further work is in progress....
