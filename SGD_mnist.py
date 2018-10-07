############################################## dataprocessing  ###########################################
######## Data Generator for MNIST digit gray images  #####################################################
####### Author: Subhas Chakraborty Date: 02-Oct-2018  ####################################################
## First step: Read both train and test images from file system, #########################################
# convert the read images into 1D pixel array to store into pandas dataframe, ############################
# for training images corresponding label is also appended to the dataframe, #############################
# save the dataframe  in seperate CSV files for training and testing  ####################################
# 2nd step: Use the genertaed CSV files only for all training, validation and trsting purposes  ##########
# Training data read into pandas dataframe, and split into training and validation set which are  ########
# fed to the model through a data generator function.  ###################################################
# Test data are unlabelled. To compare predicted values with actuals, actual images are recreated  #######
# from the test image pixel data to visualize through Matplotlib figure. #################################
##########################################################################################################
TEST = True

PREDICT_A_DIGIT = False

import numpy as np
import os
import pandas as pd
from PIL import Image
import pickle as pkl
import PIL

import matplotlib.pyplot as plt
import math

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

img2nparr = lambda img: np.array(img).astype(np.uint8)
nparr2img = lambda arr: Image.fromarray(arr)


#data preprocessing input path
train_data_img_path = './imgdata/trainImgSet' #empty - to be filled while preprocessing to create .csv file from images using datagenerator
test_data_img_path = './imgdata/testImgSet' #few sample test data provided

#data preprocessing output path
# train_data_path = 'E:/2018_Learning/ML/MNIST/csvdata/train.csv'
# test_data_path = 'E:/2018_Learning/ML/MNIST/csvdata/test.csv'

train_data_path = './csvdata/train.csv'
test_data_path = './csvdata/test.csv'

model_folder = './Model'

IMG_SIZE = (28, 28)

class dataManger():
    def __init__(self, imgDataPath=None):
        self.imgpath = imgDataPath
        if imgDataPath is not None:
            assert os.path.isdir(self.imgpath) == True

    def viewBatchDataDigits(self, batch_data, type='test', pred=None):
        if type == 'test':
            n = len(batch_data)
            m = math.ceil(np.sqrt(n))
            batch_img_arr = batch_data
            if pred is not None:
                temparr = np.array(['' for i in range(m * m)])
                for i in np.arange(len(pred)):
                    temparr[i] = pred[i]
                print(temparr.reshape(m, m))
        else:
            n = len(batch_data[0])
            batch_img_arr = batch_data[0]
            m = math.ceil(np.sqrt(n))
            batch_digit_arr = batch_data[1]
            temparr = np.array(['' for i in range(m*m)])
            for i in np.arange(batch_digit_arr.shape[0]):
                temparr[i] = batch_digit_arr[i]
        fig = plt.figure(figsize=(4, 4))
        for i in np.arange(n):
            img = np.reshape(batch_img_arr[i], IMG_SIZE)
            fig.add_subplot(m, m, i + 1)
            plt.imshow(img)
        plt.show()

    def feedAnImageData(self):
        Tk().withdraw()
        imgfile = askopenfilename(initialdir=test_data_img_path, title='MNIST Images')
        Tk().destroy()
        img = Image.open(imgfile)
        return np.asarray(img.resize(IMG_SIZE)).astype(np.uint8).reshape(1, IMG_SIZE[0] * IMG_SIZE[1])

class trainDataManager(dataManger):
    def __init__(self, imgDataPath=None, traindataPath=train_data_path):
        super().__init__(imgDataPath)
        self.trainDataPath = traindataPath
        self.dataloaded = False

    def trainDataGenerator(self):
        if self.imgpath is None:
            return
        if os.path.isfile(self.trainDataPath):
            print('removing...', self.trainDataPath)
            os.remove(self.trainDataPath)
        df = pd.DataFrame()
        for subdir in os.listdir(self.imgpath):
            for imgfl in os.listdir(os.path.join(self.imgpath, subdir)):
                if imgfl.split('.')[1] == '.jpg' or imgfl.split('.')[1] == '.jpeg' or imgfl.split('.')[1] == '.png':
                    img = Image.open(os.path.join(self.imgpath, subdir, imgfl))
                    if img.size != IMG_SIZE:
                        img = img.resize(IMG_SIZE)
                    df1 = pd.DataFrame(np.concatenate((img2nparr(img).reshape(1, img.size[0] * img.size[1] ), np.array(int(subdir)).reshape(1, -1)), axis = 1), columns=None)
                    df = df.append(df1)

        with open(self.trainDataPath, 'a') as f:
            df.to_csv(f)

    def loadData(self, valid_size = 0.2):
        dataset = pd.read_csv(self.trainDataPath) #shuffle(pd.read_csv(self.trainDataPath))
        self.train_ds, self.valid_ds = train_test_split(dataset, test_size=valid_size, random_state=42)
        self.dataloaded = True

    def feedBatchData(self, type='train', bath_size = 16):
        if not self.dataloaded:
            self.loadData()
        ds = self.train_ds
        if type == 'valid':
            ds = self.valid_ds
        X = ds.iloc[:, 1:-1].values
        y = ds.iloc[:, -1].values
        n_batch = X.shape[0] // bath_size #n_batch = 10 for testing the code
        for i in np.arange(n_batch):
            yield X[i*bath_size:i*bath_size+bath_size, :], y[i*bath_size:i*bath_size+bath_size]

    def feedBatchValidData(self, bath_size = 16):
        ds = self.valid_ds
        X = ds.iloc[:, 1:-1].values
        y = ds.iloc[:, -1].values
        n_batch = X.shape[0] // bath_size #n_batch = 10 for testing the code
        for i in np.arange(n_batch):
            yield X[i*bath_size:i*bath_size+bath_size, :], y[i*bath_size:i*bath_size+bath_size]

class testDataManager(dataManger):
    def __init__(self, imgDataPath=None, testdataPath=test_data_path):
        super().__init__(imgDataPath)
        self.testDataPath = testdataPath
    def testDataGenerator(self):
        if self.imgpath is None:
            return
        if os.path.isfile(self.testDataPath):
            print('removing...', self.testDataPath)
            os.remove(self.testDataPath)
        df = pd.DataFrame()
        #for subdir in os.listdir(self.imgpath):
        for imgfl in os.listdir(self.imgpath):
            if imgfl.split('.')[1] == '.jpg' or imgfl.split('.')[1] == '.jpeg' or imgfl.split('.')[1] == '.png':
                img = Image.open(os.path.join(self.imgpath, imgfl))
                if img.size != IMG_SIZE: #(28, 28)
                    img = img.resize(IMG_SIZE)
                df1 = pd.DataFrame(img2nparr(img).reshape(1, img.size[0] * img.size[1]), columns=None)
                df = df.append(df1)
        print("writing....")
        with open(self.testDataPath, 'a') as f:
            df.to_csv(f) #, header=False, index=False)
        print("Wrote to ", self.testDataPath)

    def loadTestBatch(self, batch_size=16):
        dataset = pd.read_csv(self.testDataPath)
        X = np.array(dataset.iloc[:, 1:].values).astype(np.uint8)
        #print(X.shape)
        n_batch = X.shape[0] // batch_size
        for i in np.arange(n_batch):
            yield X[i*batch_size:i*batch_size+batch_size, :]

######################################################################################################

####################################ANN.py##########################################################
######## Simple Artificial Neural Net model############
####### Author: Subhas Chakraborty Date: 03-Oct-2018#######
####### Input attributes are 2D image pixel values #####
##### stretched into one dimensional pixel value array##

#import numpy as np

sigmoid = lambda x: 1/(1 + np.exp(-x))
softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
#softmax2 = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

# a = np.array([[2, 3, 5], [6,7,8]])
# print(softmax(a))
# print(softmax2(a))

class layer():
    def __init__(self, n_out, n_in = None, activation=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.a = None
        self.delta = None
    def initParams(self):
        self.W = np.random.uniform(
            low=-np.sqrt(1 / (self.n_in + self.n_out)),
            high=np.sqrt(1 / (self.n_in + self.n_out)),
            size=(self.n_in, self.n_out))
        self.b = np.zeros((self.n_out,))

        self.params = [self.W, self.b]
        return self

class ANN():
    def __init__(self, n_class, activation='sigmoid', input=None):
        self.layers = []
        self.n_class = n_class if n_class > 2 else 1
        self.output_activation_fn = activation
        #self.training_on = False
    def addLayer(self, layer):
        if len(self.layers) == 0:
            assert (layer.n_in is not None), 'For first input layer input shape is mandatory'
        else:
            layer.n_in = self.layers[-1].n_out
        #self.training_on = False
        self.layers.append(layer.initParams())
        return self

    def forward(self, X):
        for l in self.layers:
            l.a = X
            if l.activation == 'sigmoid':
                activation = sigmoid
            elif l.activation == 'softmax':
                activation = softmax
            X = np.dot(X, l.W) + l.b if l.activation is None else activation(np.dot(X, l.W) + l.b)
        self.layers.append(layer(0)) #Adding final layer represening output of neural net. This layer will have no output
        self.layers[-1].a = X
        return X #z

    def backward(self, h, y):
        assert h.shape == y.shape
        self.layers[-1].delta = h - y
        for i in np.arange(len(self.layers) - 2, 0, -1):
            self.layers[i].delta = np.dot(self.layers[i + 1].delta, np.transpose(self.layers[i].W)) * \
                                   self.layers[i].a * (1 - self.layers[i].a)
        return np.mean(-(y * np.log(h) + (1 - y) * np.log(1 - h))) if self.layers[-2].activation == 'sigmoid' \
            else np.mean(np.square(self.layers[-1].delta)) / 2

    def train(self, X, y, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.input_shape = X.shape
        assert (X.shape[0],) == y.shape
        if self.n_class > 2:
            y = self.one_hot(y)
        else:
            y = y.reshape(-1, 1)
        loss = self.backward(self.forward(X), y)
        self.applyGrad()
        self.clearData()
        return loss

    def predict(self, X):
        for l in self.layers:
            if l.activation == 'sigmoid':
                activation = sigmoid
            elif l.activation == 'softmax':
                activation = softmax
            X = np.dot(X, l.W) + l.b if l.activation is None else activation(np.dot(X, l.W) + l.b)
        return np.argmax(X, axis=1)

    def accuracy(self, y, y_pred):
        assert y.shape == y_pred.shape
        return np.mean(np.equal(y, y_pred))

    def loss(self, X, y):
        assert (X.shape[0], ) == y.shape
        h = X
        y = self.one_hot(y)
        for l in self.layers:
            if l.activation == 'sigmoid':
                activation = sigmoid
            elif l.activation == 'softmax':
                activation = softmax
            h = np.dot(h, l.W) + l.b if l.activation is None else activation(np.dot(h, l.W) + l.b)
        return np.mean(-(y * np.log(h) + (1- y) * np.log(1 - h))) if self.layers[-1].activation == 'sigmoid' \
                else np.mean(np.square(self.layers[-1].delta)) / 2

    def calculate_gradient(self):
        n_layers = len(self.layers) - 1 # Excluding the final output layer
        grad = []
        for nl in range(n_layers):
            DELTA = [np.dot(np.transpose(self.layers[nl].a), self.layers[nl + 1].delta) / self.input_shape[0], np.mean(self.layers[nl + 1].delta, axis=0)]
            grad.append(DELTA)
        return grad

    def applyGrad(self):
        grad = self.calculate_gradient()
        n_grad = len(grad)
        for nl in range(n_grad):
            self.layers[nl].W -= self.learning_rate * grad[nl][0]
            self.layers[nl].b -= self.learning_rate * grad[nl][1]

    def one_hot(self, y):
        y_out = np.zeros((y.shape[0], self.n_class))
        y_out[np.arange(y.shape[0]), y] = 1
        return y_out

    def clearData(self):
        for layer in self.layers:
            layer.a = None
            layer.delta = None
        self.input_shape = None
        self.layers.pop() #Remove the last layer which was appended as output layer in forward pass. It will be again appended during next batch training


############################################### Training  #####################
#############################Author: Subhas Chakraborty Date: 06-Oct-2018 #####
n_epoch = 10
batch_size = 128
learning_rate = 0.001
n_predictors = 28 * 28
n_class = 10
n_hidden1_nodes = (n_predictors + n_class) // 2
# n_hidden2_nodes = n_hidden1_nodes
# n_hidden3_nodes = n_hidden2_nodes // 2
# n_hidden4_nodes = n_hidden3_nodes // 2

def train(n_epoch=n_epoch, batch_size=batch_size, learning_rate=learning_rate):
    ann = ANN(n_class, 'sigmoid')

    ann = ann.addLayer(layer(n_hidden1_nodes, n_predictors, activation='sigmoid'))
    # ann = ann.addLayer(layer(n_hidden2_nodes, activation='sigmoid'))
    # ann = ann.addLayer(layer(n_hidden2_nodes, activation='sigmoid'))
    # ann = ann.addLayer(layer(n_hidden2_nodes, activation='sigmoid'))
    ann.addLayer(layer(ann.n_class, activation=ann.output_activation_fn))

    best_validation_loss = np.inf
    final_accuracy = 0
    threshold = 0.005
    history = {'train': [], 'valid': []}

    for epoch in range(n_epoch):
        valid_loss_accuracy = {'loss': [], 'accu': []}
        trian_DM = trainDataManager(train_data_img_path, train_data_path)
        for (i, batchdata_train) in enumerate(trian_DM.feedBatchData(bath_size=batch_size)):
            loss = ann.train(batchdata_train[0], batchdata_train[1], learning_rate=learning_rate)
            accuracy = ann.accuracy(batchdata_train[1], ann.predict(batchdata_train[0]))
            #print(loss, accuracy)
            print("Epoch: %d, Batch: %d, Training Loss: %.5f, Training Accuracy: %.5f" % (epoch+1, i+1, loss, accuracy) )
            history['train'].append([epoch, i, loss, accuracy])

        [(valid_loss_accuracy['loss'].append(ann.loss(batchdata_valid[0], batchdata_valid[1])),
                        valid_loss_accuracy['accu'].append(ann.accuracy(batchdata_valid[1], ann.predict(batchdata_valid[0])))) \
                        for batchdata_valid in trian_DM.feedBatchValidData(bath_size=batch_size)]
        this_valid_loss = np.mean(valid_loss_accuracy['loss'])
        this_accuracy = np.mean(valid_loss_accuracy['accu'])
        if this_valid_loss < best_validation_loss * (1 - threshold):
            best_validation_loss = this_valid_loss
            final_accuracy = this_accuracy
            with open(os.path.join(model_folder, 'model01'), 'wb') as f:
                pkl.dump(ann, f)
        print("Epoch: %d, Validation Loss: %.5f, Validation accuracy: %.5f" % (epoch+1, this_valid_loss, this_accuracy))

        history['valid'].append([epoch, this_valid_loss, this_accuracy])
    print('Model saved for validation loss: %.5f and accuracy: %.5f' % (best_validation_loss, final_accuracy))
    return history

def test():
    f = open(os.path.join(model_folder, 'model01'), 'rb')
    ann = pkl.load(f)
    f.close()
    testdataManager = testDataManager(test_data_img_path, test_data_path)
    for testImgBatch in testdataManager.loadTestBatch(batch_size=16):
        y_pred = ann.predict(testImgBatch)
        testdataManager.viewBatchDataDigits(testImgBatch, 'test', y_pred)

        text = input('Enter 0 to stop testing. To continue enter any other key: ')
        if text == '0':
            break

def main():
    if PREDICT_A_DIGIT:
        with open(os.path.join(model_folder, 'model01'), 'rb') as f:
            ann = pkl.load(f)
            print(ann.predict(dataManger().feedAnImageData()))
        return
    if not TEST:
        history = train()
        print('Training history: ', history['train'])
        print('Validation history: ', history['valid'])
    else:
        test()

if __name__ == '__main__':
    main()