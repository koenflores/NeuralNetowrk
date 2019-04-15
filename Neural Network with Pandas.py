# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:09:20 2019

@author: koenflores
"""


"""
Created on Fri Nov  9 14:50:56 2018


THIS NN TRAINS ON SUBJECTS 1-80 AND TESTS ON 81-100

@author: koenflores
"""
import pyautogui
import random
import numpy as np
import math

import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
#matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.autograd import Variable
from NNDocumention import NNDocumentation


##############################################################################
# BUILD NEURAL NETWORK CLASS
##############################################################################
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        #Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #Non-Linearity
        #Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        #self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self,x): #x is input
        #Linear function
        out = self.fc1(x)
        #Non-linearity
        out = F.sigmoid(out)
        #out = F.relu6(out)
        #linear function (readout)
        out = self.fc2(out)
        return F.sigmoid(out)
        #return F.relu6(out)
##############################################################################

# %%
##############################################################################
# ESTABLISH PARAMETERS
##############################################################################        

# FILE PARAMETERS
train_data = pd.read_pickle("D:/Dropbox (UFL)/MBL_Lab/Projects/KFlores/MachineLearning/NeuralNetwork/Subject_Numpy/alltraindata_1to80_pandas_truncated2.pkl")
#train_data = pd.read_pickle("D:/Dropbox (UFL)/MBL_Lab/Projects/KFlores/MachineLearning/NeuralNetwork/Subject_Numpy/alltraindata_1to80_pandas_truncated.pkl")
# PARAMETERS FOR TRAINING
num_subjects   = 100                                         # Number of subjects
num_train_subj = 80                                          # Number of subjects in training set
num_epochs     = 200                                        # Number of epochs
learning_rate  = 1e-3                                       # Learning rate
batch_size = 100                                            # Batch Size
features = [6,9,13,16]
#features = list(range(1,48))
features = list(range(1,230))

# PARAMETERS FOR THE PROCESSING
if_GPU = 0;

##############################################################################
# INSTANTIATE MODEL 
##############################################################################
input_dim  = len(features)                                  # Number of inputs
hidden_dim = 150                                            # Number of values in hidden layer
output_dim = 1                                              # Number of values in output layer

num_in = input_dim                                          # Number of features
num_out = 1                                                 # Number of outputs

# BUILD MODEL OBJECT
model = FeedforwardNeuralNetModel(input_dim, output_dim)

# ESTABLISH OPTIMIZATION 
criterion = nn.MSELoss()    # Mean Squared Error Loss
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)


##############################################################################
# TRAIN THE NEURAL NETWORK
##############################################################################

# GET LABELS FOR ALL SUBJECTS


# DETEMRINE TRAINING AND TESTING SUBJECTS
subjects = np.arange(0, num_subjects)                       # List subjects 0 to num_subjects
random.shuffle(subjects)                                    # Randomize subject order
num_test_subj  = num_subjects - num_train_subj              # Number of test subjects

display_step = 1
train_percent = 100                                         # Percent of dataframe used to train: this is 100 because we are testing with 100% of the training dataset
test_percent = 100 - train_percent
total_samples = len(train_data)                             # Number of samples (number of time points)
num_train = int((train_percent/100) * len(train_data))      # Number of training samples
num_test = len(train_data) - num_train                      # Number of testing samples (not used right now)
num_batch = int(num_train / batch_size)                     # Number of batches 

features_and_labels = features
features_and_labels.append(230)                                        # Adds labels to array
features_and_labels.append(231)
features_and_labels.append(232)
features_and_labels.append(233)
train_data = train_data.iloc[:,features_and_labels]

# LOOP OVER EPOCHS
for epoch in range(num_epochs):


    # TRAIN MODEL (I AM NOT SURE WHAT THIS DOES)
    model.train()

    # LOOP OVER BATCHES
    for i in range(num_batch):
        
        
        # GET BATCH FROM DATA AND CONVERT TO TENSOR
        
        batch_xy1 = np.array(train_data.iloc[i*batch_size:((i+1)*batch_size)])
        batch_x1 = batch_xy1[:,0:num_in]
        batch_y1 = batch_xy1[:,(num_in):(num_in+num_out)]
        labels1 = batch_y1
        
        batch_x1 = torch.from_numpy(batch_x1)        # first convert my numpy to tensor
        batch_x1 = batch_x1.float()                  # Then make it float from double since that is what model asks
        batch_x1 = Variable(batch_x1)                # Now make the data into a tensor.Variable for forward pass
        
        batch_y1 = torch.from_numpy(batch_y1)        # first convert my numpy to tensor
        batch_y1 = batch_y1.float()                  # Then make it float from double since that is what model asks
        batch_y1 = Variable(batch_y1)                # Now make the data into a tensor.Variable for forward pass
        
        # ALLOCATE TO GPU IF ON
        if if_GPU:
            batch_x1 = batch_x1.cuda()
            batch_y1 = batch_y1.cuda()
        
        # COMPUTE LOSS 
        output1 = model.forward(batch_x1)             # Run forward model
        
        loss = F.mse_loss(output1, batch_y1)          # Compute loss
        output_1 = np.transpose(np.round(output1.cpu().detach().numpy()))
        #x_acc1 = 1-np.mean(abs(output_1[0,:]-labels1))
        x_acc1 = 1-np.mean(abs(output_1[0]-labels1.squeeze()))
        
        # OPTIMIZE FOR THIS BATCH
        optimizer.zero_grad()                       # Make all the gradients zero first  
        loss.backward()                             # Perform backpropagation
        optimizer.step()                            # Step with optimizer
        
        #if ((loss.data[0]).numpy() >.3):
        if i%display_step == 0:
            print('step [{}/{}], epoch [{}/{}], loss:[{:.4f}], accuracy:[{:.4f}]'
                  .format(i+1, num_batch, epoch + 1, num_epochs, loss.data[0], x_acc1))




# %%
##############################################################################
# APPLY TRAINING DATA TO THE NEURAL NETWORK
##############################################################################
"""
print('............Test Neural Network Model With Training Data............')

# LOOP OVER TRAINING SUBJECTS
for subj in range(num_train_subj):
    
    # LOAD DATA        
    x_train   = np.load(subjectdata + str(train_subjects[subj]+101) + '_truncated2.npy')
    #x_train = x_train[features,:]
    n_samples = x_train.shape[1]                            # Total number of samples

    # GET LABEL
    x_out     = subj_labels[train_subjects[subj]]*np.ones([1, n_samples])

    if subj == 0:
        my_train = x_train
        my_label = subj_labels[train_subjects[subj]]
    else:
        my_train = np.concatenate((my_train, x_train))
        my_label = np.concatenate((my_label, subj_labels[train_subjects[subj]]))
         
    # GET BATCH FROM DATA 
    batch_x   = torch.from_numpy(np.transpose(x_train))     # first convert my numpy to tensor
    batch_x   = batch_x.float()                             # Then make it float from double since that is what model asks
    if if_GPU:
        batch_x = batch_x.cuda()
    output_x  = model(batch_x)                              # Put training data into the neural network
    output_x0 = np.transpose(np.round(output_x.cpu().detach().numpy()))
    x_acc = 1-np.mean(abs(output_x0-x_out))
    print('Subject [{}/{}], accuracy:[{:.4f}]'
                      .format(subj+1, num_train_subj, x_acc))
        
plt.figure()
for subj in range(num_train_subj):
    if my_label[subj] == 0:
        plt.plot(np.transpose(my_train[subj*input_dim:(subj+1)*input_dim,:]), c='b')
        plt.hold(True)
    if my_label[subj] == 1:
        plt.plot(np.transpose(my_train[subj*input_dim:(subj+1)*input_dim,:]), c='r')
        plt.hold(True)

plt.hold(False)
"""
##############################################################################
# APPLY TEST DATA TO THE NEURAL NETWORK
##############################################################################
print('............Test Neural Network Model With Test Data............')

#test_data = pd.read_pickle("D:/Dropbox (UFL)/MBL_Lab/Projects/KFlores/MachineLearning/NeuralNetwork/Subject_Numpy/alltestdata_scale1_pandas_truncated2.pkl")
test_data = pd.read_pickle("D:/Dropbox (UFL)/MBL_Lab/Projects/KFlores/MachineLearning/NeuralNetwork/Subject_Numpy/alltestdata_81to100_pandas_truncated2.pkl")
test_data = test_data.iloc[:,features_and_labels]
display_step = len(test_data)/num_test_subj
total_accuracy = 0
all_accuracy = 0

for j in range(len(test_data)):
    
        #batch_xy2 = np.array(test_data.iloc[j:j+2])
        #batch_x2 = batch_xy2[:,0:num_in]
        #batch_y2 = batch_xy2[:,(num_in):(num_in+num_out)]
        
        batch_xy2 = np.array(test_data.iloc[j])
        batch_x2 = batch_xy2[0:num_in]
        batch_y2 = batch_xy2[(num_in):(num_in+num_out)]
        
        labels2 = batch_y2
        
        batch_x2 = torch.from_numpy(batch_x2)        # first convert my numpy to tensor
        batch_x2 = batch_x2.float()                  # Then make it float from double since that is what model asks
        
        # ALLOCATE TO GPU IF ON
        if if_GPU:
            batch_x2 = batch_x2.cuda()
            batch_y2 = batch_y2.cuda()
        
        # COMPUTE LOSS 
        output2 = model(batch_x2)             # Run forward model
        output_2 = np.transpose(np.round(output2.cpu().detach().numpy()))
        #x_acc2 = 1-np.mean(abs(output_2[0,:]-labels2))
        x_acc2 = 1-np.mean(abs(output_2[0]-labels2))
        total_accuracy = total_accuracy + x_acc2
        all_accuracy = all_accuracy + x_acc2
        

        if (j+1)%display_step == 0:
            
            print('step [{}/{}], accuracy:[{:.4f}]'
                  .format(j+1, len(test_data), total_accuracy/display_step))
            total_accuracy = 0
            
print('Average Accuracy:[{:.4f}]'
                 .format(all_accuracy/len(test_data)))



