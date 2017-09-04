# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 18:24:15 2017
@author: jacob
@https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""
#==============================================================================
# activation         = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# optimizer          = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# kernel_initializer = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# batch_size         = [10, 20, 40, 60, 80, 100, 200, 300, 400, 500]
# epochs             = [10, 50, 100]
# dropout_rate       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# neurons            = [100, 200, 300, 400, 500]
# kernel_constraint  = [1, 2, 3, 4, 5]
#==============================================================================
import numpy
import itertools
import os
import pandas as pd
import time
from keras.wrappers.scikit_learn import KerasClassifier
time.sleep(5)
from keras.layers import Dropout, Dense
from keras.constraints import maxnorm
from keras.models import Sequential, model_from_json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# train
train  = pd.read_csv("/media/jacob/database/INGlife/train.csv")
train.columns = train.columns.str.replace('.','_')
trainX = train.values[:,train.columns!="Y"].astype(float)
trainY = train.values[:,train.columns=="Y"]
trainY = list(itertools.chain.from_iterable(trainY))
encoder.fit(trainY)
trainY = encoder.transform(trainY)


# valid
valid  = pd.read_csv("/media/jacob/database/INGlife/valid.csv")
valid.columns = valid.columns.str.replace('.','_')
validX = valid.values[:,valid.columns!="Y"].astype(float)
validY = valid.values[:,valid.columns=="Y"]
validY = list(itertools.chain.from_iterable(validY))
encoder.fit(validY)
validY = encoder.transform(validY)


# test 
test = pd.read_csv("/media/jacob/database/INGlife/test.csv")
test.columns = test.columns.str.replace('.','_')
testX = test.values[:,test.columns!="Y"].astype(float)
testY = test.values[:,test.columns=="Y"]
testY = list(itertools.chain.from_iterable(testY))
encoder.fit(testY)
testY = encoder.transform(testY)
#==============================================================================


#==============================================================================
# cartesian grid search 
#==============================================================================
def create_cartesian_grid(neurons1=200,neurons2=200):
    # create model
    cartesian_model = Sequential()
    cartesian_model.add(Dense(neurons1,input_dim=train.shape[1]-1))
    cartesian_model.add(Dense(neurons2))
    cartesian_model.add(Dense(1))
    # Compile model
    cartesian_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cartesian_model


# create model
numpy.random.seed(1234)
#cartesian_KerasModel      = KerasClassifier(build_fn=create_cartesian_grid, verbose=1, epochs=10)
cartesian_KerasModel      = KerasClassifier(build_fn=create_cartesian_grid, verbose=1)
neurons1                  = [10, 20]
neurons2                  = [10, 20]
cartesian_param_grid      = dict(neurons1=neurons1,neurons2=neurons2)
cartesian_grid            = GridSearchCV(estimator=cartesian_KerasModel, param_grid=cartesian_param_grid, n_jobs=1)
cartesian_grid_result     = cartesian_grid.fit(trainX, trainY) 
print("Best: %f using %s" % (cartesian_grid_result.best_score_, cartesian_grid_result.best_params_))
   

# the best hyper-parameters
best_neurons1   = cartesian_grid_result.best_params_.get('neurons1')
best_neurons2   = cartesian_grid_result.best_params_.get('neurons2')
#==============================================================================


#==============================================================================
# random grid search 
#==============================================================================
def create_random_grid(activation='relu',dropout_rate1=0.0,dropout_rate2=0.0,optimizer='Adam',kernel_initializer='normal',kernel_constraint=5):
    # create model
    random_model = Sequential()
    random_model.add(Dense(best_neurons1,input_dim=train.shape[1]-1,kernel_initializer=kernel_initializer,activation=activation,kernel_constraint=maxnorm(kernel_constraint)))
    random_model.add(Dropout(dropout_rate1))
    random_model.add(Dense(best_neurons2,kernel_initializer=kernel_initializer,activation=activation,kernel_constraint=maxnorm(kernel_constraint)))
    random_model.add(Dropout(dropout_rate2))
    random_model.add(Dense(1,kernel_initializer=kernel_initializer,activation='sigmoid'))
    # Compile model
    random_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return random_model

    
# create model
numpy.random.seed(1234)
random_KerasModel = KerasClassifier(build_fn=create_random_grid, verbose=1)


# define the grid search parameters
activation         = ['relu', 'tanh', 'sigmoid']
optimizer          = ['SGD', 'RMSprop', 'Adam']
kernel_initializer = ['uniform', 'normal', 'glorot_normal', 'glorot_uniform']
batch_size         = [100, 200, 300, 400, 500]
epochs             = [50, 100, 200]
dropout_rate1      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
dropout_rate2      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
kernel_constraint  = [1, 2, 3, 4, 5]
random_param_grid  = dict(activation=activation
                          ,optimizer=optimizer
                          ,kernel_initializer=kernel_initializer
                          ,batch_size=batch_size
                          ,epochs=epochs
                          ,dropout_rate1=dropout_rate1
                          ,dropout_rate2=dropout_rate2
                          ,kernel_constraint=kernel_constraint)

random_grid = RandomizedSearchCV(estimator=random_KerasModel, param_distributions=random_param_grid, n_jobs=1, n_iter=3 ,random_state=1234)
random_grid_result = random_grid.fit(trainX, trainY)
print("Best: %f using %s" % (random_grid_result.best_score_, random_grid_result.best_params_))


# the best hyper-parameters
best_activation         = random_grid_result.best_params_.get('activation')
best_optimizer          = random_grid_result.best_params_.get('optimizer')
best_kernel_initializer = random_grid_result.best_params_.get('kernel_initializer')
best_batch_size         = random_grid_result.best_params_.get('batch_size')
best_epochs             = random_grid_result.best_params_.get('epochs')
best_dropout_rate1      = random_grid_result.best_params_.get('dropout_rate1')
best_dropout_rate2      = random_grid_result.best_params_.get('dropout_rate2')
best_kernel_constraint  = random_grid_result.best_params_.get('kernel_constraint')
#==============================================================================


#==============================================================================
# model
#==============================================================================
model = Sequential()
model.add(Dense(best_neurons1, input_dim=train.shape[1]-1, kernel_initializer=best_kernel_initializer, activation=best_activation))
model.add(Dropout(best_dropout_rate1))
model.add(Dense(best_neurons2, kernel_initializer=best_kernel_initializer, activation=best_activation))
model.add(Dropout(best_dropout_rate2))
model.add(Dense(1, kernel_initializer=best_kernel_initializer, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX,trainY,epochs=best_epochs,batch_size=best_batch_size,verbose=1)
#model.fit(trainX,trainY,validation_split=0.33,epochs=best_epochs,batch_size=best_batch_size,verbose=1)
scores = model.evaluate(testX,testY);
print("\n","acc:", scores[1]*100)


model = Sequential()
model.add(Dense(100, input_dim=train.shape[1]-1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX,trainY,epochs=10,batch_size=1000,verbose=1)


# save model 
os.getcwd()
model_json = model.to_json()
with open("model.json","w") as json_file:
  json_file.write(model_json)
model.save_weights("model.h5")
print("\n","save model!")


# load model 
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# predict test
proba_test_TF = loaded_model.predict_proba(testX)
proba_test_TF = [val for sublist in proba_test_TF for val in sublist]
proba_test_TF = [round(i,6) for i in proba_test_TF]
proba_test_TF = pd.DataFrame({"proba_test_TF":proba_test_TF})
proba_test_TF.to_csv("/media/jacob/database/INGlife/proba_test_TF.csv",index=False)
class_test_TF = loaded_model.predict_classes(testX)
class_test_TF = [val for sublist in class_test_TF for val in sublist]
class_test_TF = [round(i,6) for i in class_test_TF]
class_test_TF = pd.DataFrame({"class_test_TF":class_test_TF})
class_test_TF.to_csv("/media/jacob/database/INGlife/class_test_TF.csv",index=False)


# predict valid 
proba_valid_TF = loaded_model.predict_proba(validX)
proba_valid_TF = [val for sublist in proba_valid_TF for val in sublist]
proba_valid_TF = [round(i,6) for i in proba_valid_TF]
proba_valid_TF = pd.DataFrame({"proba_valid_TF":proba_valid_TF})
proba_valid_TF.to_csv("/media/jacob/database/INGlife/proba_valid_TF.csv",index=False)
class_valid_TF = loaded_model.predict_classes(validX)
class_valid_TF = [val for sublist in class_valid_TF for val in sublist]
class_valid_TF = [round(i,6) for i in class_valid_TF]
class_valid_TF = pd.DataFrame({"class_valid_TF":class_valid_TF})
class_valid_TF.to_csv("/media/jacob/database/INGlife/class_valid_TF.csv",index=False)


print("\n","done!")
#==============================================================================



