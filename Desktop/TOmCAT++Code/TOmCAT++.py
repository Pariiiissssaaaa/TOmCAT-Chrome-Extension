import numpy as np
from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(5)
np.random.seed(5)

from collections import defaultdict
import operator
from operator import itemgetter
import random
import json
from sklearn.metrics import classification_report
import more_itertools as mit

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras import optimizers

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier



import argparse

#...... default values of the hyperparameters are obtained through grid-search ........
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batchSize', default=8, type=int, help='Size of the batches')

parser.add_argument('--lr', default=0.001, type=float, help='Learning Rate')

parser.add_argument('--epochs', default=2, type=int, help='Number of epochs')

parser.add_argument('--embeddingSize', default=16, type=int, help='Size of Embedding Layer')

parser.add_argument('--Dropout', default=0.5, type=float, help='Dropout rate')

parser.add_argument('--hiddenSize', default=64, type=int, help='Size of hidden states')

parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizing function')

parser.add_argument('--init', default='he_uniform', type=str, help='Weight initialization')


args = parser.parse_args()
print (args)


learn_rate=args.lr
dropRate=args.Dropout
epochs=args.epochs
embedding_size=args.embeddingSize
hidden_size=args.hiddenSize
batch_size=args.batchSize
initial=args.init 

#fixed parameters 
maxlen = 150 #number of hidden states in LSTM model (other Hyperparameters are selected via grid-search)
minlen = 32 
overlap=20
max_features=4 # size of vocabulary, in our rating scenario we have 4 possible rating 1,2-->0, 3-->1, 4-->2, 5-->3


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.RandomState(5).permutation(len(a))
    return a[p], b[p]



#manipulated and benign items ids 
manipulated_items, benign_items=set(), set()


#Add Secondry prods, ids 1 to 3467 belong to target products
for i in range(1, 3468):
    manipulated_items.add(str(i))


#Add Primary prods
with open('Primary_simple.txt') as f:
    for line in f:
        line=line.strip().split(',')
        manipulated_items.add(line[0])


#Add benign prods
with open('Expanded_Benign_simple.txt') as f:
    for line in f:
        line=line.strip().split(',')
        if line[0] not in manipulated_items:
            benign_items.add(line[0])


#manipulated and benign rating series
prods_benign, prods_manipulated={}, {}



def ReadData(inputFile, prod_set, prods):
    
    with open(inputFile) as f:
        for line in f:
            line=line.strip().split(',')
            if line[0] in prod_set:
                iD, rate, time=line[0], line[2], line[3]
                if iD not in prods:
                    prods[iD] = []
                prods[iD].append((int(float(rate)), float(time)))
        for prod in prods:
            prods[prod] = sorted(prods[prod], key=operator.itemgetter(1))




ReadData('Expanded_Benign_simple.txt', manipulated_items, prods_manipulated)
ReadData('Primary_simple.txt', manipulated_items, prods_manipulated)
ReadData('Expanded_Benign_simple.txt', benign_items, prods_benign)





usedTargetInTrain=set()


def Build_Train_Set(prods, flag, lmin):
    global x_train, y_train, usedTargetInTrain
    for prod in prods:
        l=len(prods[prod])
        if l>=lmin and l<=maxlen:
            usedTargetInTrain.add(prod)
            x=[j[0]-2 if j[0]!=1 else j[0]-1 for j in prods[prod]]
            x=np.array(x)
            x_train.append(x)
            if flag==1:
                y_train.append(1)
            else:
                y_train.append(0)



x_train=[]
y_train=[]
Build_Train_Set(prods_manipulated,1,minlen)
Build_Train_Set(prods_benign,-1,100)


# conver to np array
x_train=np.array(x_train)
y_train=np.array(y_train)


#......shuffled input samples .........
#x_train, y_train=unison_shuffled_copies(x_train,y_train)


print (y_train.shape)
print (x_train.shape)


def ReadData_test(inputFile, manipulated_items, benign_items):
    global prods_test_manipulated, prods_test_benign
    prods_test={}
    with open(inputFile) as f:
        for line in f:
            line=line.strip().split(',')
            if str(line[0]) not in usedTargetInTrain:
                iD, rate, time=line[0], line[2], line[3]
                if iD not in prods_test:
                    prods_test[iD] = []
                prods_test[iD].append((int(float(rate)), float(time)))

                    
        for prod in prods_test:
            prods_test[prod] = sorted(prods_test[prod], key=operator.itemgetter(1))
            if prod in manipulated_items:
                prods_test_manipulated[prod]=prods_test[prod]
            else:
                prods_test_benign[prod]=prods_test[prod]


prods_test_manipulated={}
prods_test_benign={}
ReadData_test('Expanded_Benign_simple.txt', manipulated_items, benign_items)
ReadData_test('Primary_simple.txt', manipulated_items, benign_items)



#...........Chunking with overlap ...........
def Chunking(iterable):
    windows = list(mit.windowed(iterable, n=maxlen, step=maxlen-overlap))
    l=len(windows)
    if None in set(windows[l-1]):
        X=[x for x in windows[l-1] if x is not None]
        windows[l-1]=X
    return windows


def Build_Test_Set(prods, flag):
    global x_test, y_test, trackOfChunks
    for prod in prods:
        l=len(prods[prod])
        x=[j[0]-2 if j[0]!=1 else j[0]-1 for j in prods[prod]]
        if flag==1:
            if l<=maxlen:
                x=np.array(x)
                y_test.append(1)
                x_test.append(x)
                trackOfChunks.append(1)
            else:
                X=Chunking(x)
                trackOfChunks.append(len(X))
                for chunk in X:
                    tmp=np.array(chunk)
                    x_test.append(tmp)
                    y_test.append(1)
                
        elif flag==-1:
            if l<=maxlen:
                x_test.append(x)
                y_test.append(0)
            else:
                x_test.append(x[l-maxlen:])
                y_test.append(0)






trackOfChunks=[]
x_test=[] # 30% training and 70% testing
y_test=[]


Build_Test_Set(prods_test_manipulated,1)
Build_Test_Set(prods_test_benign, -1)


x_test=np.array(x_test)
y_test=np.array(y_test)


#......... RNN model ..........
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = np.array(y_train)

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen, init=initial))
model.add(Bidirectional(LSTM(hidden_size, init=initial)))# memory/hidden units
model.add(Dropout(dropRate))
model.add(Dense(1, activation='sigmoid', init=initial))# first parameter is output shape 

optimizer = optimizers.Adam(lr=learn_rate, beta_1 = 0.9, beta_2 = 0.999)
model.compile(loss='binary_crossentropy', optimizer=optimizer , metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[x_train, y_train])





#....... Evaluation .....
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
output_array = model.predict(x_test)

prediction=[]
l=len(output_array)
for i in range(l):
    if output_array[i][0]>=0.5:
        prediction.append(1)
    else:
        prediction.append(0)


#..... at least one segment detected as manipulated then the item is labeled as manipulated
detected, count=0, 0
WhatAreNotDetected=[]
l=len(trackOfChunks) #number of manipulated products 
for i in range(l):
    T=True
    for j in range(trackOfChunks[i]):
        if prediction[count]==1 and T==True:
            detected+=1
            T=False
        count+=1
    if T==True:
        WhatAreNotDetected.append(i)     
        
notDetected=len(WhatAreNotDetected)

benign_pred=prediction[len(prediction)-len(prods_test_benign):] # Prediction of benign products

pred_test, actual_test=[], []
for i in range(detected):
    pred_test.append(1)
    actual_test.append(1)
for i in range (notDetected):
    pred_test.append(0)
    actual_test.append(1)
for i in range(benign_pred.count(0)):
    pred_test.append(0)
    actual_test.append(0)
for i in range(len(prods_test_benign)-benign_pred.count(0)):
    pred_test.append(1)
    actual_test.append(0)


# ..... Accuracy .......
Acc=(detected+benign_pred.count(0))*1.0/(len(prods_test_manipulated)+len(prods_test_benign))
print "Accuracy:"
print Acc

print (classification_report(actual_test, pred_test))


# #........ Keras models can be used in scikit-learn by wrapping them with the KerasClassifier or KerasRegressor class.
# #............. Parameter Tuning .............
# # https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# # Function to create model, required for KerasClassifier
# def create_model(init_mode='uniform'):
#     model = Sequential()
#     model.add(Embedding(max_features, embedding_size, input_length=maxlen, init=init_mode))
#     model.add(Bidirectional(LSTM(hidden_size, init=init_mode)))# 64 is memory/hidden units
#     model.add(Dropout(dropRate))
#     model.add(Dense(1, activation='sigmoid', init=init_mode))# first parameter is output shape 
    
#     #Comppile model
#     optimizer = optimizers.Adamax(lr=learn_rate, beta_1 = 0.9, beta_2 = 0.999)
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
#     return model




# print(len(x_train), 'train sequences')

# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# y_train = np.array(y_train)





# np.random.seed(5)

# # create model
# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

# # define the grid search parameters
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# param_grid = dict(init_mode=init_mode)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

# grid_result = grid.fit(x_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))













