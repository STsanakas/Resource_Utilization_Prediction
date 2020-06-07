from os import path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import time
import random
import statistics
import pandas
import math
import csv
import random
import logging
from functools import reduce
from operator import add
from tqdm import tqdm
import geopy.distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras



### Customization Section. Feel free to experiment with these variables ###
generations = 10  # Number of times to evole the population.
population = 20  # Number of networks in each generation.
mypath='/home/user/Documents/Resource_Utilization_Prediction/' #Home Folder of the project
mydatasetfile=mypath+'Datasets/dataset.csv'
best_score=1
myverbose=0
EarlyStopper = EarlyStopping(patience=1, monitor='loss', mode='min')
### End of Customization Section ###


def hasLSTM(model):
	firstlayer=model.layers[0].name[:4]
	if(firstlayer=='lstm'):
		answer=True
	else:
		answer=False
	return answer	


def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		if out_end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def data_preparation(datasetfile):
  dataframe = pd.read_csv(datasetfile, engine='python')  
  dataset = dataframe.values
  dataset = dataset.astype('float32')
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(dataset)
  dataset = scaler.transform(dataset)
  return dataset, scaler
  

def compile_model(network, dataset):
  nb_neurons=[]
  activation=[]
  dropout=[]
  nb_layers = network['nb_layers']
  lstms=network['lstms']
  lookback=network['lookback']
  implementation1=network['implementation1']
  units1=network['units1']
  lstm_activation1=network['lstm_activation1']
  recurrent_activation1=network['recurrent_activation1']
  implementation2=network['implementation2']
  units2=network['units2']
  lstm_activation2=network['lstm_activation2']
  recurrent_activation2=network['recurrent_activation2']
  nb_neurons.append(network['nb_neurons1'])
  nb_neurons.append(network['nb_neurons2'])
  nb_neurons.append(network['nb_neurons3'])
  nb_neurons.append(network['nb_neurons4'])
  nb_neurons.append(network['nb_neurons5'])
  activation.append(network['activation1'])  
  activation.append(network['activation2'])
  activation.append(network['activation3'])
  activation.append(network['activation4'])
  activation.append(network['activation5'])
  dropout.append(network['dropout1'])  
  dropout.append(network['dropout2'])  
  dropout.append(network['dropout3'])  
  dropout.append(network['dropout4'])  
  dropout.append(network['dropout5'])  
  optimizer = network['optimizer']
  learning_rate = network['learning_rate']
  batch_size = network['batch_size']
  epochs = network['epochs']
  model = Sequential()
  if(lstms==1):
	  model.add(LSTM(units1, input_shape=(lookback, dataset.shape[1]), activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1))
	  for i in range(nb_layers):
		  model.add(Dense(nb_neurons[i], activation=activation[i]))
		  model.add(Dropout(dropout[i]))		  
  elif (lstms==2):
	  model.add(LSTM(units1, input_shape=(lookback, dataset.shape[1]), activation=lstm_activation1, recurrent_activation=recurrent_activation1, implementation=implementation1, return_sequences=True))
	  model.add(LSTM(units2, activation=lstm_activation2, recurrent_activation=recurrent_activation2, implementation=implementation2))
	  for i in range(nb_layers):
		  model.add(Dense(nb_neurons[i], activation=activation[i]))
		  model.add(Dropout(dropout[i]))		  		  
  elif(lstms==0):
	  model.add(Dense(nb_neurons[0], input_shape=(dataset.shape[1],), activation=activation[0]))
	  model.add(Dropout(dropout[0]))		  	  
	  for i in range(nb_layers):
		  if i>0:
			  model.add(Dense(nb_neurons[i], activation=activation[i]))
			  model.add(Dropout(dropout[i]))			  	  
  model.add(Dense(dataset.shape[1]))    
  model.compile(loss='mean_squared_error', optimizer=optimizer(learning_rate=learning_rate))
  if hasLSTM(model):
    dataset,datasetY=split_sequences(dataset,lookback,1)  
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])  
  else:
    dataset,datasetY=split_sequences(dataset,1,1)  
    dataset=dataset.reshape(dataset.shape[0],dataset.shape[2])
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])  
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  trainY, testY = datasetY[0:train_size,:], datasetY[train_size:len(dataset),:]  
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=myverbose, callbacks=[EarlyStopper])  
  return model, testX, testY

def evaluate(model, testX, testY):
  global best_score
  testScore = model.evaluate(testX, testY, verbose=0)  
  lb=testX.shape[1]
  if(best_score>testScore):
    best_score=testScore    
    model.save(mypath+'Models/best.h5')
    print(lb, file=open(mypath+'Models/lookback.txt','w'))
  return testScore
  
def metrics(model, X_test, y_test, scaler):
	prediction=model.predict(X_test)	
	mse = mean_squared_error(y_test, prediction)
	mae = mean_absolute_error(y_test, prediction)
	y_test=scaler.inverse_transform(y_test)
	prediction=scaler.inverse_transform(prediction)
	mse_ram = mean_squared_error(y_test[:,10], prediction[:,10])
	mae_ram = mean_absolute_error(y_test[:,10], prediction[:,10])
	mse_cpu = mean_squared_error(y_test[:,4], prediction[:,4])
	mae_cpu = mean_absolute_error(y_test[:,4], prediction[:,4])
	return mse, mae, mse_ram, mae_ram, mse_cpu, mae_cpu

def inference(model, X):
	i=random.randrange(len(X))	
	startS=time.time()	
	prediction=model.predict(X[[i]])	
	timeS=time.time()-startS
	i=random.randrange(len(X)-100)
	startB=time.time()
	prediction=model.predict(X[i:i+101])
	timeB=time.time()-startB
	return timeS, timeB
  
def train_and_score(network, dataset, scaler):
  #dataset, scaler = data_preparation()
  model, testX, testY = compile_model(network, dataset)
  model_score = evaluate(model, testX, testY)
  return model_score


class Network():

  def __init__(self, nn_param_choices=None):
	  self.accuracy = 0.
	  self.nn_param_choices = nn_param_choices
	  self.network = {}  # (dic): represents MLP network parameters

  def create_random(self):
	  for key in self.nn_param_choices:
		  self.network[key] = random.choice(self.nn_param_choices[key])

  def create_set(self, network):
	  self.network = network

  def train(self, dataset, scaler):
	  if self.accuracy == 0.:
		  self.accuracy = train_and_score(self.network, dataset, scaler)
	
  def print_network(self):
	  print("Network error: %.5f" % (self.accuracy))
	  print("Trained by %s optimizer with a learning rate of %.3f" % (self.network['optimizer'], self.network['learning_rate']))
	  print("%d epochs of training with %.d batch size" % (self.network['epochs'], self.network['batch_size']))
	  if self.network['lstms']>0:
		  print("A lookback of %d timesteps was used at the input" % (self.network['lookback']))
		  print('LSTM with %d units, %s activation and %s recurrent activation'% (self.network['units1'], self.network['lstm_activation1'], self.network['recurrent_activation1']))
		  if self.network['lstms']>1:
			  print('LSTM with %d units, %s activation and %s recurrent activation'% (self.network['units2'], self.network['lstm_activation2'], self.network['recurrent_activation2']))
	  print('Dense Layer with %d neurons, and %s activation'% (self.network['nb_neurons1'], self.network['activation1']))
	  if self.network['dropout1']>0:
		  print('Dropout Layer %.1f'% (self.network['dropout1']))
	  if self.network['nb_layers']>1:
		  print('Dense Layer with %d neurons, and %s activation'% (self.network['nb_neurons2'], self.network['activation2']))
		  if self.network['dropout2']>0:
			  print('Dropout Layer %.1f'% (self.network['dropout2']))
		  if self.network['nb_layers']>2:
			  print('Dense Layer with %d neurons, and %s activation'% (self.network['nb_neurons3'], self.network['activation3']))
			  if self.network['dropout3']>0:
				  print('Dropout Layer %.1f'% (self.network['dropout3']))
			  if self.network['nb_layers']>3:
				  print('Dense Layer with %d neurons, and %s activation'% (self.network['nb_neurons4'], self.network['activation4']))
				  if self.network['dropout4']>0:
					  print('Dropout Layer %.1f'% (self.network['dropout4']))
				  if self.network['nb_layers']>4:
					  print('Dense Layer with %d neurons, and %s activation'% (self.network['nb_neurons5'], self.network['activation5']))
					  if self.network['dropout5']>0:
						  print('Dropout Layer %.1f'% (self.network['dropout5']))
	  print('Output layer with linear activation')			
			
class Optimizer():

  def __init__(self, nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
	  self.mutate_chance = mutate_chance
	  self.random_select = random_select
	  self.retain = retain
	  self.nn_param_choices = nn_param_choices

  def create_population(self, count):
	  pop = []
	  for _ in range(0, count):
		  network = Network(self.nn_param_choices)
		  network.create_random()
		  pop.append(network)
	  return pop

  @staticmethod
  def fitness(network):
	  return network.accuracy

  def grade(self, pop):
	  summed = reduce(add, (self.fitness(network) for network in pop))
	  return summed / float((len(pop)))

  def breed(self, mother, father):
	  children = []
	  for _ in range(2):
		  child = {}
		  for param in self.nn_param_choices:
			  child[param] = random.choice([mother.network[param], father.network[param]])
		  network = Network(self.nn_param_choices)
		  network.create_set(child)
		  if self.mutate_chance > random.random():
			  network = self.mutate(network)
		  children.append(network)
	  return children

  def mutate(self, network):
	  mutation = random.choice(list(self.nn_param_choices.keys()))
	  network.network[mutation] = random.choice(self.nn_param_choices[mutation])
	  return network

  def evolve(self, pop):
	  graded = [(self.fitness(network), network) for network in pop]
	  graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
	  retain_length = int(len(graded)*self.retain)
	  parents = graded[:retain_length]
	  for individual in graded[retain_length:]:
		  if self.random_select > random.random():
			  parents.append(individual)
	  parents_length = len(parents)
	  desired_length = len(pop) - parents_length
	  children = []
	  while len(children) < desired_length:
		  male = random.randint(0, parents_length-1)
		  female = random.randint(0, parents_length-1)
		  if male != female:
			  male = parents[male]
			  female = parents[female]
			  babies = self.breed(male, female)
			  for baby in babies:
				  if len(children) < desired_length:
					  children.append(baby)
	  parents.extend(children)
	  return parents



def train_networks(networks, dataset, scaler):
  pbar = tqdm(total=len(networks))
  for network in networks:
	  network.train(dataset, scaler)
	  pbar.update(1)
  pbar.close()

def generate(generations, population, nn_param_choices, dataset, scaler):
  optimizer = Optimizer(nn_param_choices)
  networks = optimizer.create_population(population)
  for i in range(generations):
	  print("***Doing generation %d of %d***" % (i + 1, generations))
	  train_networks(networks, dataset, scaler)
	  networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
	  if i != generations - 1:
		  networks = optimizer.evolve(networks)
  networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
  print_networks(networks[:1])


def print_networks(networks):
  print('-'*80)
  for network in networks:
	  network.print_network()
  print('-'*80)

def main():  


  nn_param_choices = {
	'lstms':[1,2],
	'lookback':[1, 2, 3, 4, 5],
	'implementation1':[1,2],
	'units1':[2,8,16,32,64,128],
	'lstm_activation1':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'recurrent_activation1':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'implementation2':[1,2],
	'units2':[2,8,16,32,64,128],
	'lstm_activation2':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'recurrent_activation2':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
	'nb_layers': [1, 2, 3, 4, 5],
    'nb_neurons1': [2,8,16,32,64,128],    
	'activation1': ['tanh', 'sigmoid', 'linear', 'relu'],
	'dropout1':[0,0.1,0.2,0.3,0.4,0.5],
    'nb_neurons2': [2,8,16,32,64,128],    
    'activation2': ['tanh', 'sigmoid', 'linear', 'relu'],
	'dropout2':[0,0.1,0.2,0.3,0.4,0.5],    
    'nb_neurons3': [2,8,16,32,64,128],    
    'activation3': ['tanh', 'sigmoid', 'linear', 'relu'],
    'dropout3':[0,0.1,0.2,0.3,0.4,0.5],
    'nb_neurons4': [2,8,16,32,64,128],    
    'activation4': ['tanh', 'sigmoid', 'linear', 'relu'],
    'dropout4':[0,0.1,0.2,0.3,0.4,0.5],
    'nb_neurons5': [2,8,16,32,64,128],
    'activation5': ['tanh', 'sigmoid', 'linear', 'relu'],
    'dropout5':[0,0.1,0.2,0.3,0.4,0.5],
	'optimizer': [keras.optimizers.RMSprop, keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.Adagrad, keras.optimizers.Adadelta, keras.optimizers.Adamax, keras.optimizers.Nadam],
	'learning_rate':[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
	'batch_size':[1,10,50,100,500,1000],
	'epochs':[20,50,100,200],	
	
  }
  dataset, scaler = data_preparation(mydatasetfile)
  generate(generations, population, nn_param_choices, dataset, scaler)


start=time.time()
main()
mytime=time.time()-start


mymodel=keras.models.load_model(mypath+'Models/best.h5')
mydatasetfile=mypath+'Datasets/dataset.csv'
dataset_name = mydatasetfile.split("/")[-1]
dataset, scaler = data_preparation(mydatasetfile)
text_file = open(mypath+'Models/lookback.txt', 'r')
lookback= int(text_file.read(1))
if hasLSTM(mymodel):
    dataset,datasetY=split_sequences(dataset,lookback,1) 
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])   
else:
    dataset,datasetY=split_sequences(dataset,1,1)  
    dataset=dataset.reshape(dataset.shape[0],dataset.shape[2])
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainY, testY = datasetY[0:train_size,:], datasetY[train_size:len(dataset),:]
mse, mae, mse_ram, mae_ram, mse_cpu, mae_cpu = metrics(mymodel, testX, testY, scaler)
infS, infB = inference(mymodel, testX)

print('Genetic Algorithm results using', dataset_name, 'as dataset')
print('Best mse: %.6f	mae: %.6f	rmse: %.6f	training time: %.0f s' % (mse, mae, math.sqrt(mse), mytime))
print('Single inference time: %.3f s	batch inference time: %.3f s ' % (infS, infB))
print('RAM mse: %.6f	RAM mae: %.6f	RAM rmse: %.6f' % (mse_ram, mae_ram, math.sqrt(mse_ram)))
print('CPU mse: %.6f	CPU mae: %.6f	CPU rmse: %.6f' % (mse_cpu, mae_cpu, math.sqrt(mse_cpu)))
