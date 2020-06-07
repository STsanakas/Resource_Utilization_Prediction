mypath='/home/user/Documents/Resource_Utilization_Prediction/' #Home Folder of the project
mydatasetfile=mypath+'Datasets/dataset.csv'
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
from os import path
import numpy as np
import pandas as pd
import time
import math
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

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
  dataset,datasetY=split_sequences(dataset,1,1)  
  dataset=dataset.reshape(dataset.shape[0],dataset.shape[2])
  datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])  
  train_size = int(len(dataset) * 0.67)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  trainY, testY = datasetY[0:train_size,:], datasetY[train_size:len(dataset),:]
  return train, trainY, test, testY, scaler

def evaluate(model, X_test, y_test, scaler):
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


def main():    
	start=time.time()
    best_mse=1.0
    best_mae=1.0
    X_train, y_train, X_test, y_test, scaler = data_preparation(mydatasetfile)    
    learning_rate=[0.05, 0.15, 0.30]
    max_depth=[ 3, 5, 10, 15]
    min_child_weight=[ 3, 7]
    gamma=[ 0.0, 0.2, 0.4]
    colsample_bytree=[ 0.3, 0.7] 
    start = time.time()
    mytime=0
    trial=0
    print('	percentage  mse          mae          rmse')
    total_tries=len(learning_rate)*len(max_depth)*len(min_child_weight)*len(gamma)*len(colsample_bytree)
    for i in range(len(learning_rate)):
        for j in range(len(max_depth)):
            for k in range(len(min_child_weight)):
                for l in range(len(gamma)):
                    for m in range(len(colsample_bytree)):    						                   
                        xg_reg = MultiOutputRegressor(xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = colsample_bytree[m], learning_rate = learning_rate[i], max_depth = max_depth[j], gamma = gamma[l], min_child_weight= min_child_weight[k]))
                        trial=trial+1
                        percentage=100*(trial/total_tries)
                        best_rmse=math.sqrt(best_mse)
                        print('\r', "	%.2f %%     %.6f     %.6f     %.6f" % (percentage, best_mse, best_mae, best_rmse), end='')  
                        xg_reg.fit(X_train,y_train) 
                        mse, mae, mse_ram, mae_ram, mse_cpu, mae_cpu=evaluate(xg_reg, X_test, y_test, scaler)
                        if (mse<best_mse):
                          best_mse=mse
                          best_mae=mae
                          best_mse_ram=mse_ram
                          best_mae_ram=mae_ram
                          best_mse_cpu=mse_cpu
                          best_mae_cpu=mae_cpu
                          infS, infB = inference(xg_reg, X_test)
    print('')
    print('done fitting')                        
    dataset_name = mydatasetfile.split("/")[-1]
    mytime = round(time.time()-start)
    print('XGBoost results using', dataset_name, 'as dataset')
    print('Best mse: %.6f	mae: %.6f	rmse: %.6f	training time: %.0f s' % (best_mse, best_mae, best_rmse, mytime))
    print('Single inference time: %.3f s	batch inference time: %.3f s ' % (infS, infB))
    print('RAM mse: %.6f	RAM mae: %.6f	RAM rmse: %.6f' % (best_mse_ram, best_mae_ram, math.sqrt(best_mse_ram)))
    print('CPU mse: %.6f	CPU mae: %.6f	CPU rmse: %.6f' % (best_mse_cpu, best_mae_cpu, math.sqrt(best_mse_cpu)))
    print('XGBoost,',mytime,',',best_rmse,',',best_mae,',',infS,',',infB,',',math.sqrt(best_mse_cpu),',',best_mae_cpu,',',math.sqrt(best_mse_ram),',',best_mae_ram, file=open(mypath+'results.csv','a'))
    
main()                                                    
