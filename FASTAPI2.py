# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:23:07 2022

@author: 39333
"""
import nest_asyncio
import pickle
nest_asyncio.apply()
import pandas as pd
from fastapi import FastAPI
import uvicorn
app = FastAPI(title='benza',
    description='MACHINE-LEARNING_NODEL FOR FORECAST benza')
model = pickle.load(open('modelfuel.pkl','rb'))
# Define predict function
@app.post('/predict')
def predict(ENGINESIZE,CYLINDERS,FUELCONSUMPTION_COMB):
    data = pd.DataFrame([[ENGINESIZE,CYLINDERS,FUELCONSUMPTION_COMB]])
    data.columns = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']
    predictions = (model.predict(data)) 
    return {'prediction': (predictions[0])}

if __name__ == '__main__':
    uvicorn.run(app, port=8081)
