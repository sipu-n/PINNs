import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import matplotlib.pyplot as plt
model=keras.Sequential([keras.layers.Flatten(),
                        keras.layers.Dense(100,acivation='relu'),
                        keras.layers.Dense(100,activation='relu'),
                        keras.layers.Dense(10,acivation='relu'),
                        keras.layers.Dense(1,activation='relu')])
a=pd.read_excel('unique.xlsx')


    