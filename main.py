import json
import pickle
from nltk_functions import *
import numpy as np
import random
from tensorflow.keras.models import load_model

with open('data.json' , 'r') as f:
    data= json.load(f)

words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
model = load_model('chatbot_model.h5')





