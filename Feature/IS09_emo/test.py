import pandas as pd 
import gzip, pickle

with gzip.open("IS09_emotion_feature.pkl") as f:
    d = pickle.load(f)