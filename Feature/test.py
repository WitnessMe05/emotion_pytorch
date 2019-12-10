import gzip, pickle
import subprocess
import pandas as pd 


tt=pd.read_csv("/home/gnlenfn/remote/pytorch_emotion/Feature/IS10_paraling.csv").sort_values(by="name")
cl = pd.read_csv("/home/gnlenfn/remote/pytorch_emotion/Feature/emotion_classes.csv",header=None)
cl.columns = ['index', 'EMOTION']
cl = cl.drop(['index'], axis=1)
tt = tt.drop(['class'], axis=1).reset_index(drop=True)
res = pd.concat([tt,cl], axis=1).reset_index(drop=True)

# Set sesseion and type columns
sess=[]
rec=[]
for i in range(res.shape[0]):
    sess.append(res['name'][i][1:6])
    rec.append(res['name'][i][8:13])
res["Session"] = sess
res["Type"] = rec
print(res)

# Save feature file
#res.to_pickle(featfile.split(".")[0] + ".pkl", compression='gzip')