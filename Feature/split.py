import pandas as pd
import numpy as np
import gzip, pickle

with gzip.open("/home/gnlenfn/data/remote/pytorch_emotion/Feature/feature.pkl") as ifp:
    df = pickle.load(ifp)
em = pd.read_csv("/home/gnlenfn/data/remote/pytorch_emotion/Feature/emotion_classes.csv")

need = list(df.columns)[2:-1]

leng = em.shape[0]
emo = ['neu', 'ang', 'sad', 'hap', 'exc']
x = []
y = []
ses = []
tp = []
for i in range(leng):    
    tmp_emo = em.EMOTION[i] 
    tmp = df[df.name == em.TURN_NAME[i]][need]
    arr = tmp.to_numpy()
    
    sess = em.TURN_NAME[i][0:5]
    ty = em.TURN_NAME[i][7:12]
    
    if tmp_emo in emo:        
        x.append(arr)
        y.append(tmp_emo)
        ses.append(sess)
        tp.append(ty)
    if i % 1000 == 0:
        print("{} / {}".format(i, leng))

data = np.array(x)
label = np.array(y)
ses = np.array(ses)
tp = np.array(tp)

np.savez("/home/gnlenfn/data/remote/pytorch_emotion/Feature/data_save", x=data, y=label, s=ses, t=tp)
