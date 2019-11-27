import subprocess
import sys
import os
import os.path
import numpy as np
import pandas as pd
from scipy.io import arff
import load_iemocap as li
from HTK import HTKFile
import matplotlib.pyplot as plt
import pickle, gzip

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            #print("Directory is already exists")
            pass
    except OSError:
        print ('Error: Creating directory. ' +  directory)

opensmile_path = os.path.expanduser("/home/gnlenfn/Downloads/opensmile-2.3.0/")
iemocap_path = "/home/gnlenfn/doc/IEMOCAP/"

sess = ["Session1", "Session2", "Session3", "Session4", "Session5"]
df_lab = pd.DataFrame()

for sessions in sess:
    wav_files_path = os.path.join(iemocap_path, sessions + "/sentences/wav/")
    wavfiles_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(wav_files_path)) for f in fn]
    wavfiles = np.array(wavfiles_list)
    
    label_files_path = os.path.join(iemocap_path, sessions + "/dialog/EmoEvaluation/")
    labfiles = os.listdir(label_files_path)

    realwavfiles = li.returnrealfiles(wavfiles)
    reallabfiles = li.returnrealfiles(labfiles)
    #print(realwavfiles)
    for wavfile_index in range(len(realwavfiles)):
        wav_filename = str(realwavfiles[wavfile_index])
        print(realwavfiles[wavfile_index])
        if wav_filename.split("/")[-1].split(".")[0][7:12] == 'scrip':
            wav_fullpath = os.path.join(wav_files_path, wav_filename)
            #print(wav_fullpath)
            lab_fullpath = li.find_matching_label_file(wav_filename, reallabfiles, label_files_path)
            #print(lab_fullpath)

            opensmile_conf = os.path.join(opensmile_path + "config/IS09_emotion.conf") # choose configuration file
            result_path = "./test/"#+sessions+"/"+wav_filename.split("/")[-2]
            createFolder(result_path)
            featfile = result_path + 'IS09_emo.arff'
            command = "SMILExtract -I {input} -C {conf} -O {output} -instname {name}".format(input=wav_fullpath, conf=opensmile_conf, output=featfile, name=wav_fullpath.split("/")[-1].split(".")[0])
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            
            n=6
            labels = li.readlabtxt(lab_fullpath, n)
            df_lab = df_lab.append(labels)

        else: # impro 일때!
            wav_fullpath = os.path.join(wav_files_path, wav_filename)
            lab_fullpath = li.find_matching_label_file(wav_filename, reallabfiles, label_files_path)

            opensmile_conf = os.path.join(opensmile_path + "config/IS09_emotion.conf") # choose configuration file
            result_path = "./test/"#+sessions+"/"+wav_filename.split("/")[-2]
            createFolder(result_path)
            featfile = result_path + 'IS09_emo.arff'
            command = "SMILExtract -I {input} -C {conf} -O {output} -instname {name}".format(input=wav_fullpath, conf=opensmile_conf, output=featfile, name=wav_fullpath.split("/")[-1].split(".")[0])
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

            if sessions in ["Session1", "Session2", "Session3"]:
                n=8
                labels = li.readlabtxt(lab_fullpath,n)
                df_lab = df_lab.append(labels)
            else:
                n=6
                labels = li.readlabtxt(lab_fullpath, n)
                df_lab = df_lab.append(labels)


    print("End of {}!".format(sessions))


com = "cd test && python arff2csv.py"
subprocess.run(com, shell=True)
df_lab = df_lab.drop_duplicates().reset_index(drop=True)
df_lab = df_lab.sort_values(by='TURN_NAME')
cl = df_lab["EMOTION"].reset_index(drop=True)

# Set emotion class column
tt=pd.read_csv("./test/IS09_emo.csv").sort_values(by="name")
tt = tt.drop(['class'], axis=1).reset_index(drop=True)
res = pd.concat([tt,cl], axis=1).reset_index(drop=True)

# Set sesseion and type columns
sess=[]
rec=[]
for i in range(res.shape[0]):
    sess.append(res['name'][i][1:6])
    rec.append(res['name'][i][8:13])
new_d["Session"] = sess
new_d["Type"] = rec

# Save feature file
res.to_pickle(result_path+"IS09_emotion_feature.pkl", compression='gzip')