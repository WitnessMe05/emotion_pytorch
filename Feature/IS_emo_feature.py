import subprocess
import sys
import os
import os.path
import numpy as np
import pandas as pd
from scipy.io import arff
import load_iemocap as li
import matplotlib.pyplot as plt
import pickle, gzip
from arff2pandas import a2p

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            #print("Directory is already exists")
            pass
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def stp(char):
    return char.strip("'")

opensmile_path = os.path.expanduser("/home/gnlenfn/Downloads/opensmile-2.3.0/")
iemocap_path = "/home/gnlenfn/data/corpus/IEMOCAP/"

sess = ["Session1", "Session2", "Session3", "Session4", "Session5"]
df_lab = pd.DataFrame()
ct = 0
for sessions in sess:
    
    wav_files_path = os.path.join(iemocap_path, sessions + "/sentences/wav/")
    wavfiles_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(wav_files_path)) for f in fn]
    wavfiles = np.array(wavfiles_list)
    
    label_files_path = os.path.join(iemocap_path, sessions + "/dialog/EmoEvaluation/")
    labfiles = os.listdir(label_files_path)

    realwavfiles = li.returnrealfiles(wavfiles)
    reallabfiles = li.returnrealfiles(labfiles)
    for wavfile_index in range(len(realwavfiles)):
        wav_filename = str(realwavfiles[wavfile_index])
        #print(realwavfiles[wavfile_index])
        if wav_filename.split("/")[-1].split(".")[0][7:12] == 'scrip':
            wav_fullpath = os.path.join(wav_files_path, wav_filename)
            print(wav_fullpath)
            lab_fullpath = li.find_matching_label_file(wav_filename, reallabfiles, label_files_path)
            #print(lab_fullpath)

            opensmile_conf = os.path.join(opensmile_path + "config/IS09_emotion.conf") # choose configuration file
            result_path = "/home/gnlenfn/data/remote/pytorch_emotion/Feature/"
            createFolder(result_path)
            featfile = result_path + 'test.arff'
            command = "SMILExtract -I {input} -C {conf} -lldarffoutput {output} -appendarfflld 1 -instname {name}".format(input=wav_fullpath, conf=opensmile_conf, output=featfile, name=wav_fullpath.split("/")[-1].split(".")[0])
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            
            n=6
            labels = li.readlabtxt(lab_fullpath, n)
            df_lab = df_lab.append(labels)

        else: # impro 일때!
            wav_fullpath = os.path.join(wav_files_path, wav_filename)
            print(wav_fullpath)
            lab_fullpath = li.find_matching_label_file(wav_filename, reallabfiles, label_files_path)

            opensmile_conf = os.path.join(opensmile_path + "config/IS09_emotion.conf") # choose configuration file
            result_path = "/home/gnlenfn/data/remote/pytorch_emotion/Feature/"
            createFolder(result_path)
            featfile = result_path + 'test.arff'
            command = "SMILExtract -I {input} -C {conf} -lldarffoutput {output} -appendarfflld 1 -instname {name}".format(input=wav_fullpath, conf=opensmile_conf, output=featfile, name=wav_fullpath.split("/")[-1].split(".")[0])
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

            if sessions in ["Session1", "Session2", "Session3"]:
                n=8
                labels = li.readlabtxt(lab_fullpath,n)
                df_lab = df_lab.append(labels)
                #print(df_lab)
            else:
                n=6
                labels = li.readlabtxt(lab_fullpath, n)
                df_lab = df_lab.append(labels)
                #print(df_lab)
        # with open("./IS_emo.csv", 'at') as ifp:
        #     with open("/home/gnlenfn/data/remote/pytorch_emotion/Feature/test.arff", 'r') as f:
        #         if ct == 0:
        #             pass
        #         else:
        #             tmp = f.readline()
        #         file = f.read()
        #         ifp.write(file)
        #     ct += 1
        #     if ct % 100 == 0:
        #         print("########################################################################")

    print("End of {}!".format(sessions))
com = "cd /home/gnlenfn/data/remote/pytorch_emotion/Feature && python arff2csv.py"
cout = subprocess.check_output(com, shell=True, stderr=subprocess.STDOUT) 
feature_final = result_path + "test.csv"

df_lab = df_lab.drop_duplicates().reset_index(drop=True)
df_lab = df_lab.sort_values(by='TURN_NAME')
cl = df_lab[["TURN_NAME","EMOTION"]].reset_index(drop=True)
cl.to_csv(result_path + "emotion_classes.csv", header=True)
print("Emotion_classes.csv out!")

tmp = pd.read_csv("/home/gnlenfn/data/remote/pytorch_emotion/Feature/test.csv")
tmp['name'] = tmp['name'].apply(stp)

for i in range(tmp.shape[0]):
    fn = tmp.name[i]
    tmp['class'] = np.array(df_lab[ df_lab["TURN_NAME"] == fn ]["EMOTION"])[0]
    if i % 10000 == 0:
        print("{} / {}".format(i, tmp.shape[0]))
        
# for i in range(tmp.shape[0]):
#     for j in range(df_lab.shape[0]):
#         if tmp.name[i] == df_lab["TURN_NAME"][j]:
#             tmp['class'] = df_lab["EMOTION"][j]
#     if i % 10000 == 0:
#         print("{} / {}".format(i, tmp.shape[0]))
# Set emotion class column
# tt=pd.read_csv(feature_final.split(".")[0] + ".csv", sep=";"
#             ).sort_values(by="name").reset_index(drop=True)
# res = pd.concat([tt,cl], axis=1).reset_index(drop=True)

# Set sesseion and type columns
# sess=[]
# rec=[]
# for i in range(tmp.shape[0]):
#     sess.append(tmp['name'][i][1:6])
#     rec.append(tmp['name'][i][8:13])
# tmp["Session"] = sess
# tmp["Type"] = rec
# print(tmp)
#import sys
#sys.exit()
# Save feature file
tmp.to_pickle(feature_final.split(".")[0] + ".pkl", compression='gzip')