import glob
import numpy as np
import re
import os, sys
import subprocess
import evaluation
import re
import logz_test as logz

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#folder = "./model/three_block_model"
folder = sys.argv[1]

steering_dict = []
collision_dict = []

logz.configure_output_dir(folder)

for infile in sorted(glob.glob(folder+"/model_weights*"), key=numericalSort):
    #print("Current File Being Processed is: " + infile)
    infile = infile.replace(folder+"/",'')
    #print("=================After replacing", infile)
    steering, collision = evaluation.main(["evaluation.py","--experiment_rootdir=model/three_block_model", "--weights_fname="+infile, "--test_dir=../datasets/evaluation_test"])
    print("Steering--------:", steering)
    print("Collision-------:", collision)
    print(collision[0]['ave_accuracy'])
    logz.log_tabular('classification accuracy', collision[0]['ave_accuracy'])
    logz.log_tabular('RMSE', steering[0]['rmse'])
    logz.log_tabular('EVA', steering[0]['evas'])
    logz.log_tabular('F-Score', collision[0]['f_score'])
    logz.dump_tabular()
   # steering_dict.append(steering)
   # collision_dict.append(collision)
    
#print("STEERING: ", steering)
#print("COLLISION: ", collision)
