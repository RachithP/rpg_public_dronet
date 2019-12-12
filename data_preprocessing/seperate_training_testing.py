#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:09:02 2017
@author: ana-rpg, Ashwin Varghese Kuruttukulam
"""
import glob
import numpy as np
import re
import os
import shutil
from copy import deepcopy
# Path to the data extracted from the Udacity dataset


def extractInfoFromFile(file_name):
    steer_stamps = []
    # Read file and extract time stamp
    try:
       steer_stamps = np.loadtxt(file_name, usecols=1, delimiter=',', skiprows=1, dtype=int)
    except:
        print('cant open'+file_name)
    return steer_stamps


def getMatching(array1, array2):
    match_stamps = []
    match_idx = []
    for i in array1:
        dist = abs(i - array2)
        idx = np.where(dist == 0)[0]
        match_stamps.append(array2[idx])
        match_idx.append(idx)
    return match_stamps, match_idx


def getSyncSteering(fname, idx):
    mat = []
    try:
        mat = np.loadtxt(fname, usecols=(6,7,8,9,10,11), skiprows=1, delimiter=',')
        mat = mat[idx,:]
    except:
        print('cant open'+fname)
    return mat

def getInterpolated(fname):
    mat = []
    try:
        interfile = open(fname, 'r')
        for line in interfile.readlines():
            mat.append(line)
    except:
        print('cant open'+fname)
    return mat


# For every bag...
a = [1]
for temp in a:
    exp = '../datasets/training/HMB_6'
    # Read images
    images = [os.path.basename(x) for x in glob.glob(exp + "/images/*.png")]
    im_stamps = []
    for im in images:
        stamp = int(re.sub(r'\.png$', '', im))
        im_stamps.append(stamp)
    im_stamps = np.array(sorted(im_stamps))

    # Extract time stamps from steerings
    file_name = exp + "/interpolated.csv"
    steer_stamps = extractInfoFromFile(file_name)

    # Time-stamp matching between images and steerings
    match_stamp, match_idx = getMatching(im_stamps, steer_stamps)
    
    match_idx = np.array(match_idx)
    match_idx = match_idx[:,0]

    interLines = getInterpolated(file_name)
    # Create file if it doesnt exist
    directory = exp+'/testing/images'
    if not os.path.exists(directory):
            os.makedirs(directory)
    inteFileName = exp+'/testing/interpolated.csv'
    if not os.path.exists(os.path.dirname(inteFileName)):
        try:
            os.makedirs(os.path.dirname(inteFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    newInterFile = open(inteFileName,'w') 
    idx = 0
    originalInterLines = deepcopy(interLines)
    while(idx<len(match_stamp)):
    #while(idx<30):
        time_stamp = match_stamp[idx]
        print(idx)
        index = match_idx[idx]
        index += 1
        newPath = shutil.copy(exp+'/images/'+str(time_stamp[0])+'.png', exp+'/testing/images')
        print(index)
        print(idx)
        print(exp+'/images/'+str(time_stamp[0])+'.png')
        print('\n')
        try:
            os.remove(exp+'/images/'+str(time_stamp[0])+'.png')
        except OSError as e: # name the Exception `e`
            print("Failed with:", e.strerror) # look what it says
            print("Error code:", e.code) 
        interLines.remove(originalInterLines[index])
        newInterFile.write(originalInterLines[index])
        idx += 5
    file_name = exp + "/interpolated.csv"
    
    os.remove(file_name)
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    newInterFile = open(file_name,'w+') 
    for line in interLines:
       newInterFile.write(line) 
    # Get matched commands
    #original_fname = exp + "/interpolated.csv"
    #sync_steer = getSyncSteering(original_fname, match_idx)

