# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:34:27 2019

@author: umistry
"""

import cv2
from skimage.feature import local_binary_pattern
import sys
import numpy as np
from tqdm import tqdm
import pickle



def getfeatures(lbp):
	features = np.zeros(256,dtype=float)
	length = lbp.shape[0]
	breadth = lbp.shape[1]
	for x in range(length):
		for y in range(breadth):
			features[int(lbp[x,y])]+=1.0
	for x in range(256):
		features[x] = features[x]/(length*breadth)	
	return features


train_image="\""+sys.argv[1]+"\""
print("We get your file.\nNow, just wait")
im = cv2.imread(train_image)
#conver 
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
radius = 3
#number of points to be considered as neighbourers 
no_points = 8 * radius
#uniform LBP is used
lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
hist = getfeatures(lbp) 
hist=np.array(hist)   
test_data=hist
tlist=[test_data]

if 'category_list' not in globals():
    with open('categorylist.pkl', 'rb') as fid:
        category_list = pickle.load(fid)
if 'knmodel' not in globals():
    with open('knmodel.pkl', 'rb') as fid:
        knmodel = pickle.load(fid)
#if 'svmmodel' not in globals():
#    with open('svmmodel.pkl', 'rb') as fid:
#        svmmodel = pickle.load(fid)   
    
knmodel_pridiction = knmodel.predict(tlist)
#svm_pridiction = svmmodel.predict(tlist)

print("knmodel_pridiction ::  ",category_list[knmodel_pridiction[0]])
#print("svm_pridiction :: ",category_list[svm_pridiction[0]])

