# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:34:27 2019

@author: umistry
"""

import cv2
import os
from skimage.feature import local_binary_pattern
import sys
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics import accuracy_score

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

category_list=[]
test_images=[]
test_labels=[]
test_data=[]

testpath=r"data\images_test"
filenames = os.listdir(testpath)
print('Getting testing Files...')
for img in filenames:
    if (os.path.isdir(testpath+"\\"+img)):
        category_list.append(img)
        flag=category_list.index(img)
        imgfilename=os.listdir(testpath+"\\"+img)
        print(img)
        for doc in tqdm(imgfilename): 
            test_images.append(testpath+"\\"+img+"\\"+doc)
            test_labels.append(flag)
            

print('\n Extract Features from testing Files...')
for test_image in tqdm(test_images):
        im = cv2.imread(test_image)
        #conver 
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        radius = 3
        #number of points to be considered as neighbourers 
        no_points = 8 * radius
        #uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        hist = getfeatures(lbp)    
        #Append histogram to test_data
        test_data.append(hist)

if 'category_list' not in globals():
    with open('categorylist.pkl', 'rb') as fid:
        category_list = pickle.load(fid)
if 'knmodel' not in globals():
    with open('knmodel.pkl', 'rb') as fid:
        knmodel = pickle.load(fid)
#if 'svmmodel' not in globals():
#    with open('svmmodel.pkl', 'rb') as fid:
#        svmmodel = pickle.load(fid)   
knmodel_pridiction=[]
#svm_pridiction=[]

for i in test_data:
    knmodel_pridiction.append(knmodel.predict([i]))
#    svm_pridiction.append(svmmodel.predict([i]))
    
knnacc = accuracy_score(test_labels, knmodel_pridiction)
print("knmodel_pridiction ::  ",knnacc)
#svmacc = accuracy_score(test_labels, svm_pridiction)
#print("svm_pridiction :: ",svmacc)


for i in range(len(test_images)):
    print("True Language ::  ",category_list[test_labels[i]],"  ",category_list[(knmodel_pridiction[i])[0]],"  :: knmodel pridiction")

