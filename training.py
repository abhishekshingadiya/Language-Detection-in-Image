# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# import cv
import matplotlib.pyplot as plt
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# Utility package -- use pip install cvutils to install
import cvutils
# To read class from file
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
from collections import Counter
import pandas as pd
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


# initiate the parser
#trainpath=sys.argv[1]
trainpath=r"data\images_train"

#temppath='W:\\Python Work\\DocClass\\class'
#############
#train data#
############
#trainpath= 'W:\\Python Work\\DocClass\\class\\train'
category_list=[]
train_images=[]
train_labels=[]
train_data=[]

filenames = os.listdir(trainpath)
print('Getting Training Files...')
for img in filenames:
    if (os.path.isdir(trainpath+"\\"+img)):
        category_list.append(img)
        flag=category_list.index(img)
        imgfilename=os.listdir(trainpath+"\\"+img)
        print(img)
        for doc in tqdm(imgfilename): 
            train_images.append(trainpath+"\\"+img+"\\"+doc)
            train_labels.append(flag)
            

print('\n Extract Features from Training Files...')
for train_image in tqdm(train_images):
        im = cv2.imread(train_image)
        #conver 
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        radius = 3
        #number of points to be considered as neighbourers 
        no_points = 8 * radius
        #uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        hist = getfeatures(lbp)    
        #Append histogram to train_data
        train_data.append(hist)
	


model = KNeighborsClassifier(n_neighbors=3,n_jobs=3)
model.fit(train_data, train_labels)

# save the category list
with open('categorylist.pkl', 'wb') as fid:
    pickle.dump(category_list, fid)  
print("\n We save Categorylist for you future use...you know we are smart....")  


print("\n K-Neighbors Classifier Training Start...")
model = KNeighborsClassifier(n_neighbors=3,n_jobs=3)
model.fit(train_data, train_labels)
# save the classifier
with open('knmodel.pkl', 'wb') as fid:
    pickle.dump(model, fid)  
print("\n K-Neighbors Classifier Model Save!!")  


  
from sklearn import svm
print("\n SVM Classifier Training Start...")
clf_svm = svm.SVC(kernel='linear', probability=True)
clf_svm.fit(train_data, train_labels)
# save the classifier
with open('svmmodel.pkl', 'wb') as fid:
    pickle.dump(clf_svm, fid)  
print("\n SVM Classifier Model Save!!")

