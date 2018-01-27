import numpy as np
import math
import random
import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import autoencoder_saliksyed as auto
import autoencoder_saliksyed2 as auto2
from sklearn.preprocessing import normalize
from random import randint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
################################################################################################################

e_sequence = np.loadtxt("matrix.stru")
e_sequence = np.asarray(e_sequence)

file = open("area.stru", 'r')
area_list = []
j = 0
for item in file:
    for i in item.split(" "):
        j+=1
        area_list.append(i)
area_list = np.asarray(area_list)[:-1]

colors = ['navy', 'turquoise', 'darkorange', 'orange', 'red', 'yellow', 'blue']
area_label = np.array(["EUROPE","MIDDLE_EAST","CENTRAL_SOUTH_ASIA","OCEANIA","AMERICA","AFRICA","EAST_ASIA"])

'''
#applying PCA
pca = PCA(n_components = 2)
p_data = pca.fit(e_sequence).transform(e_sequence)
#print(p_data.shape)
#print(area_list.shape)
print('expalined variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))


print(p_data.shape)


#label is a list that contain different name of the area
#area_list represent the area that each individual lives in


plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(p_data[area_list==target_name, 0], p_data[area_list==target_name, 1],
                alpha = .8,color = color, lw=lw, label = target_name)
    
    
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution')

plt.show()
'''
###############################################################################################################

#applying autoencoder

#a_data is the decoded data run be autoencoder

a_data = auto.deep_test(e_sequence.shape[1],e_sequence, e_sequence.shape[0])
'''
ap_data = auto2.deep_test(e_sequence.shape[1],e_sequence, e_sequence.shape[0])
'''
'''
plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(a_data[area_list==target_name, 0], a_data[area_list==target_name, 1],alpha = .8,
                color = color,   lw=lw, label = target_name)
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution with Relu')


plt.show()

'''
'''

plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(ap_data[area_list==target_name, 0], ap_data[area_list==target_name, 1],alpha = .8,
                color = color,   lw=lw, label = target_name)
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution with linear')

plt.show()
'''


