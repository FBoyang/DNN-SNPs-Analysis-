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
import autoencoder_saliksyed3 as auto3
from sklearn.preprocessing import normalize
from random import randint
import os
import rbm
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

#applying autoencoder and PCA

#a_data is the decoded data run be autoencoder

#encoding_weight colletct the weight functions of each hidden layer
encoding_weight = []
pca = PCA(n_components = 1000)
p_data = pca.fit(e_sequence).transform(e_sequence)
Weight = pca.components_.transpose()
encoding_weight.append(Weight)

pca2 = PCA(n_components = 500)
p_data2 = pca2.fit(p_data).transform(p_data)
Weight2 = pca2.components_.transpose()
encoding_weight.append(Weight2)

pca3 = PCA(n_components = 50)
p_data3 = pca3.fit(p_data2).transform(p_data2)
Weight3 = pca3.components_.transpose()
encoding_weight.append(Weight3)

pca4 = PCA(n_components = 2)
p_data4 = pca4.fit(p_data3).transform(p_data3)
Weight4 = pca4.components_.transpose()
encoding_weight.append(Weight4)

#print(len(encoding_weight))
#print(encoding_weight[0].shape)
#print(encoding_weight[1].shape)

a_data = auto3.deep_test(e_sequence.shape[1],e_sequence, e_sequence.shape[0], encoding_weight)
'''
ap_data = auto2.deep_test(e_sequence.shape[1],e_sequence, e_sequence.shape[0])
'''
plt.figure()
lw = 2
plt.scatter(a_data[area_list=="EUROPE", 0], a_data[area_list=="EUROPE", 1],alpha = .8,color = "blue",   lw=lw, label = "EUROPE")
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution with Relu')


plt.show()


'''
r = rbm.RBM(num_visible = e_sequence.shape[1], num_hidden = 1000)
r = r.run_visible(e_sequence)

r1 = rbm.RBM(num_visible = e_sequence.shape[1], num_hidden = 500)
r1 = r1.run_visible(r)

r2 = rbm.RBM(num_visible = e_sequence.shape[1], num_hidden = 100)
r2 = r2.run_visible(r1)

r3 = rbm.RBM(num_visible = e_sequence.shape[1], num_hidden = 20)
r3 = r3.run_visible(r2)

r4 = rbm.RBM(num_visible = e_sequence.shape[1], num_hidden = 2)
r4 = r4.run_visible(r3)
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


