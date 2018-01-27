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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


########################################################################################
# choose two biggest lists from the four input list and return
def Top_2(list1, list2, list3, list4):
        list_a = [list1, list2, list3, list4]
        
        count = 0
#variable "count" count how many no empty list do we have
        for t_list in list_a:
            if t_list:
                count += 1
            else:
                break

        i = 1
        while i<count:
            x = list_a[i]
            j = i-1
            while j>=0 and list_a[j][1]<x[1]:
                list_a[j+1] = list_a[j]
                j = j-1
            list_a[j+1] = x
            i = i+1
        
        return list_a[0], list_a[1]
        

                
                
            





########################################################################################
# base_calc return the two list with base name and the number of one SNP
def base_calc(array):
    #print(array[0:10])
    a=c=g=t=0
    for i in array:
        #print(i)
        if(i=='A'):
            a+=1
        elif(i=='C'):
            c+=1
        elif(i=='G'):
            g+=1
        elif(i=='T'):
            t+=1
        else:
            continue
    label = ['A', 'G', 'C', 'T']
    value = [a,g,c,t]
    #print(value)
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for i in range(4):
        if(value[i]!=0 ):
            if not list1:
                list1.append(label[i])
                list1.append(value[i])
            elif not list2:
                list2.append(label[i])
                list2.append(value[i])
            elif not list3:
                list3.append(label[i])
                list3.append(value[i])
            else:
                list4.append(label[i])
                list4.append(value[i])
    #choose the two base list with the biggest value
    list1, list2 = Top_2(list1, list2, list3, list4)         
    if not list2:
        list2.append("N")
        list2.append(0)
    return list1, list2

#####################################################################################
#encode the base:
#encode the base to 0 and 1 based on their frequency, for example if num(A)> num(C), then A should be encoded 
# as 0 and C as 1
def base_encoder(array):
    total_list = []
    #print(len(array[0]))
    #print(array.T[1])
    for i in range(len(array[0])-1):
        temp_list = []
        list1, list2 = base_calc(array.T[i])
        #print(i)
    
        if(list1[1]>list2[1]):
            Dominant_base = list1[0]
        else:
            Dominant_base = list2[0]
        #temp_list = np.array(array[:i]).tolist()
        #print(array.T[0])
        for base in array.T[i]:
            #the ith row in array's transpose is equal to the ith coloumn in array
            if(base==Dominant_base or base=="?"):
                temp_list.append(0)
            else:
                temp_list.append(1)
            #temp_list store the SNP for different individuals
        if (i==0):
            total_list = np.asarray(temp_list)
        else:
            total_list = np.vstack((total_list, np.asarray(temp_list)))
    return total_list.transpose()
        
        
        
#################################################################################

# this is the main function, reading the raw data and store into a matrix
counter = 0
counter2 = 0
counter3 = 0
file = open("phased_HGDP+India+Africa_2810SNPs-regions1to36.stru", "r") #open the dateset
area_list = []

for i, line in enumerate(file):
        if i==5:
            for j in line:
                counter2+=1
                if(j==" "):
                    counter+=1
                   
                if(counter>=7):
                    array_o = re.split(' |\n',line[counter2:])
                    #array_o = line[counter2:].split(" |\n")
                    #print(array_o)
                    counter2=0
                    counter=0
                    counter3=0 
                    area_list.append(line.split(" ", 5)[4])
                    # only split the first blank, take only the first word
                    break
        if i> 5:
            for j in line:
                counter3+=1
                if(j==" "):
                    counter+=1
                    
                if(counter>=7):
                    array2 = re.split(' |\n',line[counter3:])
                    #array2 = line[counter3:].split(" |\n")
                    counter3=0
                    counter=0
                    #print(len(array2))
                    
                    break
            #for h in range(len(array)):
            if(i%2!=0):
                #odd list
                area_list.append(line.split(" ", 5)[4])
                array_o = np.vstack((array_o,array2))
                
            else:
                #even list
                if(i==6):
                    array_e = array2
                else:
                    array_e = np.vstack((array_e,array2))

file.close()
area_list = np.array(area_list)

#####################################################################################                    
# encoded the base to 0 and 1 in chromosome 1
coded_array_o = base_encoder(array_o)
#encoded the base to 0 and q in chromosome 2
coded_array_e = base_encoder(array_e)


encoded_sequence1 = np.asmatrix(base_encoder(array_o))
encoded_sequence2 = np.asmatrix(base_encoder(array_e))


#print(encoded_sequence1)
#the final encode sequence
e_sequence_t = encoded_sequence1 + encoded_sequence2

#cast the type of the data from int to float
e_sequence_t = e_sequence_t.astype(float)
#print(e_sequence_t.shape)
#normalize the matrix by columns
#e_sequence_t = normalize(e_sequence_t, norm = 'l1', axis = 0)

#processing the matrix
#step1: X = X - E(X)
#step2: X = X/sigma
for i in range(e_sequence_t.shape[1]):
    sigma = (e_sequence_t[:, i]).std()
    mean = (e_sequence_t[:,i]).mean()
    e_sequence_t[:, i] -= mean
    e_sequence_t[:, i] /= sigma






e_sequence = np.asarray(e_sequence_t)

#######################################################################################
#applying PCA

pca = PCA(n_components = 2)
p_data = pca.fit(e_sequence).transform(e_sequence)
#print(p_data.shape)
#print(area_list.shape)
print('expalined variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
colors = ['navy', 'gray', 'darkorange', 'green', 'red', 'yellow', 'cyan']

area_label = np.array(["EUROPE","MIDDLE_EAST","CENTRAL_SOUTH_ASIA","OCEANIA","AMERICA","AFRICA","EAST_ASIA"])

plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(p_data[area_list==target_name, 0], p_data[area_list==target_name, 1],
                alpha = .8,color = color, lw=lw, label = target_name)
    
    
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SPNs distribution with PCA')

plt.show()

###############################################################################################################

#applying autoencoder

#a_data is the decoded data run be autoencoder

a_data = auto.deep_test(e_sequence.shape[1],e_sequence, e_sequence.shape[0])

ap_data = auto2.deep_test(e_sequence.shape[1],e_sequence)

plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(a_data[area_list==target_name, 0], a_data[area_list==target_name, 1],alpha = .8,
                color = color,   lw=lw, label = target_name)
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution with Relu')


plt.show()


plt.figure()
lw = 2
for color, target_name in zip(colors, area_label):
    plt.scatter(ap_data[area_list==target_name, 0], ap_data[area_list==target_name, 1],alpha = .8,
                color = color,   lw=lw, label = target_name)
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('SNPs distribution with linear')

plt.show()





