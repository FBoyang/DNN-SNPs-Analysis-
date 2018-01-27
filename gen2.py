import numpy as np
import math
import random
import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    for i in range(4):
        if(value[i]!=0 ):
            if not list1:
                list1.append(label[i])
                list1.append(value[i])
            else:
                list2.append(label[i])
                list2.append(value[i])
    if not list2:
        list2.append("N")
        list2.append(0)
    #print(list1,list2)
    return list1, list2

#####################################################################################
#encode the base:
def base_encoder(array):
    total_list = []
    #print(len(array[0]))
    #print(array.T[1])
    for i in range(len(array[0])-1):
        temp_list = []
        list1, list2 = base_calc(array.T[i])
        #print(i)
        #print(list1,list2)
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


counter = 0
counter2 = 0
counter3 = 0
file = open("phased_HGDP+India+Africa_2810SNPs-regions1to36.stru", "r")
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
                array_o = np.vstack((array_o,array2))
            else:
                if(i==6):
                    array_e = array2
                else:
                    array_e = np.vstack((array_e,array2))
                    
#####################################################################################                    
                    
                
encoded_sequence1 = np.asmatrix(base_encoder(array_o))
encoded_sequence2 = np.asmatrix(base_encoder(array_e))
#print(encoded_sequence1)
#the final encode sequence
e_sequence = encoded_sequence1 + encoded_sequence2

#######################################################################################
#applying PCA

pca = PCA(n_components = 2)
p_data = pca.fit(e_sequence).transform(e_sequence)
target_names = ["normal", "abnormal","extrem_abnormal"]
print('expalined variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

#plt.figure()
#colors = ['navy', 'turquoise', 'darkorange']
#lw = 2

#for color, i, target_name in zip(colors, [0,1,2], target_names):
#plt.scatter(p_data[e_sequence==i, 0], p_data[e_sequence==i,1], color = color, alpha = .8, lw = lw, label = target_name)
#plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
#plt.title('PCA of Gene dataset')

################################################################################################################

def create(x, layer_sizes):

	# Build the encoding layers
	next_layer_input = x

	encoding_matrices = []
	for dim in layer_sizes:
		input_dim = int(next_layer_input.get_shape()[1])

		# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
		W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

		# Initialize b to zero
		b = tf.Variable(tf.zeros([dim]))

		# We are going to use tied-weights so store the W matrix for later reference.
		encoding_matrices.append(W)

		output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

		# the input into the next layer is the output of this layer
		next_layer_input = output

	# The fully encoded x value is now stored in the next_layer_input
	encoded_x = next_layer_input

	# build the reconstruction layers by reversing the reductions
	layer_sizes.reverse()
	encoding_matrices.reverse()


	for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
		# we are using tied weights, so just lookup the encoding matrix for this step and transpose it
		W = tf.transpose(encoding_matrices[i])
		b = tf.Variable(tf.zeros([dim]))
		output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
		next_layer_input = output

	# the fully encoded and reconstructed value of x is here:
	reconstructed_x = next_layer_input

	return {
		'encoded': encoded_x,
		'decoded': reconstructed_x,
		'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
	}

"""def simple_test():
	sess = tf.Session()
	x = tf.placeholder("float", [None, 4])
	autoencoder = create(x, [2])
	init = tf.initialize_all_variables()
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(autoencoder['cost'])


	# Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
	c1 = np.array([0,0,0.5,0])
	c2 = np.array([0.5,0,0,0])

	# do 1000 training steps
	for i in range(2000):
		# make a batch of 100:
		batch = []
		for j in range(100):
			# pick a random centroid
			if (random.random() > 0.5):
				vec = c1
			else:
				vec = c2
			batch.append(np.random.normal(vec, 0.1))
		sess.run(train_step, feed_dict={x: np.array(batch)})
		if i % 100 == 0:
			print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})
"""

def deep_test():
	sess = tf.Session()
	start_dim = len(e_sequence[0])
	x = tf.placeholder("float", [None, start_dim])
	autoencoder = create(x, [4, 3, 2])
	init = tf.initialize_all_variables()
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])


	# Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
	c1 = e_sequence[0]
    #c1 = np.zeros(start_dim)
	#c1[0] = 1

	#print c1

	#c2 = np.zeros(start_dim)
	#c2[1] = 1

	# do 1000 training steps
    #for i in range(len(e_sequence)):
		# make a batch of len(e_sequence):
		#batch = []
		#for j in range(1):
			# pick a random centroid
			#if (random.random() > 0.5):
			#	vec = c1
			#else:
			#	vec = c2
  			#batch.append(np.random.normal(vec, 0.1))
		#sess.run(train_step, feed_dict={x: np.array(batch)})

       
		#if i % 100 == 0:
	batch = e_sequence
	print (len(e_sequence, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})))
	print (len(e_sequence), " original", batch[0])
	print (len(e_seuqnce), " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))
if __name__ == '__main__':
        deep_test()
