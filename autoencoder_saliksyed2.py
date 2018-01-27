""" Deep Auto-Encoder implementation
	
	An auto-encoder works as follows:
	Data of dimension k is reduced to a lower dimension j using a matrix multiplication:
	softmax(W*x + b)  = x'
	
	where W is matrix from R^k --> R^j
	A reconstruction matrix W' maps back from R^j --> R^k
	so our reconstruction function is softmax'(W' * x' + b') 
	Now the point of the auto-encoder is to create a reduction matrix (values for W, b) 
	that is "good" at reconstructing  the original data. 
	Thus we want to minimize  ||softmax'(W' * (softmax(W *x+ b)) + b')  - x||
	A deep auto-encoder is nothing more than stacking successive layers of these reductions.
"""
import tensorflow as tf
import numpy as np
import math
import random
from random import randint


def lrelu(x, alpha):

	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

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

		output = tf.matmul(next_layer_input,W) + b

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
		output = tf.matmul(next_layer_input,W) + b
		next_layer_input = output

	# the fully encoded and reconstructed value of x is here:
	reconstructed_x = next_layer_input

	return {
		'encoded': encoded_x,
		'decoded': reconstructed_x,
		'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
	}



def deep_test(s_dim, matrix, row_num):
	#s_dim takes the dimension of the input	
	sess = tf.Session()
	start_dim = s_dim
	x = tf.placeholder("float", [None, start_dim])
	autoencoder = create(x, [2000,200,20,2])
	init = tf.global_variables_initializer()
	sess.run(init)
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(autoencoder['cost'])

	x2 = tf.placeholder("float", [row_num, start_dim])
#	Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
#	randomly generate a sample batch from the matrix
#	this snip doesn't give me the right output
	batch = []
	for i in range(50):
		batch.append(matrix[randint(0,row_num-1), :])
	print(batch[2])
	print(sess.run(train_step, feed_dict = {x: batch}))
	encoded_data = sess.run(autoencoder['encoded'], feed_dict = {x: matrix})
#	print("cost is: %d" %(sess.run(autoencoder['cost'], feed_dict = {x:batch})))
#	print(decoded_data.shape)
	return encoded_data

