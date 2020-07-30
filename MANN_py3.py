## Theory
"""
Neural Turing Machine
# https://medium.com/towards-artificial-intelligence/neural-turing-machines-eaada7e7a6cc
* NTMs are designed to solve tasks that require writing to and retrieving information from an external memory.
* Compared with RNN structure with internal memory, NTMs utilize attentional mechanisms to efficiently read and write an external memory, which makes them a more favorable choice for capturing long-range dependencies.
* The   memory inter-actions  are  differentiable  end  to  end. 
* Memory encoding and retrieval in NTMs is rapid. This feature of NTM makes it a perfect candidate for meta-learning [1] and low-shot [2] prediction, since it can be used for both long-term storage   which   is   possible   with   slow   updates   of   network weights,  and  short-term  storage  with  its  external  memory module.

# http://oaji.net/articles/2017/1992-1530774163.pdf
MANN : The goal here is to modify an NTM model to excel at one-shot learning. In  one  shot learning, we make use of content based addressing mechanism. This is because for a given input, there are only two actions the controller    might    need    to    do    and    both    depend    on content-addressing. One action is that the input is very similar to a previously seen input; in this case, we might want to update whatever  we  wrote  to  memory.  The  other  action  is  that  the input is not similar to a previously seen input; in this case, we do not want to overwrite recent information, so we will instead write to the least used memory location.

# https://github.com/gopala-kr/summary/tree/master/summaries/Week-3
# https://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html
# https://towardsdatascience.com/hands-on-memory-augmented-neural-networks-implementation-part-one-a6a4a88beba3 
"""

# https://www.groundai.com/project/text-normalization-using-memory-augmented-neural-networks/1
# https://github.com/MarkPKCollier/MANNs4NMT
# https://github.com/snowkylin/ntm
# https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#a-simple-view

# READINGS:
# https://arxiv.org/pdf/1605.06065.pdf
# https://iopscience.iop.org/article/10.1088/1742-6596/1237/3/032044/pdf

"""
Keywords to note:
> Sequence to Sequence Models
> RNNs
> LSTM and GRU -> internal memory
> attentional encoder decoder 
> Applications in:
	> Machine Translation
	> One shot learning
	
> Meta learning : meta-learning  generally  refers  to  a  scenario  in  which  anagent  learns  at  two  levels,  each  associated  with  differenttime scales.   Rapid learning occurswithina task,  for ex-ample, when learning to accurately classify within a par-ticular  dataset.This  learning  is  guided  by  knowledgeaccrued  more  graduallyacrosstasks,  which  captures  theway in which task structure varies across target domains.

# https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a
"""

import tensorflow as tf
import numpy as np
import os
import random
import pickle

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize
# from imageio import imread
# from numpy import resize as imresize
# import scipy.misc.imresize as imresize
# import scipy.imageio.imread as imread

def get_sampled_data(character_folders, nb_classes=5, nb_samples=10*5):
	sampled_characters = random.sample(character_folders, nb_classes)
	labels_and_images = [(label, os.path.join(character, image_path)) \
				for label, character in zip(np.arange(nb_classes), sampled_characters) \
				for image_path in os.listdir(character)]
	sampled_data = random.sample(labels_and_images, nb_samples)
	return sampled_data


def transform_image(image_path, angle=0., s=(0,0), size=(20,20)):
	original = imread(image_path, flatten=True)
	rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
	shifted = shift(rotated, shift=s)
	resized = np.asarray(imresize(rotated, size=size), dtype=np.float32)/255
	inverted = 1. - resized
	max_value = np.max(inverted)
	if max_value > 0:
		inverted /= max_value
	return inverted

# cosine similarity for content based addressing
def cosine_similarity(x, y, eps=1e-6):
  z = tf.matmul(x, tf.transpose(y, perm=[0,2,1])) 
  z /= tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(x), 2, keep_dims=True), tf.reduce_sum(tf.square(x), 2, keep_dims=True) + eps))
  return z

def variable_float32(x, name=''):
  return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), dtype=tf.float32), name=name)

def variable_one_hot(shape, name=''):
  initial = np.zeros(shape, dtype=np.float32)
  initial[...,0] = 1
  return tf.Variable(tf.cast(initial, dtype=tf.float32), name=name)


class OmniglotGenerator(object):

	def __init__(self, data_folder, batch_size=1, nb_classes=5, nb_samples=10*5, max_rotation=np.pi/6, \
			max_shift=10, img_size=(20, 20)):
		self.data_folder = data_folder
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.nb_samples = nb_samples
		self.max_rotation = max_rotation
		self.max_shift = max_shift
		self.img_size = img_size
		self.character_folders = [os.path.join(data_folder, alphabet, character) for alphabet in os.listdir(data_folder) \
		 						if os.path.isdir(os.path.join(data_folder, alphabet)) \
		 						for character in os.listdir(os.path.join(data_folder, alphabet))]
	
	def episode(self):
		episode_input = np.zeros((self.batch_size, self.nb_samples, np.prod(self.img_size)), dtype=np.float32)
		episode_output = np.zeros((self.batch_size, self.nb_samples), dtype=np.int32)

		for i in range(self.batch_size):
			sampled_data = get_sampled_data(self.character_folders, nb_classes=self.nb_classes, nb_samples=self.nb_samples)
			sequence_length = len(sampled_data)
			labels, image_files = zip(*sampled_data)

			angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
			shifts = np.random.randint(-self.max_shift, self.max_shift + 1, size=(sequence_length, 2))

			episode_input[i] = np.asarray([transform_image(filename, angle=angle, s=shift, size=self.img_size).flatten() \
				 				for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
			episode_output[i] = np.asarray(labels, dtype=np.int32)

		return episode_input, episode_output


class mann(object):

	def __init__(self, input_size=20*20, memory_size=(128, 40),controller_size=200, nb_reads=4, nb_classes=5, batch_size=16):
		self.input_size = input_size
		self.memory_size = memory_size
		self.controller_size = controller_size
		self.nb_reads = nb_reads
		self.nb_classes = nb_classes
		self.batch_size = batch_size

	def initialize(self):
		# controller is an LSTM
		M_0 = variable_float32(1e-6 * np.ones((self.batch_size,) + self.memory_size), name='memory')
		c_0 = variable_float32(np.zeros((self.batch_size, self.controller_size)), name='controller_cell_state')
		h_0 = variable_float32(np.zeros((self.batch_size, self.controller_size)), name='controller_hidden_state')
		r_0 = variable_float32(np.zeros((self.batch_size, self.nb_reads * self.memory_size[1])), name='read_vector')
		wr_0 = variable_one_hot((self.batch_size, self.nb_reads, self.memory_size[0]), name='wr')
		wu_0 = variable_one_hot((self.batch_size, self.memory_size[0]), name='wu')

		return [M_0, c_0, h_0, r_0, wr_0, wu_0]

	def step(self, initializer, x_t):
		M_tm1, c_tm1, h_tm1, r_tm1, wr_tm1, wu_tm1 = initializer
		
		with tf.variable_scope('weights', reuse=True):
			W_key = tf.get_variable('W_key', shape=(self.nb_reads, self.controller_size, self.memory_size[1]))
			b_key = tf.get_variable('b_key', shape=(self.nb_reads, self.memory_size[1]))

			W_sigma = tf.get_variable('W_sigma', shape=(self.nb_reads, self.controller_size, 1))
			b_sigma = tf.get_variable('b_sigma', shape=(self.nb_reads, 1))

			W_xh = tf.get_variable('W_xh', shape=(self.input_size + self.nb_classes, 4*self.controller_size))
			W_hh = tf.get_variable('W_hh', shape=(self.controller_size, 4*self.controller_size))
			b_h = tf.get_variable('b_h', shape=(4*self.controller_size))

			W_o = tf.get_variable('W_o', shape=(self.controller_size + self.nb_reads * self.memory_size[1], self.nb_classes))
			b_o = tf.get_variable('b_o', shape=(self.nb_classes))

			gamma = 0.95

		def lstm_step(size, x_t, c_tm1, h_tm1, W_xh, W_hh, b_h):

			preactivations = tf.matmul(x_t, W_xh) + tf.matmul(h_tm1, W_hh) + b_h

			gf = tf.sigmoid(preactivations[:, 0:size])	#Forget Gate
			gi = tf.sigmoid(preactivations[:, size:2*size])	#Input Gate
			go = tf.sigmoid(preactivations[:, 2*size:3*size])	#Output Gate
			u = tf.tanh(preactivations[:, 3*size:4*size])

			c_t = gf*c_tm1 + gi*u	#Context vector
			h_t = go*tf.tanh(c_t)	# Next state

			return [c_t, h_t]

		[c_t, h_t] = lstm_step(self.controller_size, x_t, c_tm1, h_tm1, W_xh, W_hh, b_h)

		shape_key = (self.batch_size, self.nb_reads, self.memory_size[1])
		shape_sigma = (self.batch_size, self.nb_reads, 1)

		_W_key = tf.reshape(W_key, shape=(self.controller_size, -1))
		_W_sigma = tf.reshape(W_sigma, shape=(self.controller_size, -1))

		k_t = tf.tanh(tf.reshape(tf.matmul(h_t, _W_key), shape=shape_key) + b_key)	#Key to read
		sigma_t = tf.sigmoid(tf.reshape(tf.matmul(h_t, _W_sigma), shape=shape_sigma) + b_sigma)	#Key to write

		#wlu_tm1 is the weight representing least-used weights
		# wu_tm1 is what is available, hence the index must match for wlu as well => wlu_tm1
		_, indices = tf.nn.top_k(wu_tm1, k=self.memory_size[0])	#Only indices matter. Values are discarded
		wlu_tm1 = tf.slice(indices, [0,self.memory_size[0] - self.nb_reads], [self.batch_size,self.nb_reads])
		wlu_tm1 = tf.cast(wlu_tm1, dtype=tf.int32)	

		## Next lines are finding a mask which matches exactly with the memory locations, then initialize to 0, then set to Mtm1 
		row_idx = tf.reshape(tf.tile(tf.reshape(wlu_tm1[:,0], shape=(-1, 1)), (1, self.memory_size[1])), [-1])
		row_idx += self.memory_size[0] * tf.reshape(tf.tile(tf.reshape(list(range(self.batch_size)), shape=(-1, 1)), (1, self.memory_size[1])), [-1])
		
		col_idx = tf.tile(list(range(self.memory_size[1])), [self.batch_size])
		
		coords = tf.transpose(tf.stack([row_idx, col_idx]))
		binary_mask = tf.cast(tf.sparse_to_dense(coords, (self.batch_size*self.memory_size[0], self.memory_size[1]), 1), tf.bool)
		
		M_t = tf.where(binary_mask, tf.constant(0., shape=(self.batch_size*self.memory_size[0], self.memory_size[1])), tf.reshape(M_tm1, shape=(self.batch_size*self.memory_size[0], self.memory_size[1])))
		M_t = tf.reshape(M_t, shape=(self.batch_size, self.memory_size[0], self.memory_size[1]))

		## final step of wlu_tm1 -> the weights are to be either 0 or 1
		wlu_tm1 = tf.one_hot(wlu_tm1, self.memory_size[0], axis=-1)
		#learnable sigmoid gate. #write weights
		ww_t = tf.multiply(sigma_t, wr_tm1) + tf.multiply(1.-sigma_t, wlu_tm1)

		## adding weighted component
		M_t = M_t + tf.matmul(tf.transpose(ww_t, perm=[0,2,1]), k_t)
		## cosine similarity between key and already available elements in the memory
		K_t = cosine_similarity(k_t, M_t)
		wr_t = tf.nn.softmax(K_t)#read weight vector

		wu_t = gamma*wu_tm1 + tf.reduce_sum(wr_t, axis=1)+ tf.reduce_sum(ww_t, axis=1)#write weight vector
		#This memory is used by the controller as the input to a classifier, such as a softmax output layer, and as an additional input for the next controller state.
		r_t = tf.reshape(tf.matmul(wr_t, M_t), shape=(self.batch_size,-1))	#The memory vector

		return [M_t, c_t, h_t, r_t, wr_t, wu_t]

	def compute_output(self, input_var, target_var):
		M_0, c_0, h_0, r_0, wr_0, wu_0 = self.initialize()
		print([M_0, c_0, h_0, r_0, wr_0, wu_0])
		print()
		print()

		with tf.variable_scope('weights'):
			W_key = tf.get_variable('W_key', shape=(self.nb_reads, self.controller_size, self.memory_size[1]))
			b_key = tf.get_variable('b_key', shape=(self.nb_reads, self.memory_size[1]))

			W_sigma = tf.get_variable('W_sigma', shape=(self.nb_reads, self.controller_size, 1))
			b_sigma = tf.get_variable('b_sigma', shape=(self.nb_reads, 1))

			W_xh = tf.get_variable('W_xh', shape=(self.input_size + self.nb_classes, 4*self.controller_size))
			W_hh = tf.get_variable('W_hh', shape=(self.controller_size, 4*self.controller_size))
			b_h = tf.get_variable('b_h', shape=(4*self.controller_size))

			W_o = tf.get_variable('W_o', shape=(self.controller_size + self.nb_reads * self.memory_size[1], self.nb_classes))
			b_o = tf.get_variable('b_o', shape=(self.nb_classes))

			gamma = 0.95

		sequence_length = input_var.get_shape().as_list()[1]

		one_hot_target = tf.one_hot(target_var, self.nb_classes, axis=-1)
		offset_target_var = tf.concat([tf.zeros_like(tf.expand_dims(one_hot_target[:,0], 1)), one_hot_target[:,:-1]], axis=1)
		ntm_input = tf.concat([input_var, offset_target_var], axis=2)

		elems = tf.transpose(ntm_input, perm=[1,0,2])
		print(elems)
		print()
		print()
		#ntm_var = tf.scan(self.step,elems,M_0, c_0, h_0, r_0, wr_0, wu_0)
		ntm_var = tf.scan(self.step, elems= elems, initializer=[M_0, c_0, h_0, r_0, wr_0, wu_0])
		ntm_output = tf.transpose(tf.concat(ntm_var[2:4], axis=2), perm=[1,0,2])

		print('Done')
		print()
		print()

		#output gate
		output_var = tf.matmul(tf.reshape(ntm_output, shape=(self.batch_size*sequence_length, -1)), W_o) + b_o
		output_var = tf.reshape(output_var, shape=(self.batch_size, sequence_length, -1))
		output_var = tf.nn.softmax(output_var)

		params = [W_key, b_key, W_sigma, b_sigma, W_xh, W_hh, b_h, W_o, b_o]

		return output_var, params


from matplotlib import pyplot as plt
from argparse import ArgumentParser
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

# BATCH_SIZE = 16
# NB_CLASSES = 5
# NB_SAMPLES = 10*5
# INPUT_HEIGHT = 20
# INPUT_WIDTH = 20

# NB_READS = 4
# CONTROLLER_SIZE = 200
# MEMORY_LOCATIONS = 128
# MEMORY_WORD_SIZE = 40

# LEARNING_RATE = 1e-4
# ITERATIONS = 100000
def build_argparser():
	parser = ArgumentParser()
	BATCH_SIZE = 16
	NB_CLASSES = 5
	NB_SAMPLES = 10*5
	INPUT_HEIGHT = 20
	INPUT_WIDTH = 20

	NB_READS = 4
	CONTROLLER_SIZE = 200
	MEMORY_LOCATIONS = 128
	MEMORY_WORD_SIZE = 40

	LEARNING_RATE = 1e-4
	ITERATIONS = 100#100000

	parser.add_argument('--batch-size',
			dest='_batch_size',	help='Batch size (default: %(default)s)',
			type=int, default=BATCH_SIZE)
	parser.add_argument('--num-classes',
			dest='_nb_classes', help='Number of classes in each episode (default: %(default)s)',
			type=int, default=NB_CLASSES)
	parser.add_argument('--num-samples',
			dest='_nb_samples', help='Number of total samples in each episode (default: %(default)s)',
			type=int, default=NB_SAMPLES)
	parser.add_argument('--input-height',
			dest='_input_height', help='Input image height (default: %(default)s)',
			type=int, default=INPUT_HEIGHT)
	parser.add_argument('--input-width',
			dest='_input_width', help='Input image width (default: %(default)s)',
			type=int, default=INPUT_WIDTH)
	parser.add_argument('--num-reads',
			dest='_nb_reads', help='Number of read heads (default: %(default)s)',
			type=int, default=NB_READS)
	parser.add_argument('--controller-size',
			dest='_controller_size', help='Number of hidden units in controller (default: %(default)s)',
			type=int, default=CONTROLLER_SIZE)
	parser.add_argument('--memory-locations',
			dest='_memory_locations', help='Number of locations in the memory (default: %(default)s)',
			type=int, default=MEMORY_LOCATIONS)
	parser.add_argument('--memory-word-size',
			dest='_memory_word_size', help='Size of each word in memory (default: %(default)s)',
			type=int, default=MEMORY_WORD_SIZE)
	parser.add_argument('--learning-rate',
			dest='_learning_rate', help='Learning Rate (default: %(default)s)',
			type=float, default=LEARNING_RATE)
	parser.add_argument('--iterations',
			dest='_iterations', help='Number of iterations for training (default: %(default)s)',
			type=int, default=ITERATIONS)
	parser.add_argument('--path',
			dest='_path', help='Path to input folder (default: %(default)s)',
			type=str, default='\home')

	return parser


def omniglot(batch_size,nb_classes, nb_samples, img_size, input_size, nb_reads, controller_size,memory_size, learning_rate,max_iter, path):

	# batch_size = BATCH_SIZE
	# nb_classes = NB_CLASSES
	# nb_samples = NB_SAMPLES
	# img_size = (INPUT_HEIGHT, INPUT_WIDTH)
	# input_size = INPUT_HEIGHT*INPUT_WIDTH

	# nb_reads = NB_READS
	# controller_size = CONTROLLER_SIZE
	# memory_size = (MEMORY_LOCATIONS,MEMORY_WORD_SIZE)

	# learning_rate = LEARNING_RATE
	# max_iter = ITERATIONS

	input_var = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_samples, input_size))
	target_var = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_samples))

	# # choose a local (colab) directory to store the data.
  # local_download_path = os.path.expanduser('~/data/annotation')
  # try:
  #   os.makedirs(local_download_path)
  # except: 
  #   pass

  # # 2. Auto-iterate using the query syntax
  # #    https://developers.google.com/drive/v2/web/search-parameters
  # file_list = drive.ListFile({'q': "'1JInWt48attjYTuTyc2i0x8guCyF6KNBd' in parents"}).GetList()

  # for f in file_list:
  #   # 3. Create & download by id.
  #   print('title: %s, id: %s' % (f['title'], f['id']))
  #   fname = os.path.join(local_download_path, f['title'])
  #   print('downloading to {}'.format(fname))
  #   f_ = drive.CreateFile({'id': f['id']})
  #   f_.GetContentFile(fname)
	
	generator = OmniglotGenerator(data_folder=path, batch_size=batch_size, nb_classes=nb_classes, \
			nb_samples=nb_samples, max_rotation=0., max_shift=0, img_size=img_size)
	print("Generated Data")
	print()
	print()
	net = mann(input_size=input_size, memory_size=memory_size, controller_size=controller_size, \
			nb_reads=nb_reads, nb_classes=nb_classes, batch_size=batch_size)
	output_var, params = net.compute_output(input_var, target_var)

	i = 0
	with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
		W_key = tf.get_variable('W_key', shape=(nb_reads, controller_size, memory_size[1]))
		b_key = tf.get_variable('b_key', shape=(nb_reads, memory_size[1]))

		W_sigma = tf.get_variable('W_sigma', shape=(nb_reads, controller_size, 1))
		b_sigma = tf.get_variable('b_sigma', shape=(nb_reads, 1))

		W_xh = tf.get_variable('W_xh', shape=(input_size + nb_classes, 4*controller_size))
		W_hh = tf.get_variable('W_hh', shape=(controller_size, 4*controller_size))
		b_h = tf.get_variable('b_h', shape=(4*controller_size))

		W_o = tf.get_variable('W_o', shape=(controller_size + nb_reads * memory_size[1], nb_classes))
		b_o = tf.get_variable('b_o', shape=(nb_classes))

		gamma = 0.95

	params = [W_key, b_key, W_sigma, b_sigma, W_xh, W_hh, b_h, W_o, b_o]

	target_one_hot = tf.one_hot(target_var, nb_classes, axis=-1)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_var, labels=target_one_hot), name="cost")
	acc = tf.reduce_mean(tf.cast(tf.equal(target_var, tf.cast(tf.argmax(output_var, axis=2), dtype=tf.int32)), dtype=tf.float32))

	opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95, momentum=0.9)
	train_step = opt.minimize(cost, var_list=params)


	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	iters = []
	accuracies = []
	costs = []

	with sess.as_default():
		try:
			for i in range(max_iter):
				print(i)
				episode_input, episode_output = generator.episode()
				feed_dict = {input_var: episode_input, target_var: episode_output}
				train_step.run(feed_dict)
				if i % 5 == 0:	#(max_iter*1e-3)
					cost_val = sess.run(cost, feed_dict=feed_dict)
					acc_val = sess.run(acc, feed_dict=feed_dict)

					iters.append(i)
					costs.append(cost_val)
					accuracies.append(acc_val)
					
					print('Target Labels:')
					print(sess.run(target_var[0], feed_dict=feed_dict))
					print('Model Output:')
					print(sess.run(tf.argmax(output_var[0], axis=1), feed_dict=feed_dict))
					print('Episode ' + str(i) + ': Cost = ' + str(cost_val) + '\t Accuracy = ' + str(acc_val))
					print('')

					with open('omniglot-cost', 'wb') as fp:
						pickle.dump(costs, fp)

					with open('omniglot-acc', 'wb') as fp:
						pickle.dump(accuracies, fp)

					with open('omniglot-iters', 'wb') as fp:
						pickle.dump(iters, fp)

		except KeyboardInterrupt:
			print('\nInterrupted at Episode ' + str(i))
			print('Cost = ' + str(cost_val))
			print('Accuracy = ' + str(acc_val))
			pass

	
	fig = plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	plt.plot(iters, costs, 'b', label='Training Error', linewidth=2, alpha=0.8)
	plt.xlabel('Episodes', fontsize=22)
	plt.ylabel('Cross Entropy Loss', fontsize=22)
	plt.title('Training Error', fontsize=26)

	plt.subplot(1,2,2)
	plt.plot(iters, accuracies, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
	plt.xlabel('Episodes', fontsize=22)
	plt.ylabel('Accuracy', fontsize=22)
	plt.title('Training Accuracy', fontsize=26)
	plt.show()

def main():
	parser = build_argparser()
	args = parser.parse_args()

	batch_size = args._batch_size
	nb_classes = args._nb_classes
	nb_samples = args._nb_samples
	img_size = (args._input_height, args._input_width)
	input_size = args._input_height * args._input_width

	nb_reads = args._nb_reads
	controller_size = args._controller_size
	memory_size = (args._memory_locations, args._memory_word_size)
	
	learning_rate = args._learning_rate
	max_iter = args._iterations
	path = '/media/aiswarya/New Volume/My_works/ISI_contd/images_background' #args._path
	omniglot(batch_size,nb_classes, nb_samples, img_size, input_size, nb_reads, controller_size,memory_size, learning_rate,max_iter, path)

main()


#'/media/aiswarya/New Volume/My_works/ISI_contd/images_background'
"""
python MANN.py --batch-size 16 --num-classes 5  --num-samples 50 --input-height 20 --input-width 20 --num-reads 4 --controller-size 200 --memory-locations 128 --memory-word-size 40 --learning-rate 1e-4 --iterations 100000 --path '/media/aiswarya/New Volume/My_works/ISI_contd/images_background'
"""