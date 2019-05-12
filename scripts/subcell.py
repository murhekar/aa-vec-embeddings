# Subcellular localization task

import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def main(modelFileName):
	AAs = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
	class_names = ['Cell.membrane', 'Cytoplasm', 'Endoplasmic.reticulum', 'Golgi.apparatus',
	 'Lysosome/Vacuole', 'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extracellular']

	# Map AA to index
	index = {}
	for i in range(len(AAs)):
		index[AAs[i]] = i
	for c in ['B','J','O','U','X','Z']:
		index[c] = 0

	# Map class to index
	class_index = {}
	for i in range(len(class_names)):
		class_index[class_names[i]] = i

	# params
	num_aa = 20 # should equal number of KV stored
	num_classes = 10 # no. of subcell locations
	max_seq_length = 50 # trim till this length
	embedding_dim = 19 

	# class vectors
	class_vectors = np.identity(num_classes)

	# Create train and test data
	f = open("../data/deeploc.fasta")
	t = f.readlines()

	# X - sequence, y - location.
	X_train = [] 
	X_test = [] 
	y_train = []
	y_test = []

	for i in range(len(t)):
		if i%2 == 0:
			seq = list(t[i+1].replace('\n',''))
			int_seq = [index[x] for x in seq]
			info = t[i].split()
			class_name = info[1].split("-")[0]
			class_vector = class_vectors[class_index[class_name]]	
			if len(info) == 3 and (info[2] == "test"): # test set
				X_test.append(int_seq)
				y_test.append(class_vector)
			else: # train set
				X_train.append(int_seq)
				y_train.append(class_vector)
	
	# first and last 250 
	X_train = [ x[:25]+x[-25:] for x in X_train]
	X_test = [ x[:25]+x[-25:] for x in X_test]
	X_train = pad_sequences(X_train, maxlen=max_seq_length, padding='post')
	X_test = pad_sequences(X_test, maxlen=max_seq_length, padding='post')
	
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)

	print X_train.shape, X_test.shape, y_train.shape, y_test.shape

	# load pretrained AA embeddings
	modelName = "../models/"+modelFileName+".kv"
	wv_model = KeyedVectors.load(modelName, mmap='r')
	embedding_weights = np.zeros((num_aa, embedding_dim))
	for i in range(len(AAs)):
		embedding_weights[i] = wv_model[AAs[i]]

	del wv_model

	one_hots = np.identity(num_aa)
	f = open("../matrices/BLOSUM62.txt", "r")
	lines = f.readlines()[6:] # remove comments
	blos = [[int(x) for x in row.split()[1:-4]] for row in lines[1:-4]]
	blosum = np.array(blos) # BLOSUM matrix	
	
	# build LSTM model
	model = Sequential()
	# train embeddings
	# embed_layer = Embedding(num_aa, embedding_dim, input_length=max_seq_length) 
	# pretrained embeddings
	embed_layer = Embedding(num_aa, embedding_dim, weights=[embedding_weights], input_length=max_seq_length, trainable=False)
	# one hot embeddings
	# embed_layer = Embedding(num_aa, num_aa, weights=[one_hots], input_length=max_seq_length, trainable=False)
	# BLOSUM62 row embeddings
	# embed_layer = Embedding(num_aa, num_aa, weights=[blosum], input_length=max_seq_length, trainable=False)

	model.add(embed_layer)
	model.add(LSTM(units=32, activation='tanh', recurrent_activation='hard_sigmoid'))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.1)

	model.save("../models/subcell_"+modelFileName+".mdl")

	print "Evaluating..."
	loss, accuracy = model.evaluate(X_test, y_test)
	print accuracy

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Usage: python subcell.py aa_vec_embedding_model_name"
		sys.exit()
	modelFileName = sys.argv[1]
	main(modelFileName)