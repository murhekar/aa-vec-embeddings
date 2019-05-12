# Script to use the generated models
import sys

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

AAs = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def getColorMap():
	def invert_dict(d): 
		inverse = dict() 
		for key in d: 
			# Go through the list that is saved in the dict:
			for item in d[key]:
				# Check if in the inverted dict the key exists
				if item not in inverse: 
					# If not create a new list
					inverse[item] = [key] 
				else: 
					inverse[item].append(key) 
		return inverse

	cMap = {'g': ['R','K','H','D','E'],'r': ['S','T','N','Q'],
	'b': ['A','V','I','L','M'],'black': ['F','Y','W'], 'pink': ['C', 'G', 'P']}
	invcMap = invert_dict(cMap)
	colormap = []
	for i in range(len(AAs)):
		colormap.extend(invcMap[AAs[i]])
	return colormap

# Create the new substitution matrix using cosine similarities
def createSubMat(modelFileName, wv_model):
	f = open("../matrices/BLOSUM62.txt", "r")
	lines = f.readlines()[6:] # remove comments
	blos = [[int(x) for x in row.split()[1:-4]] for row in lines[1:-4]]
	bl = np.array(blos) # BLOSUM matrix

	# new BLOSUM matrix
	newbl = [ [wv_model.similarity(AAs[i],AAs[j]) for j in range(len(AAs))] for i in range(len(AAs))]
	
	# normalize rows so that min and max of every row of m is min and max of BLOSUM
	normm = []
	for i in range(len(newbl)):
		a = float((max(bl[i]) - min(bl[i])))/float(max(newbl[i])-min(newbl[i]))
		b = float((min(bl[i])*max(newbl[i]) - max(bl[i])*min(newbl[i])))/float(max(newbl[i])-min(newbl[i]))
		normrow = [a*x+b for x in newbl[i]]
		normm.append(normrow)
	# make it symmetric to prevent numpy floating point errors
	for i in range(len(normm)):
		for j in range(len(normm)):
			if j>i:
				normm[i][j] = normm[j][i]
	normNewBl = np.array(normm) # normalized new substitution matrix

	f = open("../matrices/sm_" + modelFileName + ".txt", "w")
	for i in range(len(AAs)):
		for j in range(len(AAs)):
			f.write( str(normNewBl[i,j]) + " " )
		f.write('\n')
	f.close()

# Visualize the old and new blosum matrices
def visualizeBlosum(modelFileName):
	f = open("../matrices/BLOSUM62.txt", "r")
	lines = f.readlines()[6:] # remove comments
	blos = [[int(x) for x in row.split()[1:-4]] for row in lines[1:-4]]
	bl = np.array(blos) # BLOSUM matrix

	# display fig
	plt.matshow(bl)
	plt.colorbar()
	plt.savefig("../figs/blosum.png")
	plt.show()

	# New BLOSUM matrix
	f = open("../matrices/sm_" + modelFileName + ".txt", "r")
	lines = f.readlines()
	lines = [lines[i] for i in range(len(lines))]
	newbl = [[float(x) for x in row.split()] for row in lines] # new BLOSUM matrix

	# display fig
	plt.matshow(newbl)
	plt.colorbar()
	plt.savefig("../figs/sm_"+modelFileName+".png")
	plt.show()

# Calculate the relative entropy of the old and new blosum matrices
def relativeEntropy(modelFileName):
	# p - background probs, q - substitution scores
	def calculation(p, q):
		H = 0 
		for i in range(len(p)):
			for j in range(len(p)):
				H += (2**(float)(q[i][j]))*p[i]*p[j]*q[i][j]
		return H

	back_probs = [0.05 for i in range(20)] # background probabilities, uniform for now
	f = open("../matrices/BLOSUM62.txt", "r")
	lines = f.readlines()[6:] # remove comments
	blos = [[int(x) for x in row.split()[1:-4]] for row in lines[1:-4]]
	bl = np.array(blos) # BLOSUM matrix	
	print "Relative Entropy of BLOSUM: ", calculation(back_probs, bl)
	
	f = open("../matrices/sm_" + modelFileName + ".txt", "r")
	lines = f.readlines() # remove comments
	lines = [lines[i] for i in range(len(lines))]
	newbl = [[float(x) for x in row.split()] for row in lines] # new BLOSUM matrix
	print "Relative Entropy of new BLOSUM: ", calculation(back_probs, newbl)

# Perform principal component analysis and view the vectors projected down to 2 dimension
def viewPCA(modelFileName, wv_model, colormap):
	X = [wv_model[x] for x in AAs]
	pca = PCA(n_components=2)
	X_pca = pca.fit(X).transform(X)
	
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap)
	for label, x, y in zip(AAs, X_pca[:, 0], X_pca[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

	plt.savefig("../figs/pca_"+modelFileName+".png")
	plt.show()

# Perform dimensionality reduction using t-SNE and view the vectors projected down to 2 dimension
def viewtSNE(modelFileName, wv_model, colormap):
	X = [wv_model[x] for x in AAs]
	tsne = TSNE(n_components=2, perplexity=6)
	X_tsne = tsne.fit_transform(X)

	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colormap)
	for label, x, y in zip(AAs, X_tsne[:, 0], X_tsne[:, 1]):
		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

	plt.savefig("../figs/tsne_"+modelFileName+".png")
	plt.show()

def main(modelFileName):
	# modelName = "../../backup/oldmodels/wvs_cbow_dim16.kv" # erase
	modelName = "../models/"+modelFileName+".kv"
	wv_model = KeyedVectors.load(modelName, mmap='r')

	print "Creating new blosum..."
	createSubMat(modelFileName, wv_model)
	print "Visualizing new blosum..."
	visualizeBlosum(modelFileName)
	print "Calculate relative entropy..."
	relativeEntropy(modelFileName)
	colormap = getColorMap()
	print "View PCA..."
	viewPCA(modelFileName, wv_model, colormap)
	print "View t-SNE..."
	viewtSNE(modelFileName, wv_model, colormap)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print "Usage: python usemodel.py modelFileName"
		sys.exit()
	modelFileName = sys.argv[1]
	main(modelFileName)