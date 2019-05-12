# Generate amino acid vector embeddings and store them as keyedvectors
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# write the protein sequences in word format, i.e. each AA is a word
def preProcessData(filename):
	f = open(filename,"r")
	seqs = f.readlines()
	print len(seqs)
	data = []
	for seq in seqs:
		aas = []
		for aa in seq.strip():
			aas.append(aa)
		data.append(aas)
	del seqs
	return data

def createModel(dataFileName, cbow_flag, dim_size):
	data = preProcessData(dataFileName)
	modelName = "wvs_" + ("cbow" if cbow_flag==1 else "sg") + "_" + ("pdb" if dataFileName == "../data/all_pdb.txt" else "uniprot") + "_" + str(dim_size)+".kv"
	print "---------------------------------------"
	print "Creating model "+modelName+"..."
	model = gensim.models.Word2Vec(data, min_count = 1, size = dim_size, window = 5, sg = (1-cbow_flag))
	print "Saving model "+modelName+"..."
	model.wv.save("../models/" + modelName) # save keyedvetors
	del model

def main():
	dim_size = 19
	# create cbow and skipgram models for both datasets
	for dataFileName in ["../data/all_pdb.txt", "../data/all_uniprot.txt"]:
		for cbow_flag in [0,1]:
			createModel(dataFileName, cbow_flag, dim_size)

if __name__ == '__main__':
	main()