**Amino acid vector embeddings, similarity scores, and protein subcellular localization
**
The unique sequence of amino acids that make up a protein impart to it distinct physical and chemical properties. Inspired by ideas in NLP like word2vec and sequence-based models, we create vector embeddings of amino acids which encode contextual information and meaningful biochemical properties. We use these vector embeddings to compute substitution matrices for the problem of protein sequence alignment. We also use these embeddings together with sequence based models for the task of predicting protein subcellular localization.

Usage:
1.	Download datafiles:
	1. PDB. Download the PDB file from http://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt, and rename it to pdb.fasta.
	2. UniProt. Download the SwissProt file from https://www.uniprot.org/downloads, and rename it to uniprot.fasta.
	3. DeepLoc. Download the DeepLoc file from http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta, and rename it to deeploc.fasta
	4. Place these three files in a folder data/

2.	Extract amino acid sequences from PDB and UniProt datafiles:
		python extractSeq.py

3. 	Train amino acid vector embeddings:
		python models.py

4. 	Use embeddings to create and visualize substitution matrix, calculate relative entropy, view PCA and t-SNE:
		python usemodel.py

5.	Train model for subcellular classification, specify embedding model to be used:
		python subcell.py wvs_cbow_pdb_19
