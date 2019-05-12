# script to write all sequences into a file

# PDB sequences
f = open("../data/pdb.fasta")
t = f.readlines()

w = open("../data/all_pdb.txt","w")

for line in t:
	if ">" in line:
		continue
	else:
		w.write(line)

f.close()
w.close()

# Uniprot sequences
f = open("../data/uniprot.fasta")
t = f.readlines()

w = open("../data/all_uniprot.txt", "w")

i = 0
while i < len(t):
	if ">" in t[i]:
		j = i+1
		seq = ""
		while (j < len(t)) and (">" not in t[j]):
			seq += t[j]
			j += 1
		# concatenate from i to j to get sequence
		w.write(seq.replace('\n','')+'\n')
	i = j

f.close()
w.close()