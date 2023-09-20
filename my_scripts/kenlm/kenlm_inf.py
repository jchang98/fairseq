import kenlm
import time
import sys,os,re
import multiprocessing as mp

norm_space = re.compile(r" +")

def get_length(txt):
	txt = norm_space.split(txt)
	return len(txt)

def get_score(i):
	line = linputs[i]
	score = model.score(line,bos=True,eos=True) / get_length(line)
	return score

if __name__ == "__main__":
	model = kenlm.Model(sys.argv[1])
	inputs = [line.strip() for line in open(sys.argv[2])]
	
	core = 10
	pool = mp.Pool(core)
	length = len(inputs)
	s = time.time()
	res = pool.map_async(get_score,range(length))
	print("Running time{}".format(time.time()-s))
	with open(sys.argv[3],'w') as f_out:
		for i,r in enumerate(res.get()):
			score = r
			f_out.write("{}\t{}\n".format(i,score))