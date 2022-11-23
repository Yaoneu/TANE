from utils import *
import time

t = 2 # threshold for second order relations
rep_dim = 100 # dimension of embedding vectors
tr = 0.9 # training ratio
th = 2 # number of out iterations

# network files
path = 'data/foursquare-twitter/'
inputfile1 = path + '5000ori_a.txt'
inputfile2 = path + '5000ori_b.txt'
anchorfile = path + 'anc.txt'

start=time.time()
print('TANE training ratio=', tr, 'dim=', rep_dim)
graph_sampling(anchorfile, path+'anc_train'+str(tr)+'.txt',
                      path+'anc_test'+str(tr)+'.txt', tr)
train_edge = read_edges(path+'anc_train'+str(tr)+'.txt')
test_edge = read_edges(path+'anc_test'+str(tr)+'.txt')

pre_emb, pre = run_embedding(inputfile1, inputfile2, an_list=train_edge, test_list=test_edge,
                    t=t, rep_dim=rep_dim, th=th, directed1=True, directed2=True)
# directed1 and directed2 mean if network1 and network2 are directed or undirected
running_time = time.time()-start
print('processing time=', running_time)
f = open(path+str(tr)+'result.txt', 'w', encoding='utf-8')
f.write("{} {}\n".format('structure' + str(tr), ' '.join([str(x) for x in pre])))
f.write("{} {}\n".format('structure with context' + str(tr), ' '.join([str(x) for x in pre])))
f.write("{} {}\n".format('running time:', running_time))
f.close()
