# encoding: utf-8
import graph
import random
import time
import numpy as np
import tane as ta
from sklearn.metrics.pairwise import cosine_similarity


def node_align(sn1_emb, sn2_emb, g1, g2, test_list, k):
    align_matrix = cosine_similarity(np.array(sn1_emb), np.array(sn2_emb))
    true_matrix = np.zeros([align_matrix.shape[0], align_matrix.shape[1]])

    for an in test_list:
        index1 = g1.look_up_dict[an[0]]
        index2 = g2.look_up_dict[an[1]]
        true_matrix[index1][index2] = 1
    print('test anchor num=', np.sum(true_matrix))

    pre = evaluate(align_matrix, true_matrix, k)

    return pre

def evaluate(alignment_matrix,true_matrix,k):
    nodenum1=true_matrix.shape[0]
    pre = [0] * k

    for i in range(nodenum1):
        if np.sum(true_matrix[i])>0:
            true_node=np.argwhere(true_matrix[i]==1)[0][0]
            sort1=np.argsort(-alignment_matrix[i])
            for j in range(len(sort1)):
                if sort1[j] == true_node and j < k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1
            sort2 = np.argsort(-alignment_matrix.T[true_node])
            for j in range(len(sort2)):
                if sort2[j] == i and j < k:
                    for p in range(j, k):
                        pre[p] = pre[p] + 1

    pre_n = np.array(pre) / (2 * np.sum(true_matrix))
    return pre_n

def graph_sampling(filename, train_filename, test_filename, r):
    f = open(filename, 'r')
    edges = []
    while 1:
        ed = f.readline()
        if ed == '':
            break
        edges.append(ed.split())
    edges_a_sample = random.sample(edges, int(r * len(edges)))
    edge_test = []
    for e in edges:
        if e not in edges_a_sample:
            edge_test.append(e)
    write_edge_file(edges_a_sample, train_filename)
    write_edge_file(edge_test, test_filename)

def write_edge_file(edge_list, file_name):
    f_edge = open(file_name, 'w', encoding='UTF-8')
    for i in range(len(edge_list)):
        f_edge.write("{} {}\n".format(str(edge_list[i][0]), str(edge_list[i][1])))
    f_edge.close()

def read_edges(filename):
    f = open(filename, 'r')
    edge_list = []
    while 1:
        l = f.readline()
        if l == '':
            break
        edge_list.append(l.split())
    return edge_list

def run_embedding(input_file1, input_file2, an_list, t, rep_dim, test_list, th,
                  directed1=True, directed2=True):
    t1 = time.time()
    g1 = graph.Graph()
    g2 = graph.Graph()
    print("Reading files......")
    g1.read_edgelist(filename=input_file1, directed=directed1)
    g2.read_edgelist(filename=input_file2, directed=directed2)
    model = ta.TANE(g1, g2, an_list, t, rep_dim=rep_dim, th=th)

    t2 = time.time()
    print("Running time is", t2 - t1)
    print("Embedding over.")

    model.get_embedding_directed()
    pre_emb = node_align(model.emb1, model.emb2, g1, g2, test_list, 30)
    pre = node_align(model.sn1_emb, model.sn2_emb, g1, g2, test_list, 30)

    print('structure precision@k=', pre_emb)
    print('structure with context precision@k=', pre_emb)

    return pre_emb, pre
