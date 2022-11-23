# encoding: utf-8
import tensorflow as tf
from collections import defaultdict
import random
import math
import numpy as np
import os
import time
import sys
from tensorflow.python.framework import ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class _TANE(object):
    def __init__(self, graph1, graph2, anchor, out_iter,
                 rep_dim=32, batch_size=1000,an_batch_size=1000,
                 negative_ratio=5, t=3):
        self.g1 = graph1
        self.g2 = graph2
        self.anchor = anchor
        self.g_a_list = self.anchor_list(anchor)
        self.rep_dim = rep_dim
        self.T_dim = 16
        self.batch_size = batch_size
        self.an_batch_size = an_batch_size
        self.negative_ratio = negative_ratio
        self.negative_ratio_att = 1

        self.th = t  # threshold of out degree
        self.out_iter = out_iter
        self.node_size1 = graph1.G.number_of_nodes()
        self.node_size2 = graph2.G.number_of_nodes()
        self.cur_epoch_sn1 = 0
        self.cur_epoch_att = 0
        # the second line of algorithm
        self.relation_generating()
        # generating negative samples
        self.gen_sampling_table()
        self.second_neg_sample()
        self.anchor_neg_list()
        self.sess = tf.compat.v1.Session()

        self.build_graph_emb()
        self.build_graph_att()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def build_graph_emb(self):
        # embedding
        tf.compat.v1.disable_eager_execution()
        self.h = tf.compat.v1.placeholder(tf.int32, [None])
        self.t = tf.compat.v1.placeholder(tf.int32, [None])
        self.sign1 = tf.compat.v1.placeholder(tf.float32, [None])
        self.sign2 = tf.compat.v1.placeholder(tf.float32, [None])
        self.sign3 = tf.compat.v1.placeholder(tf.float32, [None])
        self.wf = tf.compat.v1.placeholder(tf.float32, [None])
        self.ws = tf.compat.v1.placeholder(tf.float32, [None])

        cur_seed = random.getrandbits(32)
        ini = tf.initializers.glorot_uniform(seed=cur_seed)
        self.sn1_emb = tf.compat.v1.get_variable(name="sn1embedding",
                                                 shape=[self.node_size1, self.rep_dim],
                                                 initializer=ini)
        self.sn1_parent = tf.compat.v1.get_variable(name="sn1_parent_contexts",
                                                    shape=[self.node_size1, self.rep_dim],
                                                    initializer=ini)
        self.sn1_child = tf.compat.v1.get_variable(name="sn1_child_context",
                                                   shape=[self.node_size1, self.rep_dim],
                                                   initializer=ini)
        self.sn2_emb = tf.compat.v1.get_variable(name="sn2embedding",
                                                 shape=[self.node_size2, self.rep_dim],
                                                 initializer=ini)
        self.sn2_parent = tf.compat.v1.get_variable(name="sn2_parent_contexts",
                                                    shape=[self.node_size2, self.rep_dim],
                                                    initializer=ini)
        self.sn2_child = tf.compat.v1.get_variable(name="sn2_child_context",
                                                   shape=[self.node_size2, self.rep_dim],
                                                   initializer=ini)
        # embedding learning
        # sn1
        h_emb_sn1 = tf.nn.embedding_lookup(self.sn1_emb, self.h)
        h_c_context_sn1 = tf.nn.embedding_lookup(self.sn1_child, self.h)
        t_emb_sn1 = tf.nn.embedding_lookup(self.sn1_emb, self.t)
        t_p_context_sn1 = tf.nn.embedding_lookup(self.sn1_parent, self.t)

        # sn2
        h_emb_sn2 = tf.nn.embedding_lookup(self.sn2_emb, self.h)
        h_c_context_sn2 = tf.nn.embedding_lookup(self.sn2_child, self.h)
        t_emb_sn2 = tf.nn.embedding_lookup(self.sn2_emb, self.t)
        t_p_context_sn2 = tf.nn.embedding_lookup(self.sn2_parent, self.t)

        # anchor
        h_emb_an = tf.nn.embedding_lookup(self.sn1_emb, self.h)
        h_p_context_an = tf.nn.embedding_lookup(self.sn1_parent, self.h)
        h_c_context_an = tf.nn.embedding_lookup(self.sn1_child, self.h)
        t_emb_an = tf.nn.embedding_lookup(self.sn2_emb, self.t)
        t_p_context_an = tf.nn.embedding_lookup(self.sn2_parent, self.t)
        t_c_context_an = tf.nn.embedding_lookup(self.sn2_child, self.t)

        # SN1 loss
        loss_sn1_1 = -tf.reduce_mean((self.sign2*self.wf+self.sign3*self.ws) * tf.math.log_sigmoid(self.sign1 * tf.reduce_sum(tf.multiply(
            h_emb_sn1, t_p_context_sn1), axis=1)))

        loss_sn1_2 = -tf.reduce_mean((self.sign2*self.wf+self.sign3*self.ws) * tf.math.log_sigmoid(self.sign1 * tf.reduce_sum(tf.multiply(
            t_emb_sn1, h_c_context_sn1), axis=1)))

        self.loss_social1 = loss_sn1_1 + loss_sn1_2

        # SN2 loss
        loss_sn2_1 = -tf.reduce_mean((self.sign2*self.wf+self.sign3*self.ws) * tf.math.log_sigmoid(self.sign1 * tf.reduce_sum(tf.multiply(
            h_emb_sn2, t_p_context_sn2), axis=1)))
        loss_sn2_2 = -tf.reduce_mean((self.sign2*self.wf+self.sign3*self.ws) * tf.math.log_sigmoid(self.sign1 * tf.reduce_sum(tf.multiply(
            t_emb_sn2, h_c_context_sn2), axis=1)))

        self.loss_social2 = loss_sn2_1 + loss_sn2_2

        # anchor loss
        self.loss_anchor_pf = -self.sign2 * tf.reduce_mean(self.wf * tf.compat.v1.log_sigmoid(
            self.sign1 * tf.reduce_sum(tf.multiply(h_emb_an, t_p_context_an), axis=1))) - \
                              self.sign3 * tf.reduce_mean(self.wf * tf.compat.v1.log_sigmoid(
            self.sign1 * tf.reduce_sum(tf.multiply(t_emb_an, h_p_context_an), axis=1)))
        self.loss_anchor_cf = -self.sign2 * tf.reduce_mean(self.wf * tf.compat.v1.log_sigmoid(
            self.sign1 * tf.reduce_sum(tf.multiply(h_emb_an, t_c_context_an), axis=1))) - \
                              self.sign3 * tf.reduce_mean(self.wf * tf.compat.v1.log_sigmoid(
            self.sign1 * tf.reduce_sum(tf.multiply(t_emb_an, h_c_context_an), axis=1)))

        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=0.001, global_step=self.cur_epoch_sn1, decay_steps=20,
            decay_rate=0.9, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        tf.compat.v1.add_to_collection("train_vars_emb_sn1",
                                       [self.sn1_emb, self.sn1_parent,self.sn2_child])
        tf.compat.v1.add_to_collection("train_vars_emb_sn2",
                                       [self.sn2_emb, self.sn2_parent,self.sn2_child])
        tf.compat.v1.add_to_collection("train_vars_emb_an",
                                       [self.sn1_emb, self.sn1_parent,self.sn1_child,
                                        self.sn2_emb, self.sn2_parent,self.sn2_child])
        an_train_vars = tf.compat.v1.get_collection("train_vars_emb_an")
        sn1_train_vars = tf.compat.v1.get_collection("train_vars_emb_sn1")
        sn2_train_vars = tf.compat.v1.get_collection("train_vars_emb_sn2")

        self.train_op_anpf = optimizer.minimize(self.loss_anchor_pf,
                                                     var_list=an_train_vars)

        self.train_op_ancf = optimizer.minimize(self.loss_anchor_cf,
                                                     var_list=an_train_vars)

        self.train_op_sn1 = optimizer.minimize(self.loss_social1,
                                                    var_list=sn1_train_vars)

        self.train_op_sn2 = optimizer.minimize(self.loss_social2,
                                                    var_list=sn2_train_vars)
        print("Embedding Initialized")

    def build_graph_att(self):
        #attention计算相关
        tf.compat.v1.disable_eager_execution()
        self.node1 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node2 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node12 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node21 = tf.compat.v1.placeholder(tf.int32, [None])
        self.child1 = tf.compat.v1.placeholder(tf.int32, [None])
        self.child2 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node_neg1 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node_neg2 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node_neg12 = tf.compat.v1.placeholder(tf.int32, [None])
        self.node_neg21 = tf.compat.v1.placeholder(tf.int32, [None])
        self.child_neg1 = tf.compat.v1.placeholder(tf.int32, [None])
        self.child_neg2 = tf.compat.v1.placeholder(tf.int32, [None])

        cur_seed = random.getrandbits(32)
        ini = tf.initializers.glorot_uniform(seed=cur_seed)
        self.acf1 = tf.compat.v1.get_variable(name="sn1_acf", shape=[2*self.rep_dim, 1],
                                              initializer=ini)
        self.acf2 = tf.compat.v1.get_variable(name="sn2_acf", shape=[2*self.rep_dim, 1],
                                              initializer=ini)
        self.acs1 = tf.compat.v1.get_variable(name="sn1_acs", shape=[2*self.rep_dim, 1],
                                              initializer=ini)
        self.acs2 = tf.compat.v1.get_variable(name="sn2_acs", shape=[2*self.rep_dim, 1],
                                              initializer=ini)
        self.aaf = tf.compat.v1.get_variable(name="anchor_f", shape=[2*self.rep_dim, 1],
                                             initializer=ini)
        self.aas = tf.compat.v1.get_variable(name="anchor_s", shape=[2*self.rep_dim, 1],
                                             initializer=ini)
        # attention learning
        node1=tf.nn.embedding_lookup(self.sn1_emb,self.node1)
        node12=tf.nn.embedding_lookup(self.sn1_emb,self.node12)
        node1_child=tf.nn.embedding_lookup(self.sn1_emb,self.child1)
        node2=tf.nn.embedding_lookup(self.sn2_emb,self.node2)
        node21=tf.nn.embedding_lookup(self.sn2_emb,self.node21)
        node2_child=tf.nn.embedding_lookup(self.sn2_emb,self.child2)
        node_neg1=tf.nn.embedding_lookup(self.sn1_emb,self.node_neg1)
        node_neg12=tf.nn.embedding_lookup(self.sn1_emb,self.node_neg12)
        node_neg_child1=tf.nn.embedding_lookup(self.sn1_emb,self.child_neg1)
        node_neg2=tf.nn.embedding_lookup(self.sn2_emb,self.node_neg2)
        node_neg21 = tf.nn.embedding_lookup(self.sn2_emb, self.node_neg21)
        node_neg_child2=tf.nn.embedding_lookup(self.sn2_emb,self.child_neg2)

        att1=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node1_child,node1], 1),self.acf1)) * node1_child, axis=0)
        att2=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node2_child,node2], 1),self.acf2)) * node2_child, axis=0)
        att_neg1=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child1,node_neg1], 1),self.acf1)) * node_neg_child1, axis=0)
        att_neg2=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child2,node_neg2], 1),self.acf2)) * node_neg_child2,axis=0)
        att12=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node2_child,node12], 1),self.aaf)) * node2_child, axis=0)
        att21=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node1_child,node21], 1),self.aaf)) * node1_child, axis=0)
        att_neg12=tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child2,node_neg12], 1),self.aaf)) * node_neg_child2, axis=0)
        att_neg21 = tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child1, node_neg21], 1),self.aaf))* node_neg_child1, axis=0)

        e1=att1+att12
        e2=att2+att21
        e_neg1=att_neg1+att_neg12
        e_neg2=att_neg2+att_neg21

        self.loss_att=tf.square(tf.sigmoid(tf.reduce_sum(tf.multiply(
            e1, e2)))-1)+tf.square(tf.sigmoid(tf.reduce_sum(tf.multiply(
            e_neg1, e_neg2)))-0)

        att1_s = tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node1_child, node1], 1),self.acs1)) * node1_child, axis=0)
        att2_s = tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node2_child, node2], 1),self.acs2)) * node2_child, axis=0)
        att_neg1_s = tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child1,node_neg1], 1),self.acs1)) * node_neg_child1, axis=0)
        att_neg2_s = tf.reduce_sum(tf.sigmoid(tf.matmul(tf.concat(
            [node_neg_child2, node_neg2], 1),self.acs2)) * node_neg_child2, axis=0)

        e1_s = att1_s
        e2_s = att2_s
        e_neg1_s = att_neg1_s
        e_neg2_s = att_neg2_s
        self.loss_att_second = tf.square(tf.sigmoid(tf.reduce_sum(tf.multiply(
            e1_s, e2_s))) - 1) + tf.square(tf.sigmoid(tf.reduce_sum(tf.multiply(
            e_neg1_s, e_neg2_s))) - 0)

        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=0.001, global_step=self.cur_epoch_att, decay_steps=10,
            decay_rate=0.9, staircase=False)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        tf.compat.v1.add_to_collection("train_vars_attention_first",
                                       [self.acf1, self.acf2,
                                        self.aaf])
        tf.compat.v1.add_to_collection("train_vars_attention_second",
                                       [self.acs1, self.acs2])

        att_train_vars = tf.compat.v1.get_collection("train_vars_attention_first")
        self.train_op_att = optimizer.minimize(self.loss_att, var_list=att_train_vars)
        att_train_vars_second = tf.compat.v1.get_collection("train_vars_attention_second")
        self.train_op_att_second = optimizer.minimize(self.loss_att_second, var_list=att_train_vars_second)
        print('t=',self.th,"attention Initialized,embedding dimension is", self.rep_dim)

    def relation_generating(self):
        # second order relation generation with nodes in g1 and g2
        npf1 = defaultdict(list)
        ncf1 = defaultdict(list)
        nps1 = defaultdict(list)
        ncs1 = defaultdict(list)
        npf2 = defaultdict(list)
        ncf2 = defaultdict(list)
        nps2 = defaultdict(list)
        ncs2 = defaultdict(list)
        look_up1 = self.g1.look_up_dict
        edges1 = [(look_up1[x[0]], look_up1[x[1]]) for x in self.g1.G.edges()]
        look_up2 = self.g2.look_up_dict
        edges2 = [(look_up2[x[0]], look_up2[x[1]]) for x in self.g2.G.edges()]
        for ed in edges1:
            npf1[ed[0]].append(ed[1])
            ncf1[ed[1]].append(ed[0])
        for ed in edges2:
            npf2[ed[0]].append(ed[1])
            ncf2[ed[1]].append(ed[0])
        for key in npf1.keys():
            fparent = npf1.get(key)
            if len(fparent) < self.th:
                for fp in fparent:
                    sparent = npf1.get(fp)
                    if sparent:
                        for sp in sparent:
                            nps1[key].append(sp)
        for key in ncf1.keys():
            fchild = ncf1.get(key)
            if len(fchild) < self.th:
                for fc in fchild:
                    schild = ncf1.get(fc)
                    if schild:
                        for sc in schild:
                            ncs1[key].append(sc)
        for key in npf2.keys():
            fparent = npf2.get(key)
            if len(fparent) < self.th:
                for fp in fparent:
                    sparent = npf2.get(fp)
                    if sparent:
                        for sp in sparent:
                            nps2[key].append(sp)
        for key in ncf2.keys():
            fchild = ncf2.get(key)
            if len(fchild) < self.th:
                for fc in fchild:
                    schild = ncf2.get(fc)
                    if schild:
                        for sc in schild:
                            ncs2[key].append(sc)
        # first order relations and second order relations finished
        second_parent1 = []
        second_parent2 = []
        for key in nps1.keys():
            sparent = nps1.get(key)
            for sp in sparent:
                second_parent1.append([key, sp])
        for key in nps2.keys():
            sparent = nps2.get(key)
            for sp in sparent:
                second_parent2.append([key, sp])
        second_child1 = []
        second_child2 = []
        for key in ncs1.keys():
            schild = ncs1.get(key)
            for sc in schild:
                second_child1.append([sc, key])  # all relations in list are nodes in g1 to nodes in g2
        for key in ncs2.keys():
            schild = ncs2.get(key)
            for sc in schild:
                second_child2.append([sc, key])
        # Eliminate duplication
        second_sn1 = second_parent1 + second_child1
        self.second_sn1 = list(set([tuple(t) for t in second_sn1]))
        second_sn2 = second_parent2 + second_child2
        self.second_sn2 = list(set([tuple(t) for t in second_sn2]))
        print("SN1 second relationship number",len(self.second_sn1))
        print("SN2 second relationship number",len(self.second_sn2))
        # generating second order relation list
        self.ncs1=self.node_size1*[[]]
        self.ncs2=self.node_size2*[[]]
        for key in ncs1.keys():
            child1=ncs1.get(key)
            self.ncs1[key]=child1
        for key in ncs2.keys():
            child2=ncs2.get(key)
            self.ncs2[key]=child2

        # anchor对应其partner的first parent, second parent, first child and second child
        self.an_pf1 = []
        self.an_pf2 = []
        self.an_cf1 = []
        self.an_cf2 = []
        for an in self.g_a_list:
            n_sn1 = an[0]
            n_sn2 = an[1]
            pf_sn1 = npf1.get(n_sn1)
            cf_sn1 = ncf1.get(n_sn1)
            pf_sn2 = npf2.get(n_sn2)
            cf_sn2 = ncf2.get(n_sn2)
            if pf_sn1:
                for pf1 in pf_sn1:
                    self.an_pf2.append([pf1, n_sn2])
            if cf_sn1:
                for cf1 in cf_sn1:
                    self.an_cf2.append([cf1, n_sn2])
            if pf_sn2:
                for pf2 in pf_sn2:
                    self.an_pf1.append([n_sn1, pf2])
            if cf_sn2:
                for cf2 in cf_sn2:
                    self.an_cf1.append([n_sn1, cf2])
        print("Relation generated")

    def second_neg_sample(self):
        node1=[]
        node2=[]
        table_size = 1e8
        power = 0.75
        node_degree1 = np.zeros(self.node_size1)
        node_degree2 = np.zeros(self.node_size2)
        for ss1 in self.second_sn1:
            node_degree1[ss1[0]]=node_degree1[ss1[0]]+1
        for ss2 in self.second_sn2:
            node_degree2[ss2[0]]=node_degree2[ss2[0]]+1
        norm1 = sum([math.pow(node_degree1[i], power) for i in range(len(node1))])
        norm2 = sum([math.pow(node_degree2[i], power) for i in range(len(node2))])
        self.second_sampling_table1 = np.zeros(int(table_size), dtype=np.uint32)
        self.second_sampling_table2 = np.zeros(int(table_size), dtype=np.uint32)
        p1 = 0
        i1 = 0
        for j in range(len(node1)):
            p1 += float(math.pow(node_degree1[j], power)) / norm1
            while i1 < table_size and float(i1) / table_size < p1:
                self.second_sampling_table1[i1] = j
                i1 += 1
        p2 = 0
        i2 = 0
        for j in range(len(node2)):
            p2 += float(math.pow(node_degree2[j], power)) / norm2
            while i2 < table_size and float(i2) / table_size < p2:
                self.second_sampling_table2[i2] = j
                i2 += 1
        print("Second-order negative sampling finished")

    def anchor_list(self,edge_list):
        new_list=[]
        look_back1 = self.g1.look_back_list
        look_back2 = self.g2.look_back_list
        for el in edge_list:
            for i in range(len(look_back1)):
                if el[0]==look_back1[i]:
                    sn1=i
            for j in range(len(look_back2)):
                if el[1]==look_back2[j]:
                    sn2=j
            new_list.append([sn1,sn2])
        return new_list

    def anchor_neg_list(self):
        self.neg_an_order1 = []
        self.neg_an_order2 = []
        self.an_order1=[]
        self.an_order2=[]
        for i in range(len(self.anchor)):
            if self.g1.G.in_edges(self.anchor[0]) != [] and self.g2.G.in_edges(self.anchor[1]) != []:
                self.an_order1.append(self.g_a_list[i])

            if self.ncs1[self.g_a_list[i][0]] !=[] and self.ncs2[self.g_a_list[i][1]]!=[]:
                self.an_order2.append(self.g_a_list[i])
        for j in range(self.negative_ratio):
            for i in range(len(self.an_order1)):
                while 1:
                    index = random.randint(0, min(self.node_size1 - 1, self.node_size2 - 1))
                    if index != self.an_order1[i][1] and (
                            [self.an_order1[i][0], index] not in self.neg_an_order1) and \
                            self.g2.G.in_edges(self.g2.look_back_list[index]) !=[]:
                        break
                self.neg_an_order1.append([self.an_order1[i][0], index])
                while 1:
                    index = random.randint(0, min(self.node_size1 - 1, self.node_size2 - 1))
                    if index != self.an_order1[i][0] and (
                            [index, self.an_order1[i][1]] not in self.neg_an_order1) and \
                            self.g1.G.in_edges(self.g1.look_back_list[index]) != []:
                        break
                self.neg_an_order1.append([index, self.an_order1[i][1]])
        for j in range(self.negative_ratio):
            for i in range(len(self.an_order2)):
                while 1:
                    index = random.randint(0, min(self.node_size1 - 1, self.node_size2 - 1))
                    if index != self.an_order2[i][1] and (
                            [self.an_order2[i][0], index] not in self.neg_an_order2) and \
                            self.ncs2[index]!=[]:
                        break
                self.neg_an_order2.append([self.an_order2[i][0], index])
                while 1:
                    index = random.randint(0, min(self.node_size1 - 1, self.node_size2 - 1))
                    if index != self.an_order2[i][0] and (
                            [index, self.an_order2[i][1]] not in self.neg_an_order2) and \
                            self.ncs1[index]!=[]:
                        break
                self.neg_an_order2.append([index, self.an_order2[i][1]])
        print("order1 length",len(self.an_order1))
        print("order2 length",len(self.an_order2))

    def train_one_epoch_attention(self):
        sum_loss=0

        batches=self.batch_iter_att()
        for batch in batches:
            sn1_node, sn1_child, sn2_node, sn2_child, sn1_node_neg, \
            sn1_child_neg, sn2_node_neg, sn2_child_neg, sn12_node, \
            sn21_node, sn12_node_neg, sn21_node_neg = batch
            feed_dict = {
                self.node1: sn1_node,
                self.child1: sn1_child,
                self.node2: sn2_node,
                self.child2:sn2_child,
                self.node_neg1:sn1_node_neg,
                self.child_neg1:sn1_child_neg,
                self.node_neg2:sn2_node_neg,
                self.child_neg2:sn2_child_neg,
                self.node12:sn12_node,
                self.node21:sn21_node,
                self.node_neg12:sn12_node_neg,
                self.node_neg21:sn21_node_neg,
            }
            _, cur_loss_att = self.sess.run([self.train_op_att, self.loss_att], feed_dict)
            sum_loss=cur_loss_att+sum_loss

        batches = self.batch_iter_att_second()
        for batch in batches:
            sn1_node, sn1_child, sn2_node, sn2_child, sn1_node_neg, \
            sn1_child_neg, sn2_node_neg, sn2_child_neg, sn12_node, \
            sn21_node, sn12_node_neg, sn21_node_neg = batch
            feed_dict = {
                self.node1: sn1_node,
                self.child1: sn1_child,
                self.node2: sn2_node,
                self.child2: sn2_child,
                self.node_neg1: sn1_node_neg,
                self.child_neg1: sn1_child_neg,
                self.node_neg2: sn2_node_neg,
                self.child_neg2: sn2_child_neg,
                self.node12: sn12_node,
                self.node21: sn21_node,
                self.node_neg12: sn12_node_neg,
                self.node_neg21: sn21_node_neg,
            }
            _, cur_loss_att_second = self.sess.run([self.train_op_att_second,
                                             self.loss_att_second], feed_dict)
            sum_loss = cur_loss_att_second + sum_loss
        # print('Attention epoch:{} processing time :{!s}  loss:{!s}'.format(
        #     self.cur_epoch_att,te - ts, sum_loss))
        self.cur_epoch_att += 1
        return sum_loss

    def train_one_epoch_embedding(self):
        sum_loss1 = 0.0
        ts=time.time()

        batches1 = self.batch_iter1()
        for batch in batches1:
            h1, t1, sign1, sign2, sign3, wf, ws = batch
            feed_dict = {
                self.h: h1,
                self.t: t1,
                self.sign1: sign1,
                self.sign2: sign2,
                self.sign3: sign3,
                self.wf: wf,
                self.ws: ws,
            }
            _, cur_loss_sn1 = self.sess.run([self.train_op_sn1, self.loss_social1], feed_dict)
            sum_loss1 = sum_loss1 + cur_loss_sn1

        batches2 = self.batch_iter2()
        for batch in batches2:
            h2, t2, sign1, sign2, sign3, wf,ws = batch
            feed_dict = {
                self.h: h2,
                self.t: t2,
                self.sign1: sign1,
                self.sign2: sign2,
                self.sign3: sign3,
                self.wf: wf,
                self.ws: ws,
            }
            _, cur_loss_sn2 = self.sess.run([self.train_op_sn2, self.loss_social2], feed_dict)
            sum_loss1 = sum_loss1 + cur_loss_sn2

        batches_apf = self.batch_iter_anpf()
        for batch in batches_apf:
            h_a, t_a, sign1, sign2, sign3,w = batch
            feed_dict = {
                self.h: h_a,
                self.t: t_a,
                self.sign1: sign1,
                self.wf: w,
                self.sign2: sign2,
                self.sign3: sign3,
            }
            _, cur_loss_an1 = self.sess.run([self.train_op_anpf, self.loss_anchor_pf],
                                            feed_dict)
            sum_loss1 = sum_loss1 + cur_loss_an1

        batches_acf = self.batch_iter_ancf()
        for batch in batches_acf:
            h_a, t_a, sign1, sign2, sign3,w = batch
            feed_dict = {
                self.h: h_a,
                self.t: t_a,
                self.sign1: sign1,
                self.wf: w,
                self.sign2: sign2,
                self.sign3: sign3,
            }
            _, cur_loss_ancf1 = self.sess.run([self.train_op_ancf, self.loss_anchor_cf],
                                              feed_dict)
            sum_loss1 = sum_loss1 + cur_loss_ancf1

        te = time.time()
        if self.cur_epoch_sn1%100==0:
            print('Epoch:{} processing time :{!s} sum of loss:{!s}'.format(
                self.cur_epoch_sn1, te - ts, sum_loss1))
        self.cur_epoch_sn1 += 1
        return sum_loss1

    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s

    def weight_intra(self, att, index1, index2):
        # computing weights, index1 is node in g1 and index2 is node in g2
        if self.out_iter == 0:
            return 1.0
        else:
            if att == 'af1':
                return self.sigmoid(np.dot(np.concatenate(self.sn1_combine[index1],
                                                    self.sn1_combine[index2]), self.af1))
            elif att =='af2':
                return self.sigmoid(np.dot(np.concatenate(self.sn2_combine[index1],
                                                      self.sn2_combine[index2]), self.af2))
            elif att == 'as1':
                return self.sigmoid(np.dot(np.concatenate(self.sn1_combine[index1],
                                                      self.sn1_combine[index2]), self.as1))
            elif att == 'as2':
                return self.sigmoid(np.dot(np.concatenate(self.sn2_combine[index1],
                                                      self.sn2_combine[index2]), self.as2))
            elif att == 'af12':
                return self.sigmoid(np.dot(np.concatenate(self.sn1_combine[index1],
                                                      self.sn2_combine[index2]), self.af))
            elif att == 'af21':
                return self.sigmoid(np.dot(np.concatenate(self.sn2_combine[index2],
                                                      self.sn1_combine[index1]), self.af))
            else:
                print("attention category error")
                sys.exit(1)

    def embedding(self):
        # generating embedding for attention learning
        emb1 = self.sn1_emb.eval(session=self.sess)
        emb1_p = self.sn1_parent.eval(session=self.sess)
        emb1_c = self.sn1_child.eval(session=self.sess)
        emb2 = self.sn2_emb.eval(session=self.sess)
        emb2_p = self.sn2_parent.eval(session=self.sess)
        emb2_c = self.sn2_child.eval(session=self.sess)
        self.sn1_combine = np.concatenate((emb1, emb1_p, emb1_c), axis=1)
        self.sn2_combine = np.concatenate((emb2, emb2_p, emb2_c), axis=1)
        self.af1 = self.acf1.eval(session=self.sess)
        self.af2 = self.acf2.eval(session=self.sess)
        self.as1 = self.acs1.eval(session=self.sess)
        self.as2 = self.acs2.eval(session=self.sess)
        self.af = self.aaf.eval(session=self.sess)

    def batch_iter_att(self):
        s_index = 0
        pos_data_size=len(self.an_order1)
        shuffle_indices = np.random.permutation(np.arange(pos_data_size))

        while s_index<pos_data_size:
            h=self.an_order1[shuffle_indices[s_index]][0]
            t=self.an_order1[shuffle_indices[s_index]][1]
            # edges
            sn1_inedges=self.g1.G.in_edges(self.g1.look_back_list[h])
            sn2_inedges=self.g2.G.in_edges(self.g2.look_back_list[t])
            # split edges, first node is child
            sn1_child=[]
            sn2_child=[]
            for ie1 in sn1_inedges:
                sn1_child.append(self.g1.look_up_dict[ie1[0]])
            for ie2 in sn2_inedges:
                sn2_child.append(self.g2.look_up_dict[ie2[0]])
            sn1_node=[h]*len(sn1_child)
            sn2_node=[t]*len(sn2_child)
            sn12_node=[h]*len(sn2_child)
            sn21_node=[t]*len(sn1_child)

            neg=random.choice(self.neg_an_order1)
            # negative edges
            sn1_child_negedges=self.g1.G.in_edges(self.g1.look_back_list[neg[0]])
            sn2_child_negedges=self.g2.G.in_edges(self.g2.look_back_list[neg[1]])

            sn1_child_neg =[]
            sn2_child_neg =[]
            for scn1 in sn1_child_negedges:
                sn1_child_neg.append(self.g1.look_up_dict[scn1[0]])
            for scn2 in sn2_child_negedges:
                sn2_child_neg.append(self.g2.look_up_dict[scn2[0]])

            sn1_node_neg=[neg[0]]*len(sn1_child_neg)
            sn2_node_neg=[neg[1]]*len(sn2_child_neg)
            sn12_node_neg=[neg[0]]*len(sn2_child_neg)
            sn21_node_neg=[neg[1]]*len(sn1_child_neg)

            yield sn1_node,sn1_child,sn2_node,sn2_child,sn1_node_neg,\
                  sn1_child_neg,sn2_node_neg,sn2_child_neg,sn12_node,\
                  sn21_node,sn12_node_neg,sn21_node_neg
            s_index = s_index+1

    def batch_iter_att_second(self):
        s_index = 0
        pos_data_size=len(self.an_order2)
        shuffle_indices = np.random.permutation(np.arange(pos_data_size))

        while s_index<pos_data_size:
            h=self.an_order2[shuffle_indices[s_index]][0]
            t=self.an_order2[shuffle_indices[s_index]][1]
            sn1_child=self.ncs1[h]
            sn2_child=self.ncs2[t]
            sn1_node = [h] * len(sn1_child)
            sn2_node = [t] * len(sn2_child)
            sn12_node = [h] * len(sn2_child)
            sn21_node = [t] * len(sn1_child)

            neg=random.choice(self.neg_an_order2)
            sn1_child_neg=self.ncs1[neg[0]]
            sn2_child_neg=self.ncs2[neg[1]]
            sn1_node_neg = [neg[0]] * len(sn1_child_neg)
            sn2_node_neg = [neg[1]] * len(sn2_child_neg)
            sn12_node_neg = [neg[0]] * len(sn2_child_neg)
            sn21_node_neg = [neg[1]] * len(sn1_child_neg)

            yield sn1_node,sn1_child,sn2_node,sn2_child,sn1_node_neg,\
                  sn1_child_neg,sn2_node_neg,sn2_child_neg,sn12_node,\
                  sn21_node,sn12_node_neg,sn21_node_neg
            s_index = s_index+1

    def batch_iter1(self):
        look_up1 = self.g1.look_up_dict
        table_size = 1e8
        numNodes1 = self.node_size1

        edges1 = [(look_up1[x[0]], look_up1[x[1]]) for x in self.g1.G.edges()]
        data_size1 = self.g1.G.number_of_edges()
        data_size2 = len(self.second_sn1)
        data_size = data_size1 + data_size2
        shuffle_indices1 = np.random.permutation(np.arange(data_size1))
        shuffle_indices2 = np.random.permutation(np.arange(data_size2))

        # positive or negative mod
        mod1 = 0
        mod_size = 1 + self.negative_ratio
        h1 = []
        t1 = []
        sign1 = 0
        sign2 = 0  # sign2=1 means first order relation
        sign3 = 0  # sign3=1 means second order relation

        start_index1 = 0
        end_index1 = min(start_index1 + self.batch_size, data_size)
        while start_index1 < data_size:
            if mod1 == 0:
                sign1 = 1.
                if start_index1 < data_size1 and end_index1 < data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    h1 = []
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(start_index1, end_index1):
                        cur_h = edges1[shuffle_indices1[i]][0]
                        cur_t = edges1[shuffle_indices1[i]][1]
                        h1.append(cur_h)
                        t1.append(cur_t)
                        wf.append(self.weight_intra('af1',cur_h,cur_t))
                        ws.append(0)
                elif start_index1 < data_size1 and end_index1 >= data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    end_index1 = data_size1
                    h1 = []
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(start_index1, end_index1):
                        cur_h = edges1[shuffle_indices1[i]][0]
                        cur_t = edges1[shuffle_indices1[i]][1]
                        h1.append(cur_h)
                        t1.append(cur_t)
                        wf.append(self.weight_intra('af1',cur_h, cur_t))
                        ws.append(0)
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h1 = []
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(start_index1 - data_size1, end_index1 - data_size1):
                        cur_h = self.second_sn1[shuffle_indices2[i]][0]
                        cur_t = self.second_sn1[shuffle_indices2[i]][1]
                        h1.append(cur_h)
                        t1.append(cur_t)
                        wf.append(0)
                        ws.append(self.weight_intra('as1',cur_h, cur_t))

            else:
                sign1 = -1.
                if start_index1 < data_size1 and end_index1 < data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(len(h1)):
                        cur_t=self.sampling_table1[random.randint(0, table_size - 1)]
                        t1.append(cur_t)
                        wf.append(self.weight_intra('af1',h1[i], cur_t))
                        ws.append(0)
                elif start_index1 < data_size1 and end_index1 >= data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    end_index1 = data_size1
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(len(h1)):
                        cur_t=self.sampling_table1[random.randint(0, table_size - 1)]
                        t1.append(cur_t)
                        wf.append(self.weight_intra('af1',h1[i], cur_t))
                        ws.append(0)
                else:
                    sign2 = 0.
                    sign3 = 1.
                    t1 = []
                    wf = []
                    ws = []
                    for i in range(len(h1)):
                        cur_t=self.second_sampling_table1[random.randint(0, table_size - 1)]
                        t1.append(cur_t)
                        wf.append(0)
                        ws.append(self.weight_intra('as1',h1[i], cur_t))
            yield h1, t1, [sign1], [sign2], [sign3],wf,ws
            mod1 += 1
            mod1 %= mod_size
            if mod1 == 0:
                start_index1 = end_index1
                end_index1 = min(start_index1 + self.batch_size, data_size)

    def batch_iter2(self):
        look_up2 = self.g2.look_up_dict
        table_size = 1e8
        numNodes2 = self.node_size2

        edges2 = [(look_up2[x[0]], look_up2[x[1]]) for x in self.g2.G.edges()]
        data_size1 = self.g2.G.number_of_edges()
        data_size2 = len(self.second_sn2)
        data_size = data_size1 + data_size2
        shuffle_indices1 = np.random.permutation(np.arange(data_size1))
        shuffle_indices2 = np.random.permutation(np.arange(data_size2))
        sign1 = 0
        sign2 = 0  # sign2=1 means first order relation
        sign3 = 0  # sign3=1 means second order relation
        h2 = []
        t2 = []
        start_index2 = 0
        end_index2 = min(start_index2 + self.batch_size, data_size)
        mod1 = 0
        mod_size = 1 + self.negative_ratio
        while start_index2 < data_size:
            if mod1 == 0:
                sign1 = 1.
                if start_index2 < data_size1 and end_index2 < data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    h2 = []
                    t2 = []
                    wf=[]
                    ws=[]
                    for i in range(start_index2, end_index2):
                        if not random.random() < self.edge_prob2[shuffle_indices1[i]]:
                            shuffle_indices1[i] = self.edge_alias2[shuffle_indices1[i]]
                        cur_h = edges2[shuffle_indices1[i]][0]
                        cur_t = edges2[shuffle_indices1[i]][1]
                        h2.append(cur_h)
                        t2.append(cur_t)
                        wf.append(self.weight_intra('af2',cur_h, cur_t))
                        ws.append(0)
                elif start_index2 < data_size1 and end_index2 >= data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    end_index2 = data_size1
                    h2 = []
                    t2 = []
                    wf = []
                    ws = []
                    for i in range(start_index2, end_index2):
                        if not random.random() < self.edge_prob2[shuffle_indices1[i]]:
                            shuffle_indices1[i] = self.edge_alias2[shuffle_indices1[i]]
                        cur_h = edges2[shuffle_indices1[i]][0]
                        cur_t = edges2[shuffle_indices1[i]][1]
                        h2.append(cur_h)
                        t2.append(cur_t)
                        wf.append(self.weight_intra('af2',cur_h, cur_t))
                        ws.append(0)
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h2 = []
                    t2 = []
                    wf = []
                    ws = []
                    for i in range(start_index2 - data_size1, end_index2 - data_size1):
                        cur_h = self.second_sn2[shuffle_indices2[i]][0]
                        cur_t = self.second_sn2[shuffle_indices2[i]][1]
                        h2.append(cur_h)
                        t2.append(cur_t)
                        wf.append(0)
                        ws.append(self.weight_intra('as2',cur_h, cur_t))
            else:
                sign1 = -1.
                if start_index2 < data_size1 and end_index2 < data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    t2 = []
                    wf = []
                    ws = []
                    for i in range(len(h2)):
                        cur_t=self.sampling_table2[random.randint(0, table_size - 1)]
                        t2.append(cur_t)
                        wf.append(self.weight_intra('af2',h2[i], cur_t))
                        ws.append(0)
                elif start_index2 < data_size1 and end_index2 >= data_size1:
                    sign2 = 1.
                    sign3 = 0.
                    end_index2 = data_size1
                    t2 = []
                    wf = []
                    ws = []
                    for i in range(len(h2)):
                        cur_t=self.sampling_table2[random.randint(0, table_size - 1)]
                        t2.append(cur_t)
                        wf.append(self.weight_intra('af2',h2[i], cur_t))
                        ws.append(0)
                else:
                    sign2 = 0.
                    sign3 = 1.
                    t2 = []
                    wf = []
                    ws = []
                    for i in range(len(h2)):
                        cur_t=self.second_sampling_table2[random.randint(0, table_size - 1)]
                        t2.append(cur_t)
                        wf.append(0)
                        ws.append(self.weight_intra('as2',h2[i], cur_t))

            yield h2, t2, [sign1], [sign2], [sign3],wf,ws
            mod1 += 1
            mod1 %= mod_size
            if mod1 == 0:
                start_index2 = end_index2
                end_index2 = min(start_index2 + self.batch_size, data_size)


    def batch_iter_anpf(self):
        data_size_anpf = len(self.an_pf1) + len(self.an_pf2)
        new_list_p = self.an_pf1 + self.an_pf2
        len_anpf1 = len(self.an_pf1)
        start_index = 0
        end_index = min(start_index + self.an_batch_size, data_size_anpf)
        shuffle_indices = np.random.permutation(np.arange(len(new_list_p)))
        table_size = 1e8
        mod1 = 0
        mod_size = 1 + self.negative_ratio
        h_a = []
        t_a = []
        while start_index < data_size_anpf:
            if mod1 == 0:
                sign1 = 1.
                if end_index < len_anpf1:
                    sign2 = 1.
                    sign3 = 0.
                    h_a = []
                    t_a = []
                    w=[]
                    for i in range(start_index, end_index):
                        cur_h = new_list_p[shuffle_indices[i]][0]
                        cur_t = new_list_p[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',cur_h, cur_t))
                elif start_index < len_anpf1 and end_index >= len_anpf1:
                    end_index = len_anpf1
                    sign2 = 1.
                    sign3 = 0.
                    h_a = []
                    t_a = []
                    w = []
                    for i in range(start_index, end_index):
                        cur_h = new_list_p[shuffle_indices[i]][0]
                        cur_t = new_list_p[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',cur_h, cur_t))
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h_a = []
                    t_a = []
                    w=[]
                    for i in range(start_index, end_index):
                        cur_h = new_list_p[shuffle_indices[i]][0]
                        cur_t = new_list_p[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af21',cur_h, cur_t))
            else:
                sign1 = -1.
                if end_index < len_anpf1:
                    sign2 = 1.
                    sign3 = 0.
                    t_a = []
                    w = []
                    for i in range(len(h_a)):
                        cur_t = self.sampling_table2[random.randint(0, table_size - 1)]
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',h_a[i], cur_t))
                elif start_index < len_anpf1 and end_index >= len_anpf1:
                    # end_index=len(self.an_pf1)
                    sign2 = 1.
                    sign3 = 0.
                    t_a = []
                    w = []
                    for i in range(len(h_a)):
                        cur_t = self.sampling_table2[random.randint(0, table_size - 1)]
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',h_a[i], cur_t))
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h_a = []
                    w = []
                    for i in range(len(t_a)):
                        cur_h = self.sampling_table1[random.randint(0, table_size - 1)]
                        h_a.append(cur_h)
                        w.append(self.weight_intra('af21',cur_h, t_a[i]))
            yield h_a, t_a, [sign1], [sign2], [sign3],w
            mod1 += 1
            mod1 %= mod_size
            if mod1 == 0:
                start_index = end_index
                end_index = min(start_index + self.an_batch_size, data_size_anpf)

    def batch_iter_ancf(self):
        data_size_ancf = len(self.an_cf1) + len(self.an_cf2)
        new_list_c = self.an_cf1 + self.an_cf2
        len_ancf1 = len(self.an_cf1)
        start_index = 0
        end_index = min(start_index + self.an_batch_size, data_size_ancf)
        shuffle_indices = np.random.permutation(np.arange(len(new_list_c)))
        table_size = 1e8
        mod1 = 0
        mod_size = 1 + self.negative_ratio
        h_a = []
        t_a = []
        while start_index < data_size_ancf:
            if mod1 == 0:
                sign1 = 1.
                if end_index < len_ancf1:
                    sign2 = 1.
                    sign3 = 0.
                    h_a = []
                    t_a = []
                    w=[]
                    for i in range(start_index, end_index):
                        cur_h = new_list_c[shuffle_indices[i]][0]
                        cur_t = new_list_c[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',cur_h,cur_t))
                elif start_index < len_ancf1 and end_index >= len_ancf1:
                    end_index = len_ancf1
                    sign2 = 1.
                    sign3 = 0.
                    h_a = []
                    t_a = []
                    w=[]
                    for i in range(start_index, end_index):
                        cur_h = new_list_c[shuffle_indices[i]][0]
                        cur_t = new_list_c[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',cur_h, cur_t))
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h_a = []
                    t_a = []
                    w=[]
                    for i in range(start_index, end_index):
                        cur_h = new_list_c[shuffle_indices[i]][0]
                        cur_t = new_list_c[shuffle_indices[i]][1]
                        h_a.append(cur_h)
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af21',cur_h, cur_t))
            else:
                sign1 = -1.
                if end_index < len_ancf1:
                    sign2 = 1.
                    sign3 = 0.
                    t_a = []
                    w=[]
                    for i in range(len(h_a)):
                        cur_t = self.sampling_table2[random.randint(0, table_size - 1)]
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',h_a[i],cur_t))
                elif start_index < len_ancf1 and end_index >= len_ancf1:
                    # end_index = len(self.an_cf1)
                    sign2 = 1.
                    sign3 = 0.
                    t_a = []
                    w=[]
                    for i in range(len(h_a)):
                        cur_t = self.sampling_table2[random.randint(0, table_size - 1)]
                        t_a.append(cur_t)
                        w.append(self.weight_intra('af12',h_a[i],cur_t))
                else:
                    sign2 = 0.
                    sign3 = 1.
                    h_a = []
                    w=[]
                    for i in range(len(t_a)):
                        cur_h = self.sampling_table1[random.randint(0, table_size - 1)]
                        h_a.append(cur_h)
                        w.append(self.weight_intra('af21',cur_h,t_a[i]))
            yield h_a, t_a, [sign1], [sign2], [sign3],w
            mod1 += 1
            mod1 %= mod_size
            if mod1 == 0:
                start_index = end_index
                end_index = min(start_index + self.an_batch_size, data_size_ancf)

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes1 = self.node_size1
        numNodes2 = self.node_size2

        print("Pre-procesing for non-uniform negative sampling in SN1!")
        node_degree1 = np.zeros(numNodes1)  # out degree

        look_up1 = self.g1.look_up_dict
        for edge in self.g1.G.edges():
            node_degree1[look_up1[edge[0]]
            ] += self.g1.G[edge[0]][edge[1]]["weight"]

        norm1 = sum([math.pow(node_degree1[i], power) for i in range(numNodes1)])

        self.sampling_table1 = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes1):
            p += float(math.pow(node_degree1[j], power)) / norm1
            while i < table_size and float(i) / table_size < p:
                self.sampling_table1[i] = j
                i += 1

        data_size1 = self.g1.G.number_of_edges()
        self.edge_alias1 = np.zeros(data_size1, dtype=np.int32)
        self.edge_prob1 = np.zeros(data_size1, dtype=np.float32)
        large_block1 = np.zeros(data_size1, dtype=np.int32)
        small_block1 = np.zeros(data_size1, dtype=np.int32)

        total_sum1 = sum([self.g1.G[edge[0]][edge[1]]["weight"]
                          for edge in self.g1.G.edges()])
        norm_prob1 = [self.g1.G[edge[0]][edge[1]]["weight"] *
                      data_size1 / total_sum1 for edge in self.g1.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size1 - 1, -1, -1):
            if norm_prob1[k] < 1:
                small_block1[num_small_block] = k
                num_small_block += 1
            else:
                large_block1[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block1[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block1[num_large_block]
            self.edge_prob1[cur_small_block] = norm_prob1[cur_small_block]
            self.edge_alias1[cur_small_block] = cur_large_block
            norm_prob1[cur_large_block] = norm_prob1[cur_large_block] + \
                                          norm_prob1[cur_small_block] - 1
            if norm_prob1[cur_large_block] < 1:
                small_block1[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block1[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob1[large_block1[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob1[small_block1[num_small_block]] = 1

        print("Pre-procesing for non-uniform negative sampling in SN2!")
        node_degree2 = np.zeros(numNodes2)
        look_up2 = self.g2.look_up_dict
        for edge in self.g2.G.edges():
            node_degree2[look_up2[edge[0]]
            ] += self.g2.G[edge[0]][edge[1]]["weight"]

        norm2 = sum([math.pow(node_degree2[i], power) for i in range(numNodes2)])

        self.sampling_table2 = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes2):
            p += float(math.pow(node_degree2[j], power)) / norm2
            while i < table_size and float(i) / table_size < p:
                self.sampling_table2[i] = j
                i += 1

        data_size2 = self.g2.G.number_of_edges()
        self.edge_alias2 = np.zeros(data_size2, dtype=np.int32)
        self.edge_prob2 = np.zeros(data_size2, dtype=np.float32)
        large_block2 = np.zeros(data_size2, dtype=np.int32)
        small_block2 = np.zeros(data_size2, dtype=np.int32)

        total_sum2 = sum([self.g2.G[edge[0]][edge[1]]["weight"]
                          for edge in self.g2.G.edges()])
        norm_prob2 = [self.g2.G[edge[0]][edge[1]]["weight"] *
                      data_size2 / total_sum2 for edge in self.g2.G.edges()]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size2 - 1, -1, -1):
            if norm_prob2[k] < 1:
                small_block2[num_small_block] = k
                num_small_block += 1
            else:
                large_block2[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block2[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block2[num_large_block]
            self.edge_prob2[cur_small_block] = norm_prob2[cur_small_block]
            self.edge_alias2[cur_small_block] = cur_large_block
            norm_prob2[cur_large_block] = norm_prob2[cur_large_block] + \
                                          norm_prob2[cur_small_block] - 1
            if norm_prob2[cur_large_block] < 1:
                small_block2[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block2[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob2[large_block2[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob2[small_block2[num_small_block]] = 1

    def get_embedding(self):
        # return node with their corresponding embedding vectors and context vectors
        emb_vec1={}
        emb_parent1={}
        emb_child1={}
        emb_vec2={}
        emb_parent2={}
        emb_child2={}
        emb1=self.sn1_emb.eval(session=self.sess)
        emb1_p=self.sn1_parent.eval(session=self.sess)
        emb1_c=self.sn1_child.eval(session=self.sess)
        emb2=self.sn2_emb.eval(session=self.sess)
        emb2_p=self.sn2_parent.eval(session=self.sess)
        emb2_c=self.sn2_child.eval(session=self.sess)
        look_back1 = self.g1.look_back_list
        look_back2 = self.g2.look_back_list
        for i, emb in enumerate(emb1):
            emb_vec1[look_back1[i]]=emb
        for i,emb in enumerate(emb1_p):
            emb_parent1[look_back1[i]]=emb
        for i,emb in enumerate(emb1_c):
            emb_child1[look_back1[i]]=emb
        for i,emb in enumerate(emb2):
            emb_vec2[look_back2[i]]=emb
        for i,emb in enumerate(emb2_p):
            emb_parent2[look_back2[i]]=emb
        for i,emb in enumerate(emb2_c):
            emb_child2[look_back2[i]]=emb
        return emb_vec1,emb_parent1,emb_child1,emb_vec2,emb_parent2,emb_child2

    def get_embedding_direct(self):
        # return embedding vectors and context vectors
        emb1=self.sn1_emb.eval(session=self.sess)
        emb1_p=self.sn1_parent.eval(session=self.sess)
        emb1_c=self.sn1_child.eval(session=self.sess)
        emb2=self.sn2_emb.eval(session=self.sess)
        emb2_p=self.sn2_parent.eval(session=self.sess)
        emb2_c=self.sn2_child.eval(session=self.sess)

        sn1_emb = np.concatenate((emb1,emb1_p,emb1_c),axis=1)
        sn2_emb = np.concatenate((emb2,emb2_p,emb2_c),axis=1)

        return emb1, emb2, sn1_emb,sn2_emb

class TANE(object):
    def __init__(self, graph1, graph2, anchor, t, rep_dim=100, batch_size=10000,
                 an_batch_size=10000, negative_ratio=5, th=2):
        ops.reset_default_graph()

        self.emb_vec1={}
        self.emb_parent1 = {}
        self.emb_child1 = {}
        self.emb_vec2 = {}
        self.emb_parent2 = {}
        self.emb_child2 = {}

        out_iter = 0
        self.model = _TANE(graph1, graph2, anchor, out_iter, rep_dim, batch_size=batch_size,
                           an_batch_size=an_batch_size, negative_ratio=negative_ratio,t=t)

        while out_iter<th:
            for j in range(800):
                loss_emb = self.model.train_one_epoch_embedding()

            if out_iter < th - 1:
                for k in range(50):
                    loss_att = self.model.train_one_epoch_attention()
                self.model.embedding()
            out_iter = out_iter + 1
        print("Out epoch is ",out_iter,"batch size is ",batch_size)
        print("embedding epoch=",self.model.cur_epoch_sn1)
        print("embedding loss=", loss_emb)
        print("attention epoch=",self.model.cur_epoch_att)
        print("attention loss=", loss_att)

    def get_embedding(self):
        self.emb_vec1, self.emb_parent1, self.emb_child1, self.emb_vec2,\
        self.emb_parent2, self.emb_child2 = self.model.get_embedding()

    def get_embedding_directed(self):
        self.emb1, self.emb2, self.sn1_emb, self.sn2_emb = self.model.get_embedding_direct()

    def save_embeddings(self, filename, vector):
        f = open(filename, 'w')
        for node, vec in vector.items():
            f.write("{} {}\n".format(node,
                                     ' '.join([str(x) for x in vec])))
        f.close()
