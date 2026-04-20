import networkx as nx
import pickle as pkl
import time
import copy
import numpy as np
import torch
import os
import logging
import copy
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict


def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


class pprSampler():
    def __init__(self, n_ent: int, n_rel: int, topk: int, topm: int,  triple_ls: list, base_save_path: str):
        '''
            args:
            topk: number of sampled nodes for one head entity
            triple_ls: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        '''
        print('==> initializing ppr sampler...')
        # self.args = args
        self.add_manual_edges = True
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.topk = topk
        self.topm = topm
        self.triple_ls = triple_ls
        self.data_folder = base_save_path
        self.ppr_savePath = os.path.join(self.data_folder, f'ppr_scores/')
        checkPath(self.ppr_savePath)

        # build head to edges with sparse matrix
        heads, edges = [h for (h, r, t) in triple_ls], list(range(len(triple_ls)))

        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(triple_ls)))

        # change data type
        self.triple_ls = torch.LongTensor(self.triple_ls)


        # build sparse tensor self.PPR_W for matrix-computation PPR
        tmp_degree, tmp_adj = torch.zeros(self.n_ent, self.n_ent), torch.zeros(self.n_ent, self.n_ent)
        tmp_adj[self.triple_ls[:,0], self.triple_ls[:,2]] = 1   # 提取三元组里面的所有三元组映射到矩阵中
        tmp_degree = torch.diag(1 / torch.sum(tmp_adj, dim=1))
        self.PPR_W = torch.eye(self.n_ent) + torch.matmul(tmp_degree, tmp_adj)  # 单位矩阵 +

        # self.PPR_W = self.PPR_W.cuda()
        self.PPR_W = self.PPR_W.cuda()
        del tmp_adj; del tmp_degree

        print('==> checking ppr scores for each entity...')
        # 对每一个实体进行ppr分数的计�?
        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if os.path.exists(ent_ppr_savePath):
                pass
            else:
                # with default setting to generate ppr scores
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        print('finished.')

        print('==> finish sampler initilization.')

    def updateEdges(self, triple_ls):
        # co-operate with shuffle_train
        heads, edges = [h for (h, r, t) in triple_ls], list(range(len(triple_ls)))
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(triple_ls)))
        self.triple_ls = torch.LongTensor(triple_ls)

    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        return scores

    def generatePPRScoresForOneEntity(self, h):
        alpha, iteration = 0.85, 100
        scores = torch.zeros(1, self.n_ent).cuda()
        s = torch.zeros(1, self.n_ent).cuda()
        s[0, h] = 1
        for i in range(iteration):
            scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)
        scores = scores.cpu().reshape(-1).numpy()
        return scores

    def sampleSubgraph(self, ent: int, cand=None):
        # sample subgraph to get the edges
        # ppr_scores = np.array(list(self.getPPRscores(ent).values()))
        ppr_scores = np.array(self.getPPRscores(ent))

        # gurantee the candidates are sampled
        if cand != None and self.topk < self.n_ent:
            tmp_ppr_scores = copy.deepcopy(ppr_scores)
            tmp_ppr_scores[cand] = 1e8
            topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk].tolist())))
        else:
            # topk sampling
            if self.topk < self.n_ent:
                topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
            else:
                # no sampling
                topk_nodes = list(range(self.n_ent))

        # get candididate edges
        selectd_edges = self.sparseTrainMatrix[topk_nodes, :]
        _, tmp_triple_ls = selectd_edges.nonzero()

        # (h,r,t)
        edges = self.triple_ls[tmp_triple_ls]
        topk_nodes = torch.LongTensor(topk_nodes)

        # edge sampling
        mask = torch.isin(edges[:, 2], topk_nodes)

        # [n_edges, 3]
        sampled_edges = edges[mask, :]

        # edge sampling (topm edges for each subgraph)
        edge_num = int(sampled_edges.shape[0])
        # NOTE: if self.topm== 0, then skip edge sampling
        if self.topm > 0 and edge_num > self.topm:
            # ppr weight
            heads, tails = sampled_edges[:, 0], sampled_edges[:, 2]
            edge_weights = ppr_scores[heads] + ppr_scores[tails]
            edge_weights = torch.Tensor(edge_weights)
            index = torch.topk(edge_weights, self.topm).indices
            sampled_edges = sampled_edges[index]

        # get node indexing map
        node_index = torch.zeros(self.n_ent).long()
        node_index[topk_nodes] = torch.arange(len(topk_nodes))

        # connect head to all tails
        # if self.args.add_manual_edges:
        if self.add_manual_edges:
            add_edges_head2tails = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_head2tails[:, 0] = ent
            add_edges_head2tails[:, 1] = 2 * self.n_rel + 1
            add_edges_head2tails[:, 2] = topk_nodes
            add_edges_tails2head = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_tails2head[:, 0] = topk_nodes
            add_edges_tails2head[:, 1] = 2 * self.n_rel + 2
            add_edges_tails2head[:, 2] = ent
            sampled_edges = torch.cat([sampled_edges, add_edges_head2tails, add_edges_tails2head], dim=0)
        return topk_nodes, node_index, sampled_edges

    def getOneSubgraph(self, head: int, cand=None):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, cand)
        count = 0
        sub_graph = []
        for triple in self.triple_ls:
            triple = triple.tolist()
            if triple[0] in topk_nodes and triple[2] in topk_nodes:
                sub_graph.append(triple)
                count += 1

        # return [head, topk_nodes, node_index]
        return sub_graph
