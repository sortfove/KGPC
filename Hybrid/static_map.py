

import json
import os.path
from PPR_sample_2 import pprSampler

data_type = "kamed"
graph_path_base = "../data"
save_base_path = "process_output"

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 1、对知识大kg里面的实体进行提�?
# 输入数据集名称（不需要一定输入）�?我们这里输出�?大知识图谱的实体集、大知识图谱中的所有关系、大知识图谱
def get_dkg(data_type="kamed"):
    dkg = {}
    entities_dkg = []
    rel_dkg = []
    if data_type == "kamed":
        with open(os.path.join(graph_path_base, f"{data_type}_graph.json"), 'r', encoding='utf-8') as file:
            dkg = json.load(file)
            for triple_ls in dkg["graph"]:
                entities_dkg.append(triple_ls[0])
                entities_dkg.append(triple_ls[-1])
                rel_dkg.append(triple_ls[1])
    elif data_type == "meddialog":
        with open(os.path.join(graph_path_base, f"{data_type}_graph.json"), 'r', encoding='utf-8') as file:
            dkg = json.load(file)
            for triple_ls in dkg["graph"]:
                entities_dkg.append(triple_ls[0])
                entities_dkg.append(triple_ls[-1])
                rel_dkg.append(triple_ls[1])
    elif data_type == "meddg":
        with open(os.path.join(graph_path_base, f"{data_type}_graph.json"), 'r', encoding='utf-8') as file:
            dkg = json.load(file)
            for triple_ls in dkg["graph"]:
                entities_dkg.append(triple_ls[0])
                entities_dkg.append(triple_ls[-1])
                rel_dkg.append(triple_ls[1])

    return list(set(entities_dkg)), list(set(rel_dkg)), dkg  # 测得kamed数据集共3948个实�?

# 这里考虑到一件事就是输入文件的要是一个实体还是一个实体编号，如果是编号就需要做映射（决定了作映射）
# 将实体、关系进行映射（返回 实体map、关系map, dkg三元组元素对应的索引（列表）�?
def get_map(entities_dkg, rel_dkg, dkg, type_data="kamed"):
    map_dict = {}
    entities_map = {}
    rel_map = {}
    triples_index = {} # 图谱，每一个三元组都是由数字组�?
    base_path = os.path.join(save_base_path, type_data)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if os.path.exists(os.path.join(base_path, "map.json")) and os.path.exists(os.path.join(base_path, "triple_map.json")):
        with open(os.path.join(base_path, "map.json"), "r", encoding='utf-8') as file:
            map_dict = json.load(file)
        with open(os.path.join(base_path, "triple_map.json"), "r", encoding='utf-8') as file_t:
            triples_index = json.load(file_t)['graph']
        entities_map = map_dict['entities']
        rel_map = map_dict['relations']

    else:
        entities_map = {i: entities_dkg[i] for i in range(0, len(entities_dkg))}
        rel_map = {i: rel_dkg[i] for i in range(0, len(rel_dkg))}
        map_dict = {
            "entities": entities_map,
            "relations": rel_map
        }
        swapped_entities_map = {value: key for key, value in entities_map.items()}
        swapped_rel_map = {value: key for key, value in rel_map.items()}
        index_triple_ls = []
        for head, re, tail in dkg['graph']:
            index_triple_ls.append([swapped_entities_map[head], swapped_rel_map[re], swapped_entities_map[tail]])
        triples_index_dict = {
            "graph": index_triple_ls
        }
        with open(os.path.join(base_path, "map.json"), "w", encoding='utf-8') as file:
            json.dump(map_dict, file, ensure_ascii=False, indent=4)
        with open(os.path.join(base_path, "triple_map.json"), "w", encoding='utf-8') as file:
            json.dump(triples_index_dict, file, ensure_ascii=False, indent=4)
        triples_index = triples_index_dict['graph']

    return entities_map, rel_map, triples_index

# 这个函数可以调用，输入实体拿到对应的top-k个相关实体组成的子图
# 返回值为 子图、未链接到子图的实体（后面用虚线链接�? 这里输入的entities_index是一个从句子中提取出来的候选实体的编号（也就是将实体转化成编号�?
def get_static_map(entities_index, ppr_sample):
    sub_graph = []  # 用于存储多个候选实体提取的子知识图�?
    for ent in entities_index:
        ls = ppr_sample.getOneSubgraph(ent)
        sub_graph += ls
    # 我们这里还需要将子图的数字索引换成文�?暂时不用)
    return sub_graph

def get_static_map_string(entities_index, ppr_sample, entities_map, rels_map):
    sub_graph = []  # 用于存储多个候选实体提取的子知识图�?
    for ent in entities_index:
        ls = ppr_sample.getOneSubgraph(ent)
        sub_graph += ls

    # 我们这里还需要将子图的数字索引换成文�?
    sub_graph_string = []
    for Atriple in sub_graph:
        sub_graph_string.append([entities_map[Atriple[0]], rels_map[Atriple[1]], entities_map[Atriple[2]]])
    return sub_graph



def get_ppr_sample(n_samp_ent, n_samp_edge, data_type="kamed", save_base_path=""):
    entities_dkg, rel_dkg, dkg = get_dkg(data_type)
    entities_map, rel_map, triples_index = get_map(entities_dkg, rel_dkg, dkg, data_type)
    n_ent = len(entities_dkg)
    n_rel = len(rel_dkg)
    n_samp_ent = n_samp_ent
    n_samp_edge = n_samp_edge # 这里好像没用�?
    triple_ls = triples_index
    # base_save_path = os.path.join(save_base_path, data_type)
    check_path(save_base_path)
    ppr_sample = pprSampler(n_ent, n_rel, n_samp_ent, n_samp_edge, triple_ls, save_base_path)
    return ppr_sample, entities_map, rel_map


if __name__ == "__main__":
    entities_dkg, rel_dkg, dkg = get_dkg()  # 获取知识图谱的涉及到的实体集合、大知识图谱

    # 先将实体、关系作映射，映射成数字，实体关系都要做映射
    # entities_map, rel_map, triples_index  分布对应实体映射字典、关系映射字典、词转化成index的知识图谱（大kg的所有知识三元组�?
    entities_map, rel_map, triples_index = get_map(entities_dkg, rel_dkg, dkg)

    # 获取配置参数
    # loader.n_ent,  大kg实体数量
    # loader.n_rel,  关系数量
    # args.n_samp_ent,  提取的实体数量（通过PPR提取的top-k个）
    # args.n_samp_edge,  提取的边的数�?
    # base_save_path,
    n_ent = len(entities_dkg)
    n_rel = len(rel_dkg)
    n_samp_ent = 30
    n_samp_edge = 100
    triple_ls = triples_index
    base_save_path = os.path.join("process_output", data_type)
    ppr_sample = pprSampler(n_ent, n_rel, n_samp_ent, n_samp_edge, triple_ls, base_save_path)
    head, topk_nodes, node_index = ppr_sample.getOneSubgraph(0)
    # 截至到这�?topk_nodes已经提取出来了，现在就是sampled_edges还是有问�?
    # 我们做一件事，知识三元组对topk_nodes进行判别，如果头尾实体同时出现于topk_nodes就提取下来，作为当前对话小知识图�?
    count = 0
    # 这种方式提取的数据只是包含在子图中的，但是我们忽视其它没有在子图中的结点，将这些结点存入�?
    # 后面做动态图的时候一起将这些结点作为虚边结点
    for triple in triples_index:
        if triple[0] in topk_nodes and triple[2] in topk_nodes:
            print(triple)
            count += 1
    print(count)
