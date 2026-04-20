# 下面代码部分是从静态图摘抄了，方便测试
# 1、先获取对话文本的实�?
# 2、找出这些实体的所有关系（设计prompt进行评分，找到top-d个）
# 3
import copy
import json
import os.path
from PPR_sample_2 import pprSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

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

    return list(set(entities_dkg)), list(set(rel_dkg)), dkg  # 测得kamed数据集共3948个实�?

# 这里考虑到一件事就是输入文件的要是一个实体还是一个实体编号，如果是编号就需要做映射（决定了作映射）
# 将实体、关系进行映射（返回 实体map、关系map, dkg三元组元素对应的索引（列表）�?
def get_map(entities_dkg, rel_dkg, dkg, type_data="kamed"):
    map_dict = {}
    entities_map = {}
    rel_map = {}
    triples_index = {} # 图谱，每一个三元组都是由数字组�?
    base_path = os.path.join(save_base_path, data_type)
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
    # 我们这里还需要将子图的数字索引换成文�?

    return sub_graph

def get_ppr_sample(n_samp_ent, n_samp_edge, data_type="kamed", save_base_path=""):
    entities_dkg, rel_dkg, dkg = get_dkg(data_type)
    entities_map, rel_map, triples_index = get_map(entities_dkg, rel_dkg, dkg)
    n_ent = len(entities_dkg)
    n_rel = len(rel_dkg)
    n_samp_ent = n_samp_ent
    n_samp_edge = n_samp_edge # 这里好像没用�?
    triple_ls = triples_index
    base_save_path = os.path.join(save_base_path, data_type)
    check_path(base_save_path)
    ppr_sample = pprSampler(n_ent, n_rel, n_samp_ent, n_samp_edge, triple_ls, base_save_path)
    return ppr_sample


# 找出当前实体的所有关�?单个实体)(不需�?
def extract_entity_rel(entity, mini_graph):
    rel = []
    for triple in mini_graph:
        if entity in triple:
            rel.append(triple[1])
    return list(set(rel))

def use_llm(model, tokenizer, prompt):
    predict = []
    chat = [
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    # 使用 generate 方法生成响应
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )  # 解码生成的文�?
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result


# 找出当前实体的所有三元组，并返回所有的三元�?单个实体)
def extract_candidate_triple(entity, mini_graph):
    candidate_triples = []
    for triple in mini_graph:
        if entity in triple:
            candidate_triples.append(triple)
    return candidate_triples


#
# # 根据当前对话评估关系(返回一个列表，元素�?-》{entity:..., rel: ..., score:...})
# def eval_rel(model, tokenizer, entity, entity_rel, question, entities_map, rel_map, width):
#     num_rel = len(entity_rel)
#     if num_rel <= width:
#         scores_ls = []
#         for rel in entity_rel:
#             scores_ls.append({
#                 'entity': entity,
#                 "relation": rel,
#                 "score": float(85)
#             })
#         return scores_ls
#
#     # prompt = (
#     #     f"你是一个评分师，并按照以下准则做评分。请检�?实体表中{num_entities}个实�?用分号分�?,这些实体有助于回答question中的问题,"
#     #     f"并在0�?00的量表上评估它们的贡�?这些分数尽量不要相等,且都不为0)。其中分数越靠近0,贡献度越�?分数越靠�?00,贡献度越高�?
#     #     f"按照实体表的实体顺序进行分数输出,仅输出分�?其它东西不要输出�?输出格式为score1,score2,......"
#     #     f"\n\n#question:{question}"
#     #     f"\n\n#实体�?{entities}")
#
#     rels = [rel_map[str(rel)] for rel in entity_rel]
#     prompt = (f"例子�?question:我的头有点晕，带有流鼻涕的症状！#关系�?['symptom_of','check_item','to_treat']\n输出:"
#               f"symptom_of:80\ncheck_item:90\nto_treat:76\n"
#               f"你是一个评分师，你需要学习上面的处理模式并按照以下规则做评分，但是要注意不要让上面的例子影响你的评分。请检�?关系表中这{num_rel}个关系（用分号分隔），这些关系有助于回答question中的问题�?
#               f"并在0�?00的量表上评估它们的贡献（这些分数尽量不要相等）�?
#               f"其中分数越靠�?，贡献度越低，分数越靠近100，贡献度越高,注意分数不能�?�?00。按多行输出�?
#               f"每行输出对应关系和分数，输出内容中只能包含对应关系和对应分数，不要出现任何其它东西，输出格式为对应关�?分数"
#               f"\n\n#question:{question}"
#               f"\n\n#关系�?{rels}")
#
#
#     while True:
#         try:
#             result = use_llm(model, tokenizer, prompt)
#             # 处理这些数据.
#             scores_ls = []
#             swapped_dict = {value: key for key, value in rel_map.items()}
#             for line in result.split('\n'):
#                 rel, score = line.split(':')
#                 scores_ls.append({
#                     'entity': entity,
#                     "relation": swapped_dict[rel],
#                     "score": float(score)
#                 })
#             if len(scores_ls) != num_rel:
#                 continue
#             break
#         except:
#             pass
#     return scores_ls


# 根据当前对话评估关系(返回一个列表，元素�?-》{entity:..., rel: ..., score:...})
def eval_rel(model, tokenizer, entity, entity_rel, question, entities_map, rel_map, width):
    num_rel = len(entity_rel)
    if num_rel <= width:
        scores_ls = []
        for rel in entity_rel:
            scores_ls.append({
                'entity': entity,
                "relation": rel,
                "score": float(85)
            })
        return scores_ls

    # prompt = (
    #     f"你是一个评分师，并按照以下准则做评分。请检�?实体表中{num_entities}个实�?用分号分�?,这些实体有助于回答question中的问题,"
    #     f"并在0�?00的量表上评估它们的贡�?这些分数尽量不要相等,且都不为0)。其中分数越靠近0,贡献度越�?分数越靠�?00,贡献度越高�?
    #     f"按照实体表的实体顺序进行分数输出,仅输出分�?其它东西不要输出�?输出格式为score1,score2,......"
    #     f"\n\n#question:{question}"
    #     f"\n\n#实体�?{entities}")

    rels = [rel_map[str(rel)] for rel in entity_rel]
    prompt = (f"你是一个评分师，并按照以下准则做评分。请检�?关系表中这{num_rel}个关系（用分号分隔），这些关系有助于回答question中的问题�?
              f"并在0�?00的量表上评估它们的贡献（这些分数尽量不要相等）�?
              f"其中分数越靠�?，贡献度越低；分数越靠近100，贡献度越高,注意分数不能�?�?00�?
              f"按照关系表的关系顺序进行贡献度分数输�?每一个分数要对应好实体的顺序，仅输出分数，其它东西不要输出，,输出格式为score1,score2,......"
              f"\n\n#question:{question}"
              f"\n\n#关系�?{rels}")

    flag = 0
    while True:
        try:
            result = use_llm(model, tokenizer, prompt)
            # 处理这些数据.
            scores_ls = []
            swapped_dict = {value: key for key, value in rel_map.items()}
            for rel, score in zip(rels, result.split(',')):
                scores_ls.append({
                    'entity': entity,
                    "relation": swapped_dict[rel],
                    "score": float(score)
                })
            if len(scores_ls) != num_rel:
                continue
            break
        except:
            flag += 1
            if flag == 8:
                scores_ls = []
                for rel in entity_rel:
                    scores_ls.append({
                        'entity': entity,
                        "relation": rel,
                        "score": float(85)
                    })
                return scores_ls
            # if flag ==


    return scores_ls



# 获取width个最相关的关系rel_scores = {entity:..., rel: ..., score:...}
def get_width_rel(rel_scores, width):
    sorted_scores = sorted(rel_scores, key=lambda x: x['score'], reverse=True)
    return sorted_scores[:width]


# 获取候选实�?
def extract_candidate_entities(target_rel, graph, max_num_candidates):
    candidate_entities = []
    for item in target_rel:
        entity = item['entity']
        rel = item['relation']
        # 找出该实体对应关系的所有三元组
        for triple in graph:
            if (entity == triple[0]) and (int(rel) == triple[1]):
                candidate_entities.append(triple[2])
            elif (entity == triple[2]) and (int(rel) == triple[1]):
                candidate_entities.append(triple[0])
    candidate_entities = list(set(candidate_entities))
    if len(candidate_entities) > max_num_candidates:
        indices = random.sample(range(len(candidate_entities)), max_num_candidates)
        candidate_entities = [candidate_entities[i] for i in indices]
    return candidate_entities

# 对候选词语进行评�?返回{candidate:...,score:...})
def eval_candidate_entities(model, tokenizer, candidate_entities, entity_rel, question, entities_map, rel_map, width, dynamic_entities=None):
    num_entities = len(candidate_entities)
    if num_entities <= width:
        scores_ls = []
        for entity in candidate_entities:
            scores_ls.append({
                "candidate": entity,
                "score": float(80)
            })
        return scores_ls
    entities = [entities_map[str(ent)] for ent in candidate_entities]
    #
    if dynamic_entities:
        entities += dynamic_entities
        num_entities += len(dynamic_entities)

    prompt = (f"你是一个评分师，并按照以下准则做评分。请检�?实体表中{num_entities}个实�?用分号分�?,这些实体有助于回答question中的问题,"
              f"并在0�?00的量表上评估它们的贡�?这些分数尽量不要相等,且都不为0)。其中分数越靠近0,贡献度越�?分数越靠�?00,贡献度越高�?
              f"按照实体表的实体顺序进行分数输出,仅输出分�?其它东西不要输出�?输出格式为score1,score2,......"
              f"\n\n#question:{question}"
              f"\n\n#实体�?{entities}")

    # if dynamic_entities:

    flag = 0
    while True:
        try:
            result = use_llm(model, tokenizer, prompt)
            scores_ls = []
            if dynamic_entities:
                for entity, score in zip(candidate_entities+dynamic_entities, result.split(',')):
                    scores_ls.append({
                        "candidate": entity,
                        "score": float(score)
                    })
            else:
                for entity, score in zip(candidate_entities, result.split(',')):
                    scores_ls.append({
                        "candidate": entity,
                        "score": float(score)
                    })
            if len(scores_ls) != num_entities:
                flag += 1
                if flag >= 8:
                    for entity in candidate_entities:
                        scores_ls.append({
                            "candidate": entity,
                            "score": float(80)
                        })
                    return scores_ls
                continue
            break
        except:
            flag += 1
            if flag >= 8:
                scores_ls = []
                for entity in candidate_entities:
                    scores_ls.append({
                        "candidate": entity,
                        "score": float(80)
                    })
                return scores_ls
    return scores_ls

# 获取width个候选实�?
def get_width_candidates(candidate_scores, width, dynamic_entities=None):
    end_width = width
    sorted_scores = sorted(candidate_scores, key=lambda x: x['score'], reverse=True)
    if dynamic_entities:
        dyn_entities = []
        can_entities = []
        for item in sorted_scores[:width]:
            if item['candidate'] in dynamic_entities:
                end_width += 1
                dyn_entities.append(item['candidate'])
            else:
                can_entities.append({
                    "candidate": item['candidate'],
                    "score": item['score']
                })
        return dyn_entities, can_entities
    else:
        return sorted_scores[:end_width]

# 用于判断当前深度的知识足够回答大模型�?
def judge_enough_knowledge(model, tokenizer, total_triple, question, entities_map, rel_map):
    tog_triples = [[entities_map[str(tri[0])], rel_map[str(tri[1])], entities_map[str(tri[2])]] for tri in total_triple]
    while True:
        try:
            prompt = (f"你是一个专业的判断员，你的任务是去判断当前�?knowledge中的知识是否能够支撑模型"
                      f"�?question的问题进行正确回答，仅仅在['yes', 'no']中选择回答"
                      f"如果能就输出yes,不能则输出no,不要输出其它任何东西"
                      f"\n\n#question:{question}"
                      f"\n\n#knowledge:{tog_triples}")
            result = use_llm(model, tokenizer, prompt)
            # if result not in ['yes', 'no']:
            #     continue
            if 'yes' in result:
                return 'yes'
            elif 'no' in result:
                return 'no'
            continue
        except:
            continue


# 这个函数用于判断query是否对医疗诊断过程有意义，如果没有就不会进行tog，例如出现对话为“嗯嗯”，“希望你能理解�?
def judge_query(model, tokenizer, query):
    flag = 0
    while True:
        try:
            if flag >= 10:
                return 'no'
            prompt = (f"你是一个专业的判断员，你需要做的判�?query中的内容是属于医疗对话内容，而不是毫无意义的文本(比如：谢谢、幸苦你了等�?,"
                      f"仅仅在['yes', 'no']中选择回答"
                      f"#query: {query}")
            result = use_llm(model, tokenizer, prompt)
            if 'yes' in result:
                return 'yes'
            elif 'no' in result:
                return 'no'
            flag += 1
            continue
        except:
            flag += 1
            continue




# 获取知识链路（输入，所有的实体）参数：所有相关实体、question, 大kg
def extract_kg_links(model, tokenizer, entities, question, mini_graph, entities_map, rel_map, depth, width, dynamic_entities):
    # depth = 1
    judge_ues = judge_query(model, tokenizer, question)

    # 我们将对应动态图的遍历实体的时候出现过的给删掉(因为这里dynamic是用文字，entities是用对应的数字表�?
    # dynamic_entities = [entity for entity in dynamic_entities if entity not in entities]
    for ent in entities:
        if entities_map[str(ent)] in dynamic_entities:
            dynamic_entities.remove(entities_map[str(ent)])



    if judge_ues == 'no':
        return []
    total_triple = []  # 所有要用到的三元组
    dynamic_triple = []   # 动态图推理出的三元�?
    total_entities = [entities]  # 搜索过程每一轮遍历到的实体（初始化是从文本中提取的实体）
    topic_entities = entities
    history_topic = entities  # []
    for dp in range(1, depth+1):
        new_entities = []  # 本轮搜索到的实体
        new_triple = []   # 本次搜索到的三元�?
        # 经过测试可以知道，由于是无向图，第二轮深度有可能是遍历到上一轮遍历到的实体，
        # 此时不是很有意义，因此要加上历史出现的实体进行判断，让出现过的实体不再进行tog
        for entity in topic_entities:
            entity_rel = extract_entity_rel(entity, mini_graph)
            if len(entity_rel) == 0:
                continue

            rel_scores = eval_rel(model, tokenizer, entity, entity_rel, question, entities_map, rel_map, width)
            # 找出top-width个相关关�?

            target_rel = get_width_rel(rel_scores, width)
            candidate_entities = extract_candidate_entities(target_rel, mini_graph, max_num_candidates=5)

            target_entity = []
            dyn_entities = []
            if dp == 1 and len(dynamic_entities) != 0:
                candidate_scores = eval_candidate_entities(model, tokenizer, candidate_entities, entity_rel, question, entities_map, rel_map,
                                    width, dynamic_entities)
                dyn_entities, target_entity = get_width_candidates(candidate_scores, width, dynamic_entities)
                if len(dyn_entities) != 0:
                    for dy in dyn_entities:
                        dynamic_triple.append([entities_map[str(entity)], 'next', dy])

            else:
                candidate_scores = eval_candidate_entities(model, tokenizer, candidate_entities, entity_rel, question, entities_map, rel_map, width)
                target_entity = get_width_candidates(candidate_scores, width)


            # new_entities = [e['candidate'] for e in target_entity]
            new_entities += [e['candidate'] for e in target_entity if e['candidate'] not in new_entities]
        # new_entities = list(set(new_entities))
        # 找到对应的三元组
        # 为了不出现无向图的情况，这里要经过判�?
        for entity in topic_entities:
            for candidate in new_entities:
                for triple in mini_graph:
                    if (triple[0] == entity and triple[2] == candidate) or (triple[2] == entity and triple[0] == candidate):
                        total_triple.append(triple)
                        break
        # if len(new_entities) == 0:
        new_entities = [item for item in new_entities if item not in history_topic]
        history_topic += new_entities
        topic_entities = copy.deepcopy(new_entities)
        # judge_result = judge_enough_knowledge(model, tokenizer, total_triple, question, entities_map, rel_map)
        # if judge_result == "yes":
        #     break

    # 返回顺便把对应实体和关系映射改回文本形式
    tog_triples = [[entities_map[str(tri[0])], rel_map[str(tri[1])], entities_map[str(tri[2])]] for tri in total_triple]
    tog_triples += dynamic_triple
    # for triple in dynamic_triple:
    #     if

    # 最后再对三元组进行去重
    end_tog_triple = []
    for triple in tog_triples:
        for new_tri in end_tog_triple:
            if (triple[0] == new_tri[0] and triple[1] == new_tri[1] and triple[2] == new_tri[2]) or (triple[1] == "next" and triple[2] == new_tri[2]):
                continue
            end_tog_triple.extend(triple)
    return tog_triples



if __name__ == "__main__":
    model_path = "xxx"


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.pad_token_id
    )
    model = model.to("cuda:0")


    entities_dkg, rel_dkg, dkg = get_dkg()  # 获取知识图谱的涉及到的实体集合、大知识图谱

    # 先将实体、关系作映射，映射成数字，实体关系都要做映射
    # entities_map, rel_map, triples_index  分布对应实体映射字典、关系映射字典、词转化成index的知识图谱（大kg的所有知识三元组�?
    entities_map, rel_map, triples_index = get_map(entities_dkg, rel_dkg, dkg)

    n_ent = len(entities_dkg)
    n_rel = len(rel_dkg)
    n_samp_ent = 50
    n_samp_edge = 100
    triple_ls = triples_index
    base_save_path = os.path.join("process_output", data_type)
    ppr_sample = pprSampler(n_ent, n_rel, n_samp_ent, n_samp_edge, triple_ls, base_save_path)
    # test
    entities = [1335, 3898]
    # for i in entities:
    sub_graph = get_static_map(entities, ppr_sample)

    input_text = "干扰素和解霉素治疗病毒感染吗（，）咨询人性别：女咨询人年龄：1-7�?
    tog_triples = extract_kg_links(model, tokenizer, entities, input_text, sub_graph, entities_map, rel_map, 3, 3)
    print("输入的句子：")
    print(input_text)
    print("tog找到的对应的知识路径�?)
    for item in tog_triples:
        print(item)


