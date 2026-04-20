# 用于处理数据

import json
import os
from tqdm import tqdm
from static_map import get_ppr_sample, get_static_map
from Tog import extract_kg_links
import copy
from dynamic_map import get_dynamic_map
import re
hard_prompt_save_path = "process_output"
test_dynamic_map_load_path = "../dynamic_map/generation_{}/test_dynamic_result.json"
save_step = 200
extract_all = True
# 动态实体转化成对应的三元组形式
# def dynamic_triple(entities_static_index, dynamic_entities):

# main因为一开始数据字符串的拼接问题，后面我们在整个字符串的内容相关信息提取出�?
def extract_prompt(input_str: str, strip_static_kg: bool, strip_dynamic_kg: bool, using_origin: bool, model_type: str):
    knowledge = "可能相关的医疗事实：[null]"
    if using_origin:
        knowledge = ""
    else:
        if "可能与医生回复相关的医疗事实�? in input_str:
            begin_knowledge = input_str.index("可能与医生回复相关的医疗事实�?)
            end_knowledge = input_str[begin_knowledge:].index('�?) + begin_knowledge
            knowledge = input_str[begin_knowledge+len("可能与医生回复相关的医疗事实�?):end_knowledge]
            knowledge = "可能相关的医疗事实：" + knowledge + "�?
    instruction = "你是一个专业的医生，你正在与患者进行对话，请你根据后面提到的可能相关的医疗事实，对当前轮次患者的提问进行回复�?
    begin_history = input_str.index("医生和患者的对话历史�?)
    end_history = input_str[begin_history:].index(']') + begin_history

    history = input_str[begin_history:end_history]
    parts = re.split(r'patient：|doctor�?, history)[1:]
    if model_type == "llama":
        chat_str = ""
        for i in range(1, len(parts) + 1):
            if i % 2 == 1:
                name = "<|start_header_id|>user<|end_header_id|>"
            else:
                name = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            chat_str += name + parts[i - 1]
        chat_str += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        chat_str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + instruction + knowledge + '<|eot_id|>\n' + chat_str
        len_chat_str = len(chat_str)
        return chat_str, len_chat_str  # 两个返回值，因为llama生成的时候会附带原文，因此需要在此处截断

    elif model_type == "qwen1.5":
        chat_str = ""
        for i in range(1, len(parts) + 1):
            if i % 2 == 1:
                name = "<|im_start|>user\n"
            else:
                name = "<|im_start|>assistant\n"
            chat_str += name + parts[i - 1] + "<|im_end|>\n"
        chat_str += "<|im_start|>assistant"
        chat_str = "<|im_start|>system\n" + instruction + knowledge + '<|im_end|>\n' + chat_str
        len_chat_str = len(chat_str)
        return chat_str, len_chat_str  # 两个返回值，因为chatglm生成的时候会附带原文，因此需要在此处截断

    elif model_type == "deepseek_dist_llama":
        chat_str = ""
        for i in range(1, len(parts) + 1):
            if i % 2 == 1:
                if i == 1:
                    name = "xxx"
                else:
                    name = "xxx"
            else:
                name = "xxx"
            chat_str += name + parts[i - 1]
        chat_str += "xxx"
        chat_str = "xxx" + instruction + knowledge + chat_str
        len_chat_str = len(chat_str)
        return chat_str, len_chat_str  # 两个返回值，因为chatglm生成的时候会附带原文，因此需要在此处截断

# 关键实体选取
def search_entities_static_index(entity_impact_factor, top_num=5):
    if len(entity_impact_factor['first']) >= top_num or len(entity_impact_factor['second'].keys()) == 0:
        return entity_impact_factor['first']
    need_entities_length = top_num - len(entity_impact_factor['first'])
    sorted_items = sorted(entity_impact_factor['second'].items(), key=lambda item: item[1], reverse=True)
    ls = []
    val = None
    for key, value in sorted_items:
        if val or value != val:
            val = value
            if len(ls) >= need_entities_length:
                break
        if key not in entity_impact_factor['first']:
            ls.append(key)
    ls_end = []
    for item in list(reversed(entity_impact_factor['third']))[len(entity_impact_factor['first']):]:
        if len(ls_end) >= need_entities_length:
            break
        if item in ls and item not in entity_impact_factor['first']+ls_end:
            ls_end.append(item)

    return entity_impact_factor['first'] + ls_end


# 这个函数用于对话训练集进行对应格式化处理(base_path 基础的数据文件， 数据类型 data_type)
def process_hardprompt(
        model,
        tokenizer,
        base_data_path,
        ppr_sampler,
        entities_map,
        rel_map,
        data_type="kamed",
        depth=3,
        width=3,
        top_num=5,
        old_train=None,
        train_num=None,
        pre_save_path=None
):
    ls = []  # 每一个论文的对话字典
    entity_map_index = {value: key for key, value in entities_map.items()}  # 这个是用实体找到其对应的索引
    data_path = os.path.join(base_data_path, data_type)
    # old_train_filename 这边表示之前用于生成过的文件
    old_train_filename = []
    if old_train is not None:
        old_train_filename = []
        for mini_dic in old_train:
            old_train_filename.append(mini_dic['fileName'])
        old_train_filename = list(set(old_train_filename))
        ls = old_train
    enough_train = False
    save_count = 0
    for dt in ["train"]:
        nf = os.path.join(data_path, data_type + "_" + dt)
        # for target in tqdm(os.listdir(nf)[:50], desc="Building hardprompt", dynamic_ncols=True):
        # for target in tqdm(os.listdir(nf)[46:500], desc=f"Building hardprompt", dynamic_ncols=True):
        for target in tqdm(os.listdir(nf), desc=f"Building hardprompt", dynamic_ncols=True):
        # for target in os.listdir(nf):
            print(target)
            if old_train is not None:
                if target in old_train_filename:
                    continue

            # 直接存储到对应的索引即可
            entity_impact_factor = {
                "first": [],  # this turn appear entities
                "second": {},  # 字典，截至到目前为止所有实体的出现次数
                "third": []  # 截至到当前轮次的实体，越往后越新，越晚出现
            }  #
            history_dialogue = ""
            # dynamic_predict_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
            dynamic_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
            entities_list_eturn = []  # 二维列表，保留每一轮出现的实体
            end_file_path = os.path.join(nf, target)
            with open(end_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)["dialogues"]
            entities_static = []    # 该变量用于存储到当前轮次的所有实体，因为每一个turn进行知识推理都要使用所有出现过的实体（加入的时候要进行实体去重�?
            current_turn_entities = []   # 要保留当前轮次的实体，一轮包括patient和doctor
            for index in range(len(data)):
                # 每一轮的处理
                entity_impact_factor['first'] = [int(entity_map_index[item]) for item in data[index]["entity"] if item in entity_map_index.keys()]
                for i in entity_impact_factor['first']:
                    if i in entity_impact_factor['second'].keys():
                        entity_impact_factor['second'][i] += 1
                    else:
                        entity_impact_factor['second'][i] = 1
                entity_impact_factor['third'] += entity_impact_factor['first']
                ents_len = copy.deepcopy(len(entities_static))  # 对比一下实体是否有增加，没有就用原来预测出来的动态实�?
                entities_static += [item for item in data[index]["entity"] if item not in entities_static]
                # entities_static_index = [int(entity_map_index[item]) for item in entities_static]    # 将提取的实体映射成对应的index

                # 对待选实体进行筛�?取top-num个实�?
                entities_static_index = search_entities_static_index(entity_impact_factor, top_num)
                if data[index]["role"] == "patient":
                    history_dialogue += data[index]["role"] + "�? + data[index]["sentence"]
                    if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                        history_dialogue += '�?
                    current_turn_entities = data[index]["entity"]
                    question = data[index]["sentence"]
                    static_map = get_static_map(entities_static_index, ppr_sampler)
                    kg_triple = extract_kg_links(model, tokenizer, entities_static_index, question, static_map, entities_map, rel_map, depth, width, dynamic_entities)
                    if len(kg_triple) > 0:
                        kg_triple_str = ""
                        for en1, re, en2 in kg_triple:
                            kg_triple_str += f"({en1},{re},{en2}),"
                        input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[{kg_triple_str[:-1]}]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                    else:
                        # input_prompt = f"医生和患者的历史对话：[{history_dialogue}]；假如你是一个医生，请你根据前面提及到的历史对话、医疗事实进行诊断回复�?
                        input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[null]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                    output_prompt = data[index+1]['sentence']
                    ls.append({
                        "fileName": target,
                        "input": input_prompt,
                        "output": output_prompt,
                        "history": history_dialogue   # 这个history_dialogue是包含本轮的提问�?
                    })
                    save_count += 1
                    print(f"now to generate:{len(ls)}/{train_num}")
                    # 每save_step步保存一�?
                    if save_count % save_step == 0:
                        if not extract_all:
                            print(f"now to generate:{len(ls)}/{train_num}")
                        else:
                            print(f"now to generate:{len(ls)}")
                        with open(pre_save_path, 'w', encoding='utf-8') as file:
                            json.dump(ls, file, ensure_ascii=False, indent=4)
                            print(f"pre save finish of {save_step} steps")
                    if len(input_prompt) + len(output_prompt) > 1000:
                        break
                    if len(ls) >= train_num and not extract_all:
                        enough_train = True
                        break
                else:
                    history_dialogue += data[index]["role"] + "�? + data[index]["sentence"]
                    if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                        history_dialogue += '�?
                    current_turn_entities += data[index]["entity"]
                    if len(current_turn_entities) != 0:
                        entities_list_eturn.append(current_turn_entities)

                    # 动态图推理
                    if len(entities_static) != ents_len or len(dynamic_entities) == 0:
                        dynamic_entities = get_dynamic_map(model, tokenizer, entities_list_eturn, entities_static)

            if enough_train:
                break
    target_dir_path = os.path.join(hard_prompt_save_path, data_type)
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
    # target_path = os.path.join(target_dir_path, "hard_prompt_t.json")
    # with open(target_path, 'w', encoding='utf-8') as file:
    #     json.dump(ls, file, ensure_ascii=False, indent=4)
    return ls


# 处理测试集文本，不同于训练集，训练集有将其分�?
def process_hardprompt_test(model, tokenizer, base_data_path, ppr_sampler, entities_map, rel_map, data_type="kamed", only_end=True, depth=3, width=3, top_num=5):
    ls = []  # 每一个论文的对话字典
    entity_map_index = {value: key for key, value in entities_map.items()}  # 这个是用实体找到其对应的索引
    data_path = os.path.join(base_data_path, data_type)

    dynamic_predict = False
    # 加载动态图文件
    if os.path.exists(test_dynamic_map_load_path.format(data_type)):
        with open(test_dynamic_map_load_path.format(data_type), 'r', encoding='utf-8') as file:
            dynamic_pre_map = json.load(file)
        print("finish loading dynamic map file of path {}".format(test_dynamic_map_load_path.format(data_type)))
    else:
        print("------------------  warning ------------------")
        print("If you want to use the complete solution, it is recommended to first conduct dynamic graph reasoning training and obtain the corresponding results before running the main program.")
        print("如果想要使用完整方案，建议先进行动态图推理训练并得到相应的结果后再进行主程序的运行")
        print("----------------------------------------------")

        dynamic_predict = True



    for dt in ["test"]:
        nf = os.path.join(data_path, data_type + "_" + dt)
        for target in tqdm(os.listdir(nf), desc=f"Building hardprompt of test", dynamic_ncols=True):

            # 直接存储到对应的索引即可
            entity_impact_factor = {
                "first": [],  # this turn appear entities
                "second": {},  # 字典，截至到目前为止所有实体的出现次数
                "third": []  # 截至到当前轮次的实体，越往后越新，越晚出现
            }  #
            history_dialogue = ""
            # dynamic_predict_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
            dynamic_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
            entities_list_eturn = []  # 二维列表，保留每一轮出现的实体
            end_file_path = os.path.join(nf, target)
            with open(end_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)["dialogues"]
            entities_static = []  # 该变量用于存储到当前轮次的所有实体，因为每一个turn进行知识推理都要使用所有出现过的实体（加入的时候要进行实体去重�?
            current_turn_entities = []  # 要保留当前轮次的实体，一轮包括patient和doctor
            for index in range(len(data)):
                # 每一轮的处理

                entity_impact_factor['first'] = [int(entity_map_index[item]) for item in data[index]["entity"] if item in entity_map_index.keys()]
                # entity_impact_factor['first'] = [int(entity_map_index[item]) for item in data[index]["entity"]]
                for i in entity_impact_factor['first']:
                    if i in entity_impact_factor['second'].keys():
                        entity_impact_factor['second'][i] += 1
                    else:
                        entity_impact_factor['second'][i] = 1
                entity_impact_factor['third'] += entity_impact_factor['first']
                ents_len = copy.deepcopy(len(entities_static))  # 对比一下实体是否有增加，没有就用原来预测出来的动态实�?
                entities_static += [item for item in data[index]["entity"] if item not in entities_static]
                # entities_static_index = [int(entity_map_index[item]) for item in entities_static]  # 将提取的实体映射成对应的index
                entities_static_index = [int(entity_map_index[item]) for item in entities_static if item in entity_map_index.keys()]  # 将提取的实体映射成对应的index

                # 对待选实体进行筛�?取top-num个实�?
                entities_static_index = search_entities_static_index(entity_impact_factor, top_num)
                if data[index]["role"] == "patient":
                    history_dialogue += data[index]["role"] + "�? + data[index]["sentence"]
                    if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                        history_dialogue += '�?
                    current_turn_entities = data[index]["entity"]
                    question = data[index]["sentence"]
                    # 静态图
                    if index != len(data) - 2:
                        continue
                    static_map = get_static_map(entities_static_index, ppr_sampler)
                    # 动态实体（由动态图推理出来�?
                    # dynamic_entities =

                    # if len(entities_static) != ents_len or len(dynamic_entities) == 0:
                    if dynamic_predict:
                        dynamic_entities = get_dynamic_map(model, tokenizer, entities_list_eturn, entities_static)
                    else:
                        dynamic_entities = dynamic_pre_map[target]['predict']

                    dynamic_entities = [entity for entity in dynamic_entities if entity not in entities_static_index]

                    # static_map and dynamic_map 的Tog异构图推�?
                    # model, tokenizer, entities, question, mini_graph, entities_map, rel_map, depth, width)


                    # 这里先用静态图去调


                    # 为什么会有没出现的实体，是这样的，所选实体没有对应三元组

                    kg_triple = extract_kg_links(model, tokenizer, entities_static_index, question, static_map,
                                                 entities_map, rel_map, depth, width, dynamic_entities)




                    if len(kg_triple) > 0:
                        kg_triple_str = ""
                        for en1, re, en2 in kg_triple:
                            kg_triple_str += f"({en1},{re},{en2}),"
                        # input_prompt = f"医生和患者的历史对话：[{history_dialogue}]；可能与回答历史对话相关的医疗事实：{}；假如你是一个医生，请你根据前面提及到的历史对话、医疗事实进行诊断回复�?
                        input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[{kg_triple_str[:-1]}]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                    else:
                        input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[null]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                    output_prompt = data[index + 1]['sentence']
                    #     "fileName": target,
                    #     "input": input_prompt,
                    #     "output": output_prompt,
                    # })
                    ls.append({
                        "fileName": target,
                        "input": input_prompt,
                        "output": output_prompt,
                        "history": history_dialogue  # 这个history_dialogue是包含本轮的提问�?
                    })
                    break
                else:
                    history_dialogue += data[index]["role"] + "�? + data[index]["sentence"]
                    if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                        history_dialogue += '�?

                    current_turn_entities += data[index]["entity"]
                    if len(current_turn_entities) != 0:
                        entities_list_eturn.append(current_turn_entities)
                    if index != len(data) - 3:
                        continue
                    # # 动态图推理
                    # if len(entities_static) != ents_len or len(dynamic_entities) == 0:
                    #     dynamic_entities = get_dynamic_map(model, tokenizer, entities_list_eturn, entities_static)
        return ls


def process_hardprompt_test2(model, tokenizer, base_data_path, ppr_sampler, entities_map, rel_map, data_type="kamed", only_end=True, depth=3, width=3, top_num=5):
    ls = []  # 每一个论文的对话字典
    entity_map_index = {value: key for key, value in entities_map.items()}  # 这个是用实体找到其对应的索引
    data_path = os.path.join(base_data_path, data_type, "MediTOD_test.json")
    with open(data_path, "r", encoding='utf-8') as file:
        dialogues_dic = json.load(file)

    for target, item in tqdm(dialogues_dic.items(), desc=f"Building hardprompt of test", dynamic_ncols=True):

        print(target)
        # 直接存储到对应的索引即可
        entity_impact_factor = {
            "first": [],  # this turn appear entities
            "second": {},  # 字典，截至到目前为止所有实体的出现次数
            "third": []  # 截至到当前轮次的实体，越往后越新，越晚出现
        }  #
        history_dialogue = ""
        # dynamic_predict_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
        dynamic_entities = []  # 用于存储被预测出来的实体(上一轮预测的结果)
        entities_list_eturn = []  # 二维列表，保留每一轮出现的实体
        data = item['utterances']
        entities_static = []  # 该变量用于存储到当前轮次的所有实体，因为每一个turn进行知识推理都要使用所有出现过的实体（加入的时候要进行实体去重�?
        current_turn_entities = []  # 要保留当前轮次的实体，一轮包括patient和doctor
        for index in range(len(data)):
            # 每一轮的处理

            entity_impact_factor['first'] = [int(entity_map_index[item]) for item in data[index]["keywords"] if item in entity_map_index.keys()]
            # entity_impact_factor['first'] = [int(entity_map_index[item]) for item in data[index]["entity"]]
            for i in entity_impact_factor['first']:
                if i in entity_impact_factor['second'].keys():
                    entity_impact_factor['second'][i] += 1
                else:
                    entity_impact_factor['second'][i] = 1
            entity_impact_factor['third'] += entity_impact_factor['first']
            ents_len = copy.deepcopy(len(entities_static))  # 对比一下实体是否有增加，没有就用原来预测出来的动态实�?
            entities_static += [item for item in data[index]["keywords"] if item not in entities_static]
            # entities_static_index = [int(entity_map_index[item]) for item in entities_static]  # 将提取的实体映射成对应的index
            entities_static_index = [int(entity_map_index[item]) for item in entities_static if item in entity_map_index.keys()]  # 将提取的实体映射成对应的index

            # 对待选实体进行筛�?取top-num个实�?
            entities_static_index = search_entities_static_index(entity_impact_factor, top_num)
            if data[index]["speaker"] == "patient":
                history_dialogue += data[index]["speaker"] + "�? + data[index]["text"]
                if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                    history_dialogue += '�?
                current_turn_entities = data[index]["keywords"]
                question = data[index]["text"]
                # 静态图
                if index != len(data) - 2:
                    continue
                static_map = get_static_map(entities_static_index, ppr_sampler)
                # 动态实体（由动态图推理出来�?
                # dynamic_entities =

                # if len(entities_static) != ents_len or len(dynamic_entities) == 0:

                dynamic_entities = [entity for entity in dynamic_entities if entity not in entities_static_index]

                # static_map and dynamic_map 的Tog异构图推�?
                # model, tokenizer, entities, question, mini_graph, entities_map, rel_map, depth, width)
                # 这里先用静态图去调
                print(f"dynamic_entities = {dynamic_entities}")


                # 为什么会有没出现的实体，是这样的，所选实体没有对应三元组
                print(entities_static_index)

                kg_triple = extract_kg_links(model, tokenizer, entities_static_index, question, static_map,
                                             entities_map, rel_map, depth, width, dynamic_entities)




                if len(kg_triple) > 0:
                    kg_triple_str = ""
                    for en1, re, en2 in kg_triple:
                        kg_triple_str += f"({en1},{re},{en2}),"
                    # input_prompt = f"医生和患者的历史对话：[{history_dialogue}]；可能与回答历史对话相关的医疗事实：{}；假如你是一个医生，请你根据前面提及到的历史对话、医疗事实进行诊断回复�?
                    input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[{kg_triple_str[:-1]}]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                else:
                    input_prompt = f"医生和患者的对话历史：[{history_dialogue}]；可能与医生回复相关的医疗事实：[null]；你是一个专业的医生，正在给患者进行问诊，请根据前面提及的“医生和患者的对话历史”和“可能与医生回复相关的医疗事实”，进行问诊回复�?
                output_prompt = data[index + 1]['sentence']
                #     "fileName": target,
                #     "input": input_prompt,
                #     "output": output_prompt,
                # })
                ls.append({
                    "fileName": target,
                    "input": input_prompt,
                    "output": output_prompt,
                    "history": history_dialogue  # 这个history_dialogue是包含本轮的提问�?
                })
                break
            else:
                history_dialogue += data[index]["speaker"] + "�? + data[index]["text"]
                if history_dialogue[-1] not in ['�?, '?', '�?, '�?] and history_dialogue != '':
                    history_dialogue += '�?

                current_turn_entities += data[index]["keywords"]
                if len(current_turn_entities) != 0:
                    entities_list_eturn.append(current_turn_entities)
                if index != len(data) - 3:
                    continue

    return ls


# test
if __name__ == '__main__':
    base_data_path = "C:/Users/13076/PycharmProjects/paper2/new_data"
    data_type = "kamed"
    process_hardprompt(base_data_path, data_type)
