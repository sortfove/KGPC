import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import numpy as np
import os
from tqdm import tqdm
from multi_part_extract_soap import get_inputPrompt, get_soap_with_api
# import logging
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s')

model_path = "xxx"  

save_path = "process_output/{}/soap_embedding"
# save_path = "process_output"
# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
hidden_size = 4096
batch_size = 1
# model_type = "llama"

# 这里还有考虑到测试集和验证集的嵌入用不用跟着读取加载进来�?
# 这一步未做，因为我考虑到，对话的不一定说是对话段整段说完进行识别提取的，这些暂时别管（这个已经解决了，soap提取不到的部分直接显示无就好了）

# 模式一：SOAP 整体拼接作相似度
# data_path 数据位置
# data_path = "../output/SOAP_extract_multipart_new.json"
# data_path = "../process_output/kamed/kamed_soap_extract_result.json"
# data_path = "../process_output/kamed/kamed_soap_extract_result_drop_null.json"
data_path = "./process_output/{}/{}_soap_extract_result.json"

SOAP_target = {
    "S": "Subjective",
    "O": "Objective",
    "A": "Assessment",
    "P": "Plan"
}

# 将每一个对话段的SOAP拼接在一�?data是一个字典，key为数据文件名，value为字典，键位分别为SOAP，值为对应文本)
def get_SOAP_CAT(dic):
    for file_name, target_data in dic.items():
        SOAP_str = ""
        for key, value in target_data.items():
            SOAP_str += SOAP_target[key] + ":" + value + "."
        dic[file_name] = SOAP_str  # 直接修改



# 输入对话段获取嵌�?
# prompts 多个对话段，这样可以批处�?
def get_embedding_with_prompt(prompts, model, tokenizer, model_type, max_length=1028):
    # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if model_type == "llama" or model_type == "deepseek_dist_llama" or model_type == "qwen1.5":
        with torch.no_grad():
            # 这里有个问题能不能直接将tokenizer索引到对应的设备后面的数据就不用变了
            # inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            # inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=max_length, return_token_type_ids=False)
            inputs = tokenizer(prompts, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True, max_length=max_length)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            # inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)
            # results = model(**inputs, output_hidden_states=True)
            results = model(**inputs, output_hidden_states=True)
            last_hidden_states = results.hidden_states[-1]
            # 549表示文本提取的特征数，我们直接取所有特征的平均值作为语�?
            sequence_mean = last_hidden_states.mean(dim=1)
            # 这里测试一下CLS这个token有没有在llama中存�?llama1好像没有，暂时先别管)
    else:
        raise NotImplementedError

    return sequence_mean # 待修改，返回的是批处理最后一层的嵌入


def build_embedding(data_input, model, tokenizer, hidden_size, batch_size, model_type):
    # 这里面的data_input是一个字�?

    # 分批加载
    # 因为我们要分批进行获取嵌入，因此我们需要将字典分成两部分（两个列表），第一个列表保存key,第二个列表保存value进行嵌入获取
    # 这两个列表一定是有序的，其实也不需要文件名吧，只需要对应的嵌入即可�?
    name_ls = []  # 用于存储对应的对话段名（因为考虑到不需要对于的文件名，这里的文件暂时不写人），顺便写入吧，后续可以看看哪些对话段的SOAP相似(这里直接将列表写入文件即�?
    message_ls = []  # 用于存储对应的信�?
    for key, value in data_input.items():
        name_ls.append(key)
        message_ls.append(value)

    total_len = len(message_ls)
    # 初始化存储embedding的矩�?get_embedding_with_prompt返回的矩阵格式要和datastore_soap_embedding（注意点�?
    datastore_soap_embedding = np.zeros([total_len, hidden_size], dtype=np.float32)

    for start in tqdm(range(0, total_len, batch_size), desc="Building embedding", dynamic_ncols=True):
        # 这里的batch_size 用不用减1（不用，包前不包后）
        end = min(start + batch_size, total_len)
        # datastore_soap_embedding[start:end] = get_embedding_with_prompt(message_ls[start:end], model, tokenizer).cpu().numpy()
        datastore_soap_embedding[start:end] = get_embedding_with_prompt(message_ls[start:end], model, tokenizer, model_type).cpu().to(torch.float32).numpy()
    return datastore_soap_embedding, name_ls

# 加载嵌入，如果为生成过则重新生成
def load_datastore(save_path, data_input, model, tokenizer, hidden_size, batch_size, model_type, data_type):
    # if os.path.exists(save_path + '/' + model_type + '.soap_embedding.npy'):
    if os.path.exists(save_path + '/' + model_type + '.soap_embedding.npy'):
        datastore_soap_embedding = np.load(os.path.join(save_path, model_type + '.soap_embedding.npy'))
        soap_name = np.load(os.path.join(save_path, model_type + '.soap_name.npy'))
        # soap_name = np.load(save_path + '.soap_name.npy')
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 对话段的读取
        with open(data_path.format(data_type, data_type), 'r', encoding='utf-8') as file:
            dic = json.load(file)
        get_SOAP_CAT(dic)  # 直接修改
        datastore_soap_embedding, soap_name = build_embedding(dic, model, tokenizer, hidden_size, batch_size, model_type)
        np.save((save_path + '/' + model_type + '.soap_embedding.npy'), datastore_soap_embedding)
        np.save(save_path + '/' + model_type + '.soap_name.npy', soap_name)
    # 后面测试一下这个代码是否可有可�?
    # datastore_soap_embedding = torch.from_numpy(datastore_soap_embedding).to(DEVICE)
    datastore_soap_embedding = torch.from_numpy(datastore_soap_embedding).to('cpu')
    return datastore_soap_embedding, soap_name


# 获取相似的前k个SOAP_embedding，将对应的k个作均值输�?
def get_finial_embedding(embedding, datastore_soap_embedding, soap_name, top_k=1):
    embedding = embedding.to('cpu')

    # embedding = embedding.mean(dim=0).unsqueeze(0)
    # queries [B, H]
    # keys [L, H]
    dists = ((datastore_soap_embedding.unsqueeze(0) - embedding.unsqueeze(1)) ** 2).sum(-1)  # [B, L] # 距离张量
    # 这里测试一下dists的结�?
    # scaled_dists = -1.0 / knn_T * dists  # 对距离进行缩�?（不进行距离缩进会怎么样？�?
    top_dists, top_index = torch.topk(dists, top_k)  # [B, K]
    k_embedding = datastore_soap_embedding[top_index, :]  # k为矩�?
    # 先将该矩阵作均值处理，作为softprompt
    soft_prompt = k_embedding.mean(dim=1)  # 比较以下dim=1和dim=0
    return soft_prompt

# 这个函数用于主函数流程的调用，也就是用于生存对应的softprompt
# dialogue 对话�? dialogue_SOAP输入的是文本(一个字符串)
def get_soap_softprompt(dialogue_SOAP, model, tokenizer, top_k, batch_size, hidden_size, model_type, data_type):
    embedding = get_embedding_with_prompt(dialogue_SOAP, model, tokenizer, model_type)
    save_path_target = save_path.format(data_type)
    datastore_soap_embedding, soap_name = load_datastore(save_path_target, dialogue_SOAP, model, tokenizer, hidden_size, batch_size, model_type, data_type)
    final_embedding = get_finial_embedding(embedding, datastore_soap_embedding, soap_name, top_k)
    final_embedding = final_embedding.to(DEVICE)
    return final_embedding


# 这里输入模型让模型提取输入文本的soap（这个函数也可以修改一下，由api处理文本，后面看预训练模型的输出格式或者是soap提取效果�?
def get_soap_str(model, tokenizer, input_text, model_type, pattern, api_model):  # pattern�?�?）表示的是哪种模式，1表示soap作为字符串拼接起来，2表示soap分开计算相似�?
    prompt = get_inputPrompt(input_text, is_Test=True)
    # 通过模型实现回复提取soap，获取完类似于训练集的格式拼接在一�? 这里记得测试一下模型的输出
    dic = get_soap_with_api(prompt, api_model)
    soap_str = ""
    for key, value in dic.items():
        soap_str += SOAP_target[key] + ":" + value
    if pattern == 1:
        # soap_str = ""
        # for key, value in dic.items():
        #     soap_str += SOAP_target[key] + ":" + value
        return soap_str
    elif pattern == 2:  # 这里暂未确定
        return dic, soap_str

def soap_dic_to_str(dic, pattern):
    if pattern == 1:
        soap_str = ""
        for key, value in dic.items():
            soap_str += SOAP_target[key] + ":" + value
        return soap_str
    elif pattern == 2:  # 这里暂未确定
        return dic

# 用于测试
if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
    # model = ""
    # tokenizer = ""

    # 对话段的读取
    with open(data_path, 'r', encoding='utf-8') as file:
        dic = json.load(file)
    get_SOAP_CAT(dic)  # 直接修改
    #
    datastore_soap_embedding, soap_name = load_datastore(save_path, dic, model, tokenizer, hidden_size=4096, batch_size=4, model_type="llama")
    #
    print(datastore_soap_embedding.shape)
    # 这里只用于测�?
    dialogue = "patient:医生您好，这是我的问题：头顶部位脱发（男�?9岁）.doctor:你好，能否上传照片看看？.patient:图片因隐私问题无法显�?doctor:家里人有�?patient:图片因隐私问题无法显示有.doctor:考虑雄激素性脱发可�?patient:脂溢性脱发？图片因隐私问题无法显�?doctor:目前没有太好的方法，国外指南可以吃非那雄胺，但是要长期吃，有效率大概百分之七十，可能还有副作用。这个脱发除了影响美观其他不影响。目前最新的研究可能可以局部外用非那雄胺，副作用会小些，但还在研究当中.patient:外用非那雄胺�?doctor:嗯，也叫脂溢性脱�?patient:非那雄胺一毫克不是吃的吗？药片如何外用.doctor:还在研究中�?patient:单纯的外用米诺地尔酊可以�?doctor:可以试试.patient:还有其他更好建议�?doctor:可以吃点维生素b6控制油脂分泌.patient:吃维生素b族可以吗可以吃非那雄胺吗.doctor:也行，针对性没那么强非那雄胺吃不吃看你自己决定.patient:保列治可以吗分成五分.doctor:应该可以." #  新对话文本SOAP,由大模型
    dialogue_SOAP = get_soap_str(model, tokenizer, dialogue, model_type, 1)
    print(f"dialogue_SOAP={dialogue_SOAP}")
    softprompt = get_soap_softprompt(dialogue_SOAP, model, tokenizer, top_k=5, batch_size=batch_size, hidden_size=hidden_size, model_type=model_type)

