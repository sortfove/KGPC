import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# 这个代码是用于实现动态图
# 输入：前面轮次提取的实体ls1、前面轮次被预测出来的实体pr1、当前轮次提取的实体ls2、前一轮预测到的本轮的实体pr2
# 返回：下一轮可能的实体、返回的新三元组集合（在原三元组上加上被预测的结点与一条虚边跟提取出来的主实体相连（这里注意一下对看实体是否存在于原图谱中））

# def get_prompt(ls1, pr1, ls2, pr2):
#     prompt = (f"假如你是一个医生，你做的任务是根据上一轮和本轮对话出现的实体去预测下一轮对话可能出现的实体，预测出的实体数量不超过3个，仅输出预测的实体即可\n"
#               f"#上一轮对话出现的实体:{','.join(map(str, ls1 + pr1))}\n"
#               f"#本轮对话出现的实�?{','.join(map(str, ls2 + pr2))}\n"
#               f"#输出格式为下一轮可能出现的医疗实体:...,...,...\n"
#               f"#预测不出的输出格式为 没有")
#     return prompt

# 这里接受的是Turn 1 - 3 [{e1,e2},{e3},{e1,e4}] �?实体集合
def get_prompt(ls, entities):
    prompt = ""
    if len(ls) == 1:
        entities_str = ""
        for item in entities:
            entities_str += item + ','
        prompt = (f"样例输入�?实体�?发烧,头晕\n"
                  f"样例答案：流行性感�?咳嗽\n"
                  f"假如你是一个实体预测专家，你需要学习上面的例子的格式，根据#实体集去预测下一轮的可能出现新的医疗实体�?
                  f"要求预测出的实体数量不超�?个，且不能是#实体集出现过的，仅输出预测出来的实体即可，其它任何东西都不要输出\n"
                  f"#实体�?{entities_str[:-1]}\n"
                  f"#预测不出的输出格式为 no")
    elif len(ls) > 1:
        str1 = ""
        for i in range(len(ls)-1):
            str1 += ','.join(ls[i]) + "推理�? + ','.join(ls[i+1]) + '\n'
        str2 = ','.join(entities)
        prompt = (f"例子：\n"
                  f"#推理知识：头�?发烧推理出咳嗽\n头疼,发烧,咳嗽推理出乏�?流行性感冒\n头疼,发烧,咳嗽,乏力,呼吸道感染推理出流行性感冒\n"
                  f"#实体集：头疼,发烧,咳嗽,乏力,呼吸道感�?流行性感冒\n"
                  f"#样例输出：大量饮�?抗生�?奥司他韦,扁桃体炎\n\n"
                  f"\n假如你是一个医疗实体预测专家，你需要学习上面的例子的格式及其推理规律，根据#实体�?
                  f"去预测下一轮的可能出现新的医疗实体�?
                  f"要求预测出的实体数量不超�?个，且不能是#实体集出现过的，你只需要输出预测出来的实体即可，各个医疗实体之间用英文的逗号隔开，其它任何东西一定不要输出\n"
                  f"#推理知识：{str1}\n"
                  f"#实体集：{str2}\n"
                  f"#如果预测不出直接输出 no")
    return prompt


# 默认虚边的index�?5
# 输入本轮提取的实体，和预测出来的实体，进行广�?
# 返回结果为虚三元组列�?
def dynamic_triple(ls2, predict):
    Dtriple_ls = []
    for entity in ls2:
        for pentity in predict:
            Dtriple_ls.append([entity, 25, pentity])
    return Dtriple_ls

def get_predict(model, tokenizer, prompt):
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
    if "no" in result:
        return []
    result = result.strip(' ')
    if "�? in result:
        # entities = result[result.index("�?)+1:]
        entities = result[result.rfind("�?)+1:]
        predict = entities.split(",")
    elif ":" in result:
        # entities = result[result.index(":") + 1:]
        entities = result[result.rfind(":")+1:]
        predict = entities.split(",")
    else:
        predict = result.split(",")

    return predict

# 外部函数调用该函数可以获取下一轮的三元组的预测、也就是当前turn动态图(一个返回值：预测出来的实�?
def get_dynamic_map(model, tokenizer, ls, entities):
    prompt = get_prompt(ls, entities)
    # 获取预测出来的实�?
    predict = get_predict(model, tokenizer, prompt)
    # 获取预测出来的三元组
    # predict_triple = dynamic_triple(ls2, predict)
    # return predict, predict_triple
    # 由于可能出现异常输出情况，这里将对该情况作为反映�?
    for item in predict:
        if len(item) >= 15:
            return []
    return predict

if __name__ == '__main__':
    model_path = "xxxx"
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    model = model.to("cuda:0")

    ls1 = ["胃痛", "发烧"]  
    pr1 = []  
    ls2 = ["肠胃�?]  
    pr2 = []  
    new_pre = []  
    prompt = get_prompt(ls1, pr1, ls2, pr2)
    print(prompt)

    predict = get_predict(model, tokenizer, prompt)
    predict_triple = dynamic_triple(ls2, predict)
    print(predict_triple)
