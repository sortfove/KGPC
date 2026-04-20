# 这里我们先尝试利用api进行提取

import copy
import time
import json
import os
import time
from tqdm import tqdm
import torch
import sys
# base_Dpath = "../data/kamed"
# generate_path = "../output/SOAP_extract_multipart.json"
# output_base_path = "../output"
# # output_base_path = "../output_test"

# spark api 的基础配置
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

import os
from openai import OpenAI

#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查�?

# --- Relevant configuration of the Spark large model API --- #
SPARKAI_URL = ''
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
SPARKAI_DOMAIN = ''

# --- Relevant configuration of the Qwen large model API --- #
qwen_api_key = ""
qwen_api_url = ""
model_name = "qwen3-max"



using_model = "qwen_api"


# 一般只改变下面
# number_dialogue = 5 # 从取前面10000条数据对话段来进行测�?
number_save = 3 # 每处�?000条数据保存一�?
all_dialogue = True  # all_dialogue表明将进行所有对话的提取
type_data = ['train']  # 这里只提取训练集
type_dataset = "meddg"
base_Dpath = f"../data/{type_dataset}"
# generate_path = "../output/SOAP_extract_multipart.json"
output_base_path = f"../output/{type_dataset}_soap_extract_result"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
english = True  # 数据集是否未英文

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)


# 将对于的文件内的所有句子都提取出来
def extract_dialogue(file_path):
    dic = {} # 存放各个对话段的对话 key:对话文件�?�?value：对于的对话
    for target in os.listdir(file_path):
        end_path = os.path.join(file_path, target)
        with open(end_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # ls = []
            dialogue_str = ''
            for turn in data['dialogues']:
                # ls.append(turn['sentence'])
                dialogue_str += turn['role'] + ':' + turn['sentence'] + '.'
            dic.update({target: dialogue_str})
    return dic

def extract_dialogue2(file_path):  # 所有数据放在一个文件内，这边输入直接指定到具体位置的json文件
    dic = {}

    with open(f"{file_path}.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        # ls = []
        dialogue_str = ''
        for key, value in data.items():
            # ls.append(turn['sentence'])
            for turn in value['utterances']:
                dialogue_str += turn['speaker'] + ':' + turn['text'] + '.'
            dic.update({key: dialogue_str})
    return dic

# 总共需要有四个元素
# query、instructions、contexts、answers （answers不用加上�?
# 因为我们有一大类任务(multi-part完成SOAP的提�?，一大类中包�?小类（SOAP�?
data = {
    "query": "xxx",
    "context": "",
    # instructions任务指令，指导做什么任�?
    "instructions": {
        "S": "xxx",
        "O": "xxx",
        "A": "xxx",
        "P": "xxx"
    }
}

data_english = {
    "query": "SOAP notes are a structured method used by physicians to document patient visits and track their progress.Subjective refers to the patient's subjective description of their condition.Objective includes other signs or symptoms observed by the physician during the consultation.Assessment refers to the physician’s diagnostic thought process or clinical reasoning during the visit.Plan outlines the physician’s future diagnostic conclusions or care plans.You are a professional doctor. Use the SOAP note format to extract information from the following doctor-patient consultation dialogue, following the given steps and template requirements.",
    "context": "",
    # instructions任务指令，指导做什么任�?
    "instructions": {
        "S": "Summarize the patient's subjective description of their condition from this dialogue, with the output format strictly following #S: ...",
        "O": "Summarize any other illnesses or symptoms the patient mentions during the doctor's questioning in this dialogue, with the output format strictly following #O: ...",
        "A": "Summarize the sequence of the doctor’s questioning actions in this dialogue, with the output format strictly following #A: ...",
        "P": "Summarize the diagnostic conclusions or care plans the doctor is likely to provide in the future based on this dialogue, with the output format strictly following #P: ..."
    }
}


SOAP_target = {
    "S": "Subjective",
    "O": "Objective",
    "A": "Assessment",
    "P": "Plan"
}

# query = data['query']
# instructions = data['instruction']
# contexts = data['context']

def get_inputPrompt(txt, english=False, is_Test=False):
    if english:
        data = data_english

    # 因为各个对话段只有对话内容不一样，任务SOAP都是一样的，因此这里对data中的context标签进行补充
    data['context'] = "###Dialogue\n" + txt
    num = len(data['instructions'])  # multi-part 任务的数�?

    query = data['query']


    instructions = data['instructions']
    contexts = data['context']
    # filtered_instructions = '\n'.join(
    #     [f"#{i}: {instructions[str(i)]}" for i in range(1, num + 1) if str(i) in instructions])

    # 在具体文本中需要加上的prompt
    if is_Test == True:
        if english:
            other_str = "Avoid including information that is not explicitly mentioned in the dialogue. If a section lacks relevant information, skip that part. For example, if no specific subjective information is provided in the dialogue, write \"Nothing\" in the subjective section."
        else:
            other_str = "避免包含对话中没有明确提及的信息。如果某个部分没有提供相关信息，就跳过该部分。例如，如果对话中没有提供具体的主观信息，就在主观部分写“无�?"
        query += other_str

    filtered_instructions = '\n'.join(
        [f"#{SOAP_target[item]}: {instructions[item]}" for item in instructions.keys()])

    filtered_contexts = contexts
    final_string = query + "\n" + filtered_instructions + "\n\n" + filtered_contexts

    # soap_dic = {
    #     'S': '患者的主观病情自述',
    #     'O': '医生问诊过程中患者其它病情或其它症状',
    #     'A': '医生的诊断动作流�?,
    #     'P': '未来护理计划或者医生可能做的判�?
    # }
    # return f"对话:[{txt}],请帮我提取这段对话中�?患者的主观自述、客观事实、医生诊断、未来护理计划（诊断结果）四部分，返回格式以一对一的形式�?
    return final_string



def get_soap_with_api(prompt, llm_model):
    soap_dic = {}
    flag = 0
    while True:
        try:
            messages = [ChatMessage(
                role="user",
                content=get_inputPrompt(prompt)
            )]
            handler = ChunkPrintHandler()
            res = llm_model.generate([messages], callbacks=[handler]).generations[0][0].text
            # ls = res['output']['text'].strip().split('\n')
            ls = res.strip().split('\n')
            ls = list(filter(None, ls))
            for i, item in zip("SOAP", ls):
                soap_dic.update({i: item[item.index(": ") + 1:]})
            print(soap_dic)
            break
        except:

            flag += 1
            if flag > 8:
                for i in "SOAP":
                    soap_dic.update({i: "Nothing"})
                print(soap_dic)
                return soap_dic

            llm_model = ChatSparkLLM(
                spark_api_url=SPARKAI_URL,
                spark_app_id=SPARKAI_APP_ID,
                spark_api_key=SPARKAI_API_KEY,
                spark_api_secret=SPARKAI_API_SECRET,
                spark_llm_domain=SPARKAI_DOMAIN,
                streaming=False,
            )
            continue

    # 返回是一个字典（键为SOAP�?
    return soap_dic


if __name__ == '__main__':
    start_time = time.time()
    #  测试一下能不能直接初始化，再调用就可以加速了
    llm_model = None
    if using_model == "spark_api":
        llm_model = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
    elif using_model == "qwen_api":
        llm_model = OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_api_url,
        )



    for item in type_data:
        file_path = os.path.join(base_Dpath, type_dataset + "_" + item)
        #########################################换数据集
        if type_dataset in ['meddg', 'kamed', 'meddialog']:
            dialogue = extract_dialogue(file_path)
        # 下面extract_dialogue2适合MediTOD数据�?
        elif type_dataset in ['MediTOD']:
            dialogue = extract_dialogue2(file_path)
        else:
            print("# ------------------- attention�?------------------- #")
            print("The dataset label of type_dataset is filled in incorrectly, please correct it!")
            print("type_dataset的数据集标签填写错误，请纠正�?)
            sys.exit(1)  # �?表示异常退�?

        # 上面已经完成对对话段句子的提�?
        # 通过对对话段分析，提取出SOAP
        # 这里写一个字典进行存储各个对话段提取出来的SOAP
        dic_total = {}
        error_dic = {
            'error': [],
            'result_question': []
        }
        count = 0
        flag = 0
        begin = 0
        if all_dialogue:
            number_dialogue = len(list(dialogue.keys())[begin:])
        for key, value in tqdm(list(dialogue.items())[begin:], desc=f"Building SOAP of {type_dataset}", dynamic_ncols=True, total=number_dialogue):
            # if not "072525.session.json" == key:
            #     continue

            # count += 1
            # count += 1
            if count >= number_dialogue:
                break
            if count % number_save == 0 and count != 0:
                question_path = f"../output/MediTOD_soap_extract_result/question_{count-number_save+1}_{count}.json"
                with open(question_path, 'w', encoding="utf-8") as file_q:
                    json.dump(error_dic, file_q, ensure_ascii=False, indent=4)
                    print(f"question.json is save")
                gen_path = os.path.join(output_base_path, f"SOAP_extract_multipart_{count-number_save+1}_{count}.json")
                flag = count
                with open(gen_path, 'w', encoding='utf-8') as file:
                    json.dump(dic_total, file, ensure_ascii=False, indent=4)
                    print(f"{gen_path} is save")
                dic_total = {}
            # 通过一个字典来存储SOAP对应的内容：{S:...,O:...}
            soap_extract_dic = {}
            repeat_apply = 0
            while True:
                try:
                    if using_model == "qwen_api":
                        completion = llm_model.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": get_inputPrompt(value, english)},
                            ],
                            stream=False
                        )
                        response = completion.choices[0].message.content
                    else:
                        messages = [ChatMessage(
                            role="user",
                            content=get_inputPrompt(value, english)
                        )]
                        handler = ChunkPrintHandler()
                        response = llm_model.generate([messages], callbacks=[handler]).generations[0][0].text


                    if type_dataset == "MediTOD":
                        soap_ls = response.split('\n')
                        soap_ls_q = []
                        for i in soap_ls:
                            if i != "":
                                soap_ls_q.append(i)
                        soap_ls = soap_ls_q

                    else:
                        soap_ls = response.strip().split('\n')
                    soap_ls = list(filter(None, soap_ls))
                    if len(soap_ls) != len("SOAP"):
                        soap_extract_dic = None
                    else:
                        for i, item in zip("SOAP", soap_ls):
                            try:
                                soap_extract_dic.update({i: item[item.index(": ")+1:]})
                            except ValueError:
                                soap_extract_dic = None
                    break
                except (TypeError, KeyError, ConnectionError) as e:
                    print(f"{e}-{key}")
                    time.sleep(8)
                    if using_model == "qwen_api":
                        llm_model = OpenAI(
                            api_key=qwen_api_key,
                            base_url=qwen_api_url,
                        )
                    else:
                        llm_model = ChatSparkLLM(
                            spark_api_url=SPARKAI_URL,
                            spark_app_id=SPARKAI_APP_ID,
                            spark_api_key=SPARKAI_API_KEY,
                            spark_api_secret=SPARKAI_API_SECRET,
                            spark_llm_domain=SPARKAI_DOMAIN,
                            streaming=False,
                        )
                    if repeat_apply >= 2:
                        soap_extract_dic = None
                        if not soap_extract_dic:
                            error_dic['question'].append({
                                "name": key,
                                "other": response
                            })
                        break
                    repeat_apply += 1
                    continue
                except Exception as e:
                    if not soap_extract_dic:
                        error_dic['error'].append({
                            "name": key,
                            "other": e
                        })
                    soap_extract_dic = None


            dic_total.update({
                key: copy.deepcopy(soap_extract_dic)
            })
            count += 1


        gen_path = os.path.join(output_base_path, f"SOAP_extract_multipart_{flag+1}_{flag+len(dic_total)}.json")
        with open(gen_path, 'w', encoding='utf-8') as file:
            json.dump(dic_total, file, ensure_ascii=False, indent=4)
            print(f"{gen_path} has been saved successfully.")

    end_time = time.time()
    print(f"花费的时间：{end_time - start_time}")
    # local_time = time.localtime(end_time - start_time)
    # formatted_time = time.strftime("%H:%M:%S", local_time)

    time_diff = end_time - start_time
    # 将时间差转换为小时、分钟和�?
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 格式化输�?
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    print(f"花费的时间：{formatted_time}")

