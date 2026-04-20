import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import warnings
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler

from static_map import get_ppr_sample
from linear import process_hardprompt_linear
from process import process_hardprompt, process_hardprompt_test, extract_prompt
from pattern1 import get_soap_softprompt, get_soap_str, soap_dic_to_str




# ------------------- parameter ------------------- #
# data_type: Dataset name
# Chinese dataset: ['kamed', 'meddialog', 'meddg']
data_type = "meddg"
# model_type: Model name
# [llama, qwen, deepseek_dist_llama]
model_type = "llama"
# generate_configuration: Relevant configurations during the generation process of large models
generate_configuration = {
    "temperature": 0.80,
    "top_p": 0.80
}
hidden_size = 4096  # Llama3.1-8B-Chinese-Chat dim = 4096
dtype = torch.bfloat16

# DEVICE: Device selection
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# using_fine_turn: Whether to fine-tune the model
using_fine_turn = False
train_num = 5000  # 训练集数�?

is_test = True
base_data_path = "../data"
fine_tune_path = f"./process_output/{data_type}/{model_type}_fine_tune_data"


# n_samp_ent: The number of entities extracted using the PageRank algorithm
n_samp_ent = 30  # 提取的实体数量（通过PPR提取的top-k个）
if data_type == "meddg":
    n_samp_ent = 15



# Relevant configurations for the reasoning process of TOG
tog_depth = 2
tog_width = 2
num_epoch = 1
top_k = 5  # softprompt找到相似编码的个�?
batch_size = 1



if using_fine_turn:
    if not os.path.exists(fine_tune_path):
        os.mkdir(fine_tune_path)












if model_type == "llama":
    model_path = "xxx"  # 预训练模型路�?
elif model_type == "qwen1.5":
    model_path = "xxx"
elif model_type == "deepseek_dist_llama":
    model_path = "xxx"

hardprompt_test_path = f"./process_output/{data_type}/hard_prompt_test.json"
hardprompt_path = f"./process_output/{data_type}/hard_prompt.json"
train_soap_path = f"./process_output/{data_type}/{data_type}_soap_extract_result.json"
hardprompt_test_soap_path = f"./process_output/{data_type}/hard_prompt_soap_test.json"


# --- Relevant configuration of the Spark large model API --- #
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_APP_ID = 'xxx'
SPARKAI_API_SECRET = 'xxx'
SPARKAI_API_KEY = 'xxx'
SPARKAI_DOMAIN = 'generalv3.5'



warnings.filterwarnings('ignore', category=UserWarning, module='langchain')
model_api = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=False,
)





def main():
    base_save_path = os.path.join("process_output", data_type)


    # Load relevant parameters of the large model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    model = model.to(DEVICE)

    if is_test:
        ppr_sampler, entities_map, rel_map = get_ppr_sample(n_samp_ent, 10, data_type=data_type,
                                                            save_base_path=base_save_path)
        if linear_extract_kg:
            if not os.path.isfile(linear_hardprompt_test_path):
                hardprompt_test_ls = process_hardprompt_linear(base_data_path, data_type, "test", ppr_sampler, entities_map, rel_map, train_num)
                with open(linear_hardprompt_test_path, 'w', encoding='utf-8') as file:
                    json.dump(hardprompt_test_ls, file, ensure_ascii=False, indent=4)
            print("Building test linear finish!")

        # 使用linear进行图谱提取
        else:
            if not os.path.isfile(hardprompt_test_path):
                # 生成test文件
                hardprompt_test_ls = process_hardprompt_test(model, tokenizer, base_data_path, ppr_sampler, entities_map,
                                                             rel_map,
                                                             data_type, tog_depth, tog_width)
                with open(hardprompt_test_path, 'w', encoding='utf-8') as file:
                    json.dump(hardprompt_test_ls, file, ensure_ascii=False, indent=4)

    lora_config = None
    if model_type == "qwen1.5" or model_type == "llama" or model_type == "deepseek_dist_llama":
        lora_config = LoraConfig(
            r=8,  # LoRA的低秩�?
            lora_alpha=32,  # LoRA的缩放因�?
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LoRA应用的模�?
            lora_dropout=0.1  # 这个不确定用不用�?
        )
    peft_model = get_peft_model(model, lora_config).to(dtype)


    # 微调
    if using_fine_turn:
        # 检查是否存在预处理的数据，若没有则生成
        if not os.path.exists(os.path.join(fine_tune_path, 'model_parameters.pth')):
            # linear的情�?
            if linear_extract_kg:  ### linear_hardprompt_path
                if not os.path.isfile(linear_hardprompt_path):
                    hardprompt_ls = []
                    # 这里还要改一下，改成字图�?
                    ppr_sampler, entities_map, rel_map = get_ppr_sample(n_samp_ent, 10, data_type=data_type,
                                                                        save_base_path=base_save_path)
                    # hardprompt_ls = process_hardprompt_linear()
                    hardprompt_ls = process_hardprompt_linear(base_data_path, data_type, "train", ppr_sampler,
                                                              entities_map,
                                                              rel_map, train_num)
                    with open(linear_hardprompt_path, 'w', encoding='utf-8') as file:
                        json.dump(hardprompt_ls, file, ensure_ascii=False, indent=4)
                else:
                    with open(linear_hardprompt_path, 'r', encoding='utf-8') as file:
                        hardprompt_ls = json.load(file)

            else:
                if not os.path.isfile(hardprompt_path):
                    hardprompt_ls = []
                    ppr_sampler, entities_map, rel_map = get_ppr_sample(n_samp_ent, 10, data_type=data_type,
                                                                        save_base_path=base_save_path)
                    hardprompt_ls = process_hardprompt(model, tokenizer, base_data_path, ppr_sampler, entities_map,
                                                       rel_map,
                                                       data_type, tog_depth, tog_width, old_train=hardprompt_ls,
                                                       train_num=train_num,
                                                       pre_save_path=hardprompt_path)
                    with open(hardprompt_path, 'w', encoding='utf-8') as file:
                        json.dump(hardprompt_ls, file, ensure_ascii=False, indent=4)
                else:
                    with open(hardprompt_path, 'r', encoding='utf-8') as file:
                        hardprompt_ls = json.load(file)
                    print(f"hardprompt_len = {len(hardprompt_ls)}")
                    if len(hardprompt_ls) < train_num:
                        print(len(hardprompt_ls))

                        ppr_sampler, entities_map, rel_map = get_ppr_sample(n_samp_ent, 10, data_type=data_type,
                                                                            save_base_path=base_save_path)
                        hardprompt_ls = process_hardprompt(model, tokenizer, base_data_path, ppr_sampler, entities_map,
                                                           rel_map, data_type, tog_depth, tog_width,
                                                           old_train=hardprompt_ls, train_num=train_num,
                                                           pre_save_path=hardprompt_path)

            ft_data_dic = []
            peft_model.train()
            # 定义优化器和损失函数
            optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-4, eps=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            # 在hardprompt忘记给每一条数据集获取对应的编号了
            # 这里加上编号，方便soap提取存储就不用无限调用api
            # for item in hardprompt_ls:
            flag = 0
            for i in range(len(hardprompt_ls)):
                hardprompt_ls[i]['flag'] = flag
                flag += 1
            soap_remind = {}

            if not strip_pattern:
                with open(train_soap_path, 'r', encoding='utf-8') as file_s:
                    train_soap = json.load(file_s)
                # train_soap_len的作用是判断是否有增加soap的提取，因为部分dialogue的soap并没有提取到
                base_train_soap_len = len(train_soap.keys())

            #
            # 构建微调数据�?
            for epoch in range(num_epoch):
                total_loss = 0
                total_length = len(hardprompt_ls)
                for item in tqdm(hardprompt_ls[:train_num], desc=f"Fine-tune ... of epoch {epoch}/{num_epoch}",
                                 dynamic_ncols=True):  # 这里只取�?0条数�?
                    dialogue = item['history']
                    if len(item['input']) + len(item['output']) > 1000:
                        total_length -= 1
                        continue
                    # if item['flag'] not in soap_remind.keys():
                    #     dialogue_SOAP = get_soap_str(model, tokenizer, dialogue, model_type, 1, model_api)
                    #     soap_remind.update({
                    #         item['flag']: dialogue_SOAP
                    #     })
                    # else:
                    #     dialogue_SOAP = soap_remind[item['flag']]
                    if not strip_pattern:
                        if item['fileName'] in train_soap.keys():
                            dialogue_SOAP = train_soap[item['fileName']]
                            dialogue_SOAP = soap_dic_to_str(dialogue_SOAP, pattern=1)
                        else:
                            dialogue_SOAP_dic, dialogue_SOAP = get_soap_str(model, tokenizer, dialogue, model_type, 2,
                                                                            model_api)

                            train_soap.update({item['fileName']: dialogue_SOAP_dic})
                        soft_embedding = get_soap_softprompt(dialogue_SOAP, model, tokenizer, top_k, batch_size,
                                                             hidden_size, model_type, data_type)
                        soft_embedding = soft_embedding.unsqueeze(0)  # 添加批次维度
                    # 获取输入与输�?
                    # chat1 = [{"role": "user", "content": item['input']}]
                    # input_ids1 = tokenizer.apply_chat_template(
                    #     chat1, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    # ).to(model.device)

                    # 自己设计的多轮对话的prompt **************************
                    chat_str, len_chat_str = extract_prompt(
                        item['input'],
                        strip_static_kg=strip_static_kg,
                        strip_dynamic_kg=strip_dynamic_kg,
                        using_origin=using_origin,
                        model_type=model_type,
                    )

                    input_ids1 = tokenizer(chat_str, return_tensors='pt')['input_ids']
                    input_ids1 = input_ids1.to(peft_model.device)

                    ##################################

                    len_per_prompt_tokens = len(input_ids1[0])
                    output_tokens = tokenizer(item['output'], return_tensors="pt")
                    output_tokens_ids = output_tokens['input_ids'].to(model.device)
                    # 将输入和输出拼接
                    # (output_tokens_ids, torch.tensor([[128001]], device=input_ids1.device)), dim=1)
                    if model_type == "llama":
                        output_tokens_ids = torch.cat(
                            (output_tokens_ids, torch.tensor([[128009]], device=input_ids1.device)), dim=1)
                    elif model_type == "qwen1.5":
                        output_tokens_ids = torch.cat(
                            (output_tokens_ids, torch.tensor([[151643]], device=input_ids1.device)), dim=1)
                    elif model_type == "deepseek_dist_llama":
                        output_tokens_ids = torch.cat(
                            (output_tokens_ids, torch.tensor([[128001]], device=input_ids1.device)), dim=1)
                    # qwen1.5 -- 151643
                    input_ids = torch.cat((input_ids1, output_tokens_ids), dim=1)

                    padding_tokens_toLabel = torch.tensor([-100] * (len_per_prompt_tokens + 1),
                                                          device=model.device)  # +1是为了处理输入和输出拼接
                    padding_tokens_toLabel = padding_tokens_toLabel.unsqueeze(0)
                    labels = torch.cat((padding_tokens_toLabel, output_tokens_ids), dim=1)
                    # 获取硬嵌入并与软嵌入拼接
                    hard_embedding = peft_model.base_model.model.model.embed_tokens(input_ids)
                    if strip_pattern:
                        padding_tokens_toLabel = torch.tensor([-100] * len_per_prompt_tokens,
                                                              device=model.device)  # +1是为了处理输入和输出拼接
                        padding_tokens_toLabel = padding_tokens_toLabel.unsqueeze(0)
                        labels = torch.cat((padding_tokens_toLabel, output_tokens_ids), dim=1)
                        # 获取硬嵌入并与软嵌入拼接
                        hard_embedding = peft_model.base_model.model.model.embed_tokens(input_ids)
                        total_embedding = hard_embedding
                    else:
                        padding_tokens_toLabel = torch.tensor([-100] * (len_per_prompt_tokens + 1),
                                                              device=model.device)  # +1是为了处理输入和输出拼接
                        padding_tokens_toLabel = padding_tokens_toLabel.unsqueeze(0)
                        labels = torch.cat((padding_tokens_toLabel, output_tokens_ids), dim=1)
                        # 获取硬嵌入并与软嵌入拼接
                        hard_embedding = peft_model.base_model.model.model.embed_tokens(input_ids)
                        total_embedding = torch.cat([soft_embedding, hard_embedding], dim=1)
                    total_embedding = total_embedding.to(dtype)
                    # ft_data_dic.append({'embeddings': total_embedding, 'labels': labels})
                    outputs = peft_model(
                        inputs_embeds=total_embedding,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs.loss
                    total_loss += loss.item()

                    # 反向传播和优�?
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_embedding = None
                print(f"Epoch {epoch + 1} Loss: {total_loss / total_length}")

            # 有新提取的就保存下来
            if not strip_pattern:
                if base_train_soap_len < len(train_soap.keys()):
                    with open(train_soap_path, 'w', encoding='utf-8') as file_w:
                        json.dump(train_soap, file_w, ensure_ascii=False, indent=4)

            # 保存微调数据
            if not os.path.exists(os.path.join(fine_tune_path, 'model_parameters.pth')):
                if not os.path.exists(fine_tune_path):
                    os.makedirs(fine_tune_path)
                # 保存模型参数

                torch.save(peft_model.state_dict(), os.path.join(fine_tune_path, 'model_parameters.pth'))

            # torch.save(ft_data_dic, os.path.join(fine_tune_path, "total_data.pt"))


        else:
            # ft_data_dic = torch.load(os.path.join(fine_tune_path, "total_data.pt"))
            print(f"load lora path = {os.path.join(fine_tune_path, 'model_parameters.pth')}")
            # 加载保存的参�?
            # peft_model.load_state_dict(torch.load(os.path.join(fine_tune_result_path, 'model_parameters.pth')))
            model_state_dict = torch.load(os.path.join(fine_tune_path, 'model_parameters.pth'),
                                          map_location=torch.device('cpu'))
            peft_model.load_state_dict(model_state_dict)
            peft_model.to(DEVICE)  # 将模型转移到GPU

    else:
        print("un using fine tuning�?)

# 测试模块
    if is_test:
        peft_model.eval()
        if (not os.path.exists(fine_tune_path)) and (using_fine_turn):
            print("未进行微调！")
            return None
        else:
            if not os.path.isfile(hardprompt_test_path):
                ppr_sampler, entities_map, rel_map = get_ppr_sample(n_samp_ent, 10, data_type=data_type,
                                                                    save_base_path=base_save_path)
                hardprompt_test_ls = process_hardprompt_test(model, tokenizer, base_data_path, ppr_sampler,
                                                             entities_map, rel_map,
                                                             data_type, tog_depth, tog_width)
                with open(hardprompt_test_path, 'w', encoding='utf-8') as file:
                    json.dump(hardprompt_test_ls, file, ensure_ascii=False, indent=4)
            else:
                with open(hardprompt_test_path, 'r', encoding='utf-8') as file:
                    hardprompt_test_ls = json.load(file)

        if not os.path.isfile(hardprompt_test_soap_path):
            hardprompt_test_soap_ls = {}
            for item in tqdm(hardprompt_test_ls, desc="generation test dialogue soap",
                             dynamic_ncols=True):  # 这里只取�?0条数�?
                dialogue = item['history']
                dialogue_SOAP = get_soap_str(model, tokenizer, dialogue, model_type, 1, model_api)
                hardprompt_test_soap_ls.update({
                    item['fileName']: dialogue_SOAP
                })
            with open(hardprompt_test_soap_path, 'w', encoding='utf-8') as file:
                json.dump(hardprompt_test_soap_ls, file, ensure_ascii=False, indent=4)
        else:
            with open(hardprompt_test_soap_path, 'r', encoding='utf-8') as file:
                hardprompt_test_soap_ls = json.load(file)
        generate_ls = []  # 里面的元素是字典，{"labels": "医生�?喜欢吃辛辣刺激油腻食物吗�?, "predict": "医生�?�?可以查一下�?}

        if not using_fine_turn:
            # 推理时禁�?LoRA 的低秩矩�?
            # 禁用 LoRA 低秩矩阵的影�?
            for name, param in peft_model.named_parameters():
                if 'lora' in name:  # 仅影响低秩矩�?
                    param.data.zero_()  # 将低秩矩阵的参数设为零，移除影响



        for item in tqdm(hardprompt_test_ls, desc="generation test result", dynamic_ncols=True):  # 这里只取�?0条数�?
            dialogue = item['history']
            if not strip_pattern:
                if item['fileName'] in hardprompt_test_soap_ls.keys():
                    dialogue_SOAP = hardprompt_test_soap_ls[item['fileName']]
                else:
                    dialogue_SOAP = get_soap_str(model, tokenizer, dialogue, model_type, 1, model_api)
                    hardprompt_test_soap_ls.update({
                        item['fileName']: dialogue_SOAP
                    })
                soft_embedding = get_soap_softprompt(dialogue_SOAP, model, tokenizer, top_k, batch_size, hidden_size,
                                                     model_type, data_type)
                soft_embedding = soft_embedding.unsqueeze(0)  # 添加批次维度



            # 获取输入与输�?
            # chat1 = [{"role": "user", "content": item['input']}]
            # # chat1 = [{"role": "user", "content": item['input'][:-65] + '你是一名医生，你正在进行医疗对话，“医生和患者的对话历史”是你与患者的对话历史，请你对patient的问题直接进行回复，不要回复其它东西�?}] # 你是一个专业的医生，请根据前文进行回复�?
            # # chat1 = [{"role": "user", "content": item['input']}] # 你是一个专业的医生，请根据前文进行回复�?
            # input_ids1 = tokenizer.apply_chat_template(
            #     chat1, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            # ).to(model.device)
            # # 获取硬嵌入并与软嵌入拼接
            # hard_embedding = peft_model.base_model.model.model.embed_tokens(input_ids1)

            # 自己设计的多轮对话的prompt **************************
            chat_str, len_chat_str = extract_prompt(
                item['input'],
                strip_static_kg=strip_static_kg,
                strip_dynamic_kg=strip_dynamic_kg,
                using_origin=using_origin,
                model_type=model_type
            )

            input_ids1 = tokenizer(chat_str, return_tensors='pt')['input_ids']
            input_ids1 = input_ids1.to(peft_model.device)
            if model_type == "llama" or model_type == "deepseek_dist_llama":
                hard_embedding = peft_model.base_model.model.model.embed_tokens(input_ids1)
            elif model_type == "qwen1.5":
                hard_embedding = model.model.base_model.embed_tokens(input_ids1)

            # ***************测试only-hardprompt***************
            if strip_pattern:
                total_embedding = hard_embedding
            else:
                total_embedding = torch.cat([soft_embedding, hard_embedding], dim=1)

            # 确保输入数据和模型的权重都是 float16 类型
            total_embedding = total_embedding.to(dtype)
            # total_embedding = hard_embedding  # 这里测试一下去掉softprompt的效�?

            peft_model = peft_model.to(dtype)  # 或者根据硬件支持的 dtype 修改�?float16
            # 生成模型输出
            outputs = model.generate(
                inputs_embeds=total_embedding,
                max_new_tokens=2048,  # 调整生成长度
                temperature=generate_configuration['temperature'],  # 增加生成多样�?
                top_p=generate_configuration['top_p'],
                do_sample=True,  # 使用采样生成
                early_stopping=True,
                return_dict=True,
            )
            print(f"result：{tokenizer.decode(outputs[0], skip_special_tokens=True)}")


            generate_ls.append({
                "labels": "医生�?" + item['output'],
                # "predict": "医生�?" + tokenizer.decode(outputs[0], skip_special_tokens=True).replace('"', '\\"')
                "predict": "医生�?" + tokenizer.decode(outputs[0], skip_special_tokens=True).split('</think>')[-1].replace('"', '\\"')
                # "predict": "医生�?" + tokenizer.decode(outputs[0], skip_special_tokens=True)
            })

        if using_origin and not using_fine_turn:
            name = "origin"
        elif using_fine_turn:
            name = "fine_turn"
        else:
            name = "all"
        if not os.path.exists(f"../result/{model_type}/{name}"):
            os.makedirs(f"../result/{model_type}/{name}")
        with open(f"../result/{model_type}/{name}/{data_type}_generated_predictions.json", 'w', encoding='utf-8') as file_generation:
            # 打开文件并写入数�?
            for item in generate_ls:
                # 将每个JSON对象转换为字符串，并写入文件item["predict"].replace("\n", "\\n")
                label_str = item["labels"].replace('"', '\\"')
                label_str = label_str.replace("\n", "")
                predict_str = item["predict"].replace("\n", "")
                predict_str = predict_str.replace('"', '\\"')
                line = f'{{"labels": "{label_str}", "predict": "{predict_str}"}}'
                file_generation.write(line + '\n')  # 每个对象后添加换行符

        print(f'Data has been written to result/{model_type}/{name}/{data_type}_generated_predictions.json')

if __name__ == "__main__":
    main()
