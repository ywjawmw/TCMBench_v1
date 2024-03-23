# --*-- conding:utf-8 --*--
# @File : Evaluate.py
# @Software : PyCharm
# @Description :  使用不同的指标对analysis和response进行评估
import math
import sys

import evaluate
import json
import statistics
from typing import List, Dict
import numpy as np
import torch
from BARTScore.bart_score import BARTScorer
from bert_score import score
from sari import SARIsent
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer
)
import jieba
import tqdm
gpu_id = 3
print("CUDA:", gpu_id)
jieba.load_userdict("jieba_tcm.txt")    # load进入TCM术语
with open('jieba_tcm.txt', 'r', encoding='utf-8') as f:
  tcm_terms = f.readlines()
  tcm_term_db = [t.strip() for t in tcm_terms]

def read_file(model_name):
    if len(model_name) > 0 and "query" not in model_name:
        file = f'./model_test/human_test/{model_name}_factest.json'
    elif model_name == "query":
        file = './model_test/search_querys.json'
    elif model_name == "query_bing":
        file = './model_test/search_querys-bing.json'
    elif "bing_search" in model_name:
        file = '../Evaluate_result/bing_search_bing搜索结果.json'
    else:
        file = './tmnli_search_test/TCM_Q2D_QAC-test.json'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data

# Rouge-L
def rouge_score(model_name):
    rouge = evaluate.load('/data2/.../Evaluate/metrics/rouge')
    test_data = read_file(model_name)['example']
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    results = rouge.compute(predictions=predictions,
                            references=references,
                            rouge_types=['rouge1', 'rougeL'],
                            use_aggregator=False)
    # print("rouge1：", results["rouge1"])
    # print("rougeL:", results["rougeL"])
    print("rouge1：", statistics.mean(results["rouge1"]))
    print("rougeL:", statistics.mean(results["rougeL"]))
    return statistics.mean(results["rouge1"]), statistics.mean(results["rougeL"])
# BertScore
def bert_score(model_name):
    bertscore = evaluate.load('/data2/.../Evaluate/metrics/bertscore')
    test_data = read_file(model_name)['example']
    # query_list = read_file("")['example']
    # querys = [query_list[i]["question"] for i in range(len(query_list))]
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    # results = bertscore.compute(predictions=predictions, references=references, lang="zh", model_type="bert-base-chinese")
    # score = statistics.mean([round(v, 2) for v in results["f1"]])
    # print(score)

    # search result : 看query和解析之间的指标作为基线，检索出来的指标应该与解析之间的指标相差较小

    # results = bertscore.compute(predictions=predictions, references=querys, lang="zh",
    #                             model_type="bert-base-chinese")
    # score = statistics.mean([round(v, 2) for v in results["f1"]])
    # print(f"检索结果指标: {score}")

    results = bertscore.compute(predictions=predictions, references=references, lang="zh",
                                model_type="bert-base-chinese", device="cuda:2")
    score = statistics.mean([round(v, 2) for v in results["f1"]])
    print(f"Bert Score 标准解析结果指标: {score}")
    # P, R, F1 = score(predictions, references, model_type="bert-base-chinese",lang="zh", verbose=True)
    # print(P)
    # print(R)
    # print(F1)
    return score

# BartScore
def bart_Score(model_name):
    test_data = read_file(model_name)['example']
    query_list = read_file("")['example']
    # querys = [query_list[i]["question"] for i in range(len(query_list))]
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    bart_scorer = BARTScorer(device='cuda:2', checkpoint='/data2/.../ZeroFEC-master/evals/bart-lage-cnn')
    bart_scores = bart_scorer.score(references, predictions, batch_size=4)
    print("BART Score", np.mean(bart_scores))
    return np.mean(bart_scores)

    # search result : 看query和解析之间的指标作为基线，检索出来的指标应该与解析之间的指标相差较小
    #
    # bart_scores = bart_scorer.score(querys, predictions, batch_size=4)
    # print(f"检索结果指标BART Score: {np.mean(bart_scores)}")

    # bart_scores = bart_scorer.score(querys, references, batch_size=4)
    # print(f"标准解析结果指标BART Score: {np.mean(bart_scores)}")


# FactCC

def get_factcc_score(predictions, evidences):
    model = BertForSequenceClassification.from_pretrained('/data2/..../ZeroFEC-master/evals/factcc').cuda()
    tokenizer = BertTokenizer.from_pretrained('/data2/.../ZeroFEC-master/evals/bert-base-uncased')
    assert len(predictions) == len(evidences)
    res = []
    for prediction, evidence in zip(predictions, evidences):
        # dynamically determine how much input to use
        encoded_ctx = tokenizer.encode(evidence)[:-1]  # remove [SEP]
        encoded_prediction = tokenizer.encode(prediction)[1:]  # remove [CLS]
        if len(encoded_prediction) > 512:
            encoded_prediction = encoded_prediction[:256]
            encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_prediction)]  # - [SEP] - encoded_correction

            # print(tokenizer.decode(encoded_ctx_truncated))

            input_ids = torch.LongTensor(
                encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_prediction).cuda().unsqueeze(0)
            token_type_ids = torch.LongTensor(
                [0] * (len(encoded_ctx_truncated) + 1) + [1] * len(encoded_prediction)).cuda().unsqueeze(0)
            attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)

            inputs = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask}

            with torch.no_grad():
                model.eval()
                outputs = model(**inputs)
                logits = outputs[0]
                probs = torch.nn.Softmax(dim=1)(logits)
                correct_prob = probs[0][0].item()
                res.append(correct_prob)

            encoded_prediction = encoded_prediction[256:]
            encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_prediction)]  # - [SEP] - encoded_correction

            # print(tokenizer.decode(encoded_ctx_truncated))

            input_ids = torch.LongTensor(
                encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_prediction).cuda().unsqueeze(0)
            token_type_ids = torch.LongTensor(
                [0] * (len(encoded_ctx_truncated) + 1) + [1] * len(encoded_prediction)).cuda().unsqueeze(0)
            attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)

            inputs = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask}

            with torch.no_grad():
                model.eval()
                outputs = model(**inputs)
                logits = outputs[0]
                probs = torch.nn.Softmax(dim=1)(logits)
                correct_prob = probs[0][0].item()
                res.append(correct_prob)

        else:
            encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_prediction)]  # - [SEP] - encoded_correction

            # print(tokenizer.decode(encoded_ctx_truncated))

            input_ids = torch.LongTensor(
                encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_prediction).cuda().unsqueeze(0)
            token_type_ids = torch.LongTensor(
                [0] * (len(encoded_ctx_truncated) + 1) + [1] * len(encoded_prediction)).cuda().unsqueeze(0)
            attention_mask = torch.LongTensor([1] * len(input_ids)).cuda().unsqueeze(0)

            inputs = {'input_ids': input_ids,
                      'token_type_ids': token_type_ids,
                      'attention_mask': attention_mask}

            with torch.no_grad():
                model.eval()
                outputs = model(**inputs)
                logits = outputs[0]
                probs = torch.nn.Softmax(dim=1)(logits)
                correct_prob = probs[0][0].item()
                res.append(correct_prob)

    return res

def factcc_score(model_name):
    test_data = read_file(model_name)['example']
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    factcc_scores = get_factcc_score(predictions, references)
    print("FactCC Score", np.mean(factcc_scores))

# SARI
def calculate_sari(
        input_lns: List[str], output_lns: List[str], reference_lns: List[str]
) -> Dict:
    a, b, c, d = [], [], [], []
    for input, output, ref in zip(input_lns, output_lns, reference_lns):
        a_, b_, c_, d_ = SARIsent(input, output, [ref])

        a.append(a_)
        b.append(b_)
        c.append(c_)
        d.append(d_)
    return np.mean(a)

    # return {
    #     "sari_avgkeepscore": np.mean(a),
    #     "sari_avgdelscore": np.mean(b),
    #     "sari_avgaddscore": np.mean(c),
    #     "sari_finalscore": np.mean(d),
    # }

def sari_score(model_name):
    test_data = read_file(model_name)['example']
    query_list = read_file("query_bing")['example']
    querys = [query_list[i]["question"] for i in range(len(query_list))]
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    # sariscores = calculate_sari(querys, predictions, querys)   # 参考答案根据reference
    # print(f"SARI score_检索回答: {sariscores}")
    sariscores = calculate_sari(references, predictions, references)  # 参考答案根据reference  只看sari_avgkeepscore
    print(f"SARI score_解析: {sariscores}")
    return sariscores

# TNLI

def calculate_score(sentence, sample, model, tokenizer):
    inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[(sentence, sample)],
        add_special_tokens=True, padding="longest",
        truncation=True, return_tensors="pt",
        return_token_type_ids=True, return_attention_mask=True,
        max_length=512
    ).to(f"cuda:{gpu_id}")

    logits = model(**inputs).logits  # neutral is already removed
    # print(logits)
    probs = torch.argmax(logits, dim=-1)
    prob_label = probs[0].item()  # 类别
    probs1 = torch.softmax(logits, dim=-1)
    prob_1 = probs1[0][0].item()  # prob(相关程度)
    return prob_label, prob_1

import re
def split_sentences(text):
    # 利用正则表达式按照句号、感叹号、问号进行划分
    sentences = re.split(r'(?<=[。！？])\s*', text)
    # 去除空字符串和空白符
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

from collections import Counter
def tcm_score_f1(analysis_tcm_terms_counter, analysis_terms_counter, doc, tcm_term_db):
    """
    中医术语匹配度
    :param analysis_tcm_terms_counter: 解析中的中医术语以及计数
    :param analysis_terms_counter: 解析
    :param doc:  需要检测的语句
    :param tcm_term_db:  中医术语库
    :return:
    """

    doc_terms_list = list(jieba.cut(doc))
    doc_terms_counter = Counter(doc_terms_list)
    doc_tcm_terms_list = [term for term in doc_terms_list if term in tcm_term_db]
    doc_tcm_terms_counter = Counter(doc_tcm_terms_list)  # 片段中所有的中医术语计数
    if len(analysis_tcm_terms_counter) == 0:
        return 2   # 如果解析中没有中医术语，那么就不需要进行F1score的计算
    elif len(doc_tcm_terms_counter) == 0:
        return 0   # # 如果LLMs中没有中医术语，那么F1score=0, 说明这句话不符合中医诊疗语言，或者没有什么信息量
    comment_term_counter = analysis_tcm_terms_counter & doc_tcm_terms_counter  # 重复的中医术语
    recall_comment_score, precision_comment_score = 0, 0
    for term in comment_term_counter:
        recall_comment_score += comment_term_counter[term] / analysis_tcm_terms_counter[term]
        precision_comment_score += comment_term_counter[term] / doc_tcm_terms_counter[term]
    recall = recall_comment_score / len(analysis_tcm_terms_counter)   # 重复的中医术语个数/解析中的中医术语个数 —— 重叠度
    precision = precision_comment_score / len(doc_tcm_terms_counter)   # 重复的中医术语个数/ 文档的中医术语个数 —— 冗余度
    informational = len(list(set(doc_tcm_terms_list))) / len(list(set(doc_terms_list)))
    # informational = len(doc_tcm_terms_counter) / len(doc_terms_counter) * (sum(doc_terms_counter.values()) / sum(analysis_terms_counter.values()))  # 片段中的中医术语 / 片段中术语个数（不重复） * (片段的术语总个数 / 解析的术语总个数)[长度的惩罚项] —— 信息度
    if precision == 0 or recall == 0:
        return 0
    else:
        f1_score = 3 * (precision * recall * informational) / (precision + recall + informational)
        return f1_score
print("informational 使用len(list(set(doc_tcm_terms_list))) / len(list(set(doc_terms_list)))，不加入惩罚项")
def f1_score_tcm_term(analysis_tcm_terms_list, analysis_terms_list, doc, tcm_term_db):
    """
    中医术语匹配度
    :param analysis_tcm_terms_list: 解析中的中医术语
    :param analysis_terms_list: 解析
    :param doc:  需要检测的语句
    :param tcm_term_db:  中医术语库
    :return:
    """

    doc_terms_list = list(jieba.cut(doc))
    doc_tcm_terms_list = [term for term in doc_terms_list if term in tcm_term_db]
    doc_tcm_terms_list = doc_tcm_terms_list  # 片段中所有的中医术语(含有重复元素）
    if len(analysis_tcm_terms_list) == 0:
        return 2   # 如果解析中没有中医术语，那么就不需要进行F1score的计算
    elif len(doc_tcm_terms_list) == 0:
        return 0   # # 如果LLMs中没有中医术语，那么F1score=0, 说明这句话不符合中医诊疗语言，或者没有什么信息量
    comment_term = set(analysis_tcm_terms_list) & set(doc_terms_list)  # 重复的中医术语
    comment_num = 0
    # diversity = 0
    for doc_term in doc_tcm_terms_list:
        if doc_term in comment_term:
            comment_num += 1
        # else:
        #     diversity += 1   # 除解析之外的中医术语，如果有，有可能是解析中还有额外的中药信息量
    recall = comment_num / len(analysis_tcm_terms_list)   # 重复的中医术语个数/真正对的中医术语个数 —— 重叠度
    precision = comment_num / len(doc_tcm_terms_list)   # 重复的中医术语个数/ 文档的中医术语个数 —— 冗余度
    # informational = len(list(set(doc_tcm_terms_list))) / len(list(set(doc_terms_list))) * (len(doc_terms_list) / len(analysis_terms_list))  # 片段中的中医术语 / 片段中总术语个数 * (片段的长度 / 解析的长度)[长度的惩罚项] —— 信息度
    informational = len(list(set(doc_tcm_terms_list))) / len(list(set(doc_terms_list)))
    if precision == 0 or recall == 0 or informational == 0:
        return 0
    else:
        f1_score = 3 * (precision * recall * informational) / (precision + recall + informational)
        # f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

# print("使用 softmax，缩放f1的值")
print("使用 x / sum_x，缩放f1的值")
def softmax(x):
    e_x = x
    sum_x = e_x.sum(axis=0)
    if sum_x == 2 * x.size:  # 标准解析中就没有中医术语，那么解析中的每一句话的f1score都=2
        return [1/x.size] * x.size  # 加和平均
    elif sum_x == 0:
        return [0] * x.size    # 也就是LLMs中没有中医术语，那么这个解析是不好的，给一个特别低的分数
    else:
        return x / sum_x
        # exp_x = np.exp(x)
        # # 计算softmax
        # s = exp_x / np.sum(exp_x, axis=0)
        # return s

# def nli_response_analysys(sentences: List[str], sampled_passages: List[str], model, tokenizer):
#     """
#     :param sentences: list of 标准解析
#     :param sampled_passages: LLMs的解析
#     """
#     scores1 = list()  # 计算LLMs生成的解析与标准解析之间的分数
#     scores1_counter = list()  # _counter
#     num = 0
#     for sentence, sample in zip(sentences, sampled_passages):  # 解析
#         if num == 0:
#             print(f"sentence: {sentence}")
#             print(f"sample: {sample}")
#         num += 1
#         # 分句
#         response_sentence_list = split_sentences(sample)
#         # analysis_sentence_list = split_sentences(sentence)
#         tcm_score = []
#         tcm_score_counter = []
#         if len(response_sentence_list) > 0:
#             for response_sentence in response_sentence_list:  # 分句
#                 f1_score, prob_score = [], []
#                 f1_score_counter = []
#                 for analysis_sentence in [sentence]:  # LLMs分析
#                     # 统计标准解析中的中医术语
#                     analysis_terms = list(jieba.cut(analysis_sentence))
#                     analysis_terms_counter = Counter(analysis_terms)
#                     analysis_tcm_terms_list = [term for term in analysis_terms if term in tcm_term_db]
#                     analysis_tcm_terms_counter = Counter(analysis_tcm_terms_list)  # 解析中的中医术语计数
#                     analysis_tcm_terms_list = list(analysis_tcm_terms_counter.keys())  # 解析中的中医术语列表
#
#                     prob_label_A, prob_1_A = calculate_score(analysis_sentence, response_sentence, model, tokenizer)
#                     prob_label_reverse_A, prob_1_reserve_A = calculate_score(response_sentence, analysis_sentence,
#                                                                              model, tokenizer)
#                     prob = (prob_1_A + prob_1_reserve_A) / 2
#                     ################## TCM score ####################
#                     f1_term_score = f1_score_tcm_term(analysis_tcm_terms_list, analysis_terms, response_sentence,
#                                                       tcm_term_db)
#                     f1_term_score_counter = tcm_score_f1(analysis_tcm_terms_counter, analysis_terms_counter,
#                                                          response_sentence, tcm_term_db)
#                     f1_score.append(f1_term_score)
#                     prob_score.append(prob)
#                     f1_score_counter.append(f1_term_score_counter)
#                 f1_score = np.array(f1_score)
#                 prob_score = np.array(prob_score)
#                 f1_score_counter = np.array(f1_score_counter)
#                 # 对列表进行归一化
#                 try:
#                     normalized_f1_score = softmax(f1_score)
#                     normalized_f1_score_counter = softmax(f1_score_counter)
#                 except:
#                     print(response_sentence_list)
#                     print(f1_score)
#                     sys.exit()
#                 # 计算相乘并相加的结果，加权平均
#                 analysis_sentence_score = np.sum(normalized_f1_score * prob_score)
#                 tcm_score.append(analysis_sentence_score)
#
#                 # 用Counter计算的score
#                 analysis_sentence_score_counter = np.sum(normalized_f1_score_counter * prob_score)
#                 tcm_score_counter.append(analysis_sentence_score_counter)
#         else:
#             tcm_score.append(0)
#             tcm_score_counter.append(0)
#         scores_per_response = statistics.mean(tcm_score)
#         scores1.append(scores_per_response)
#         # 用Counter计算的score
#         scores_per_response_counter = statistics.mean(tcm_score_counter)
#         scores1_counter.append(scores_per_response_counter)
#     scores_per_doc = statistics.mean(scores1)
#     scores_per_doc_counter = statistics.mean(scores1_counter)
#     print("回答与解析之间的TCM Score:", scores_per_doc)
#     print("回答与解析之间的TCM Score，用Counter计算:", scores_per_doc_counter)


def predict_analysys_response(sentences: List[str], sampled_passages: List[str], model, tokenizer):
    """
    :param sentences: list of 标准解析
    :param sampled_passages: LLMs的解析
    """
    scores1 = list()  # 计算LLMs生成的解析与标准解析之间的分数
    scores1_counter = list()  # _counter
    scores1_nof1 = list()
    scores1_dotf1 = list()
    num = 0
    for sentence, sample in zip(sentences, sampled_passages):  # 解析
        if num == 0:
            print(f"sentence: {sentence}")
            print(f"sample: {sample}")
        num += 1
        #########################################################
        # # 不分句
        # analysis_tcm_terms = list(jieba.cut(sentence))
        # analysis_tcm_terms_list = [term for term in analysis_tcm_terms if term in tcm_term_db]
        # analysis_tcm_terms_list = list(set(analysis_tcm_terms_list))  # 解析中的中医术语列表
        # prob_label_A, prob_1_A = calculate_score(sentence, sample, model, tokenizer)
        # prob_label_reverse_A, prob_1_reserve_A = calculate_score(sample, sentence,
        #                                                          model, tokenizer)
        # prob = (prob_1_A + prob_1_reserve_A) / 2
        # f1_term_score = f1_score_tcm_term(analysis_tcm_terms_list, sample, tcm_term_db)
        # scores1.append(f1_term_score * prob)

        ############################################################
        # 分句
        response_sentence_list = split_sentences(sample)
        analysis_sentence_list = split_sentences(sentence)
        tcm_score = []
        tcm_score_counter = []
        prob_score_list = []
        tcm_score_dotf1 = []
        for analysis_sentence in analysis_sentence_list:  # 分句
            f1_score, prob_score = [], []
            f1_score_counter = []
            if len(response_sentence_list) > 0:
                for response_sentence in response_sentence_list:  # LLMs分析分句
                    # 统计标准解析中的中医术语
                    analysis_terms = list(jieba.cut(analysis_sentence))
                    analysis_terms_counter = Counter(analysis_terms)
                    analysis_tcm_terms_list = [term for term in analysis_terms if term in tcm_term_db]
                    analysis_tcm_terms_counter = Counter(analysis_tcm_terms_list)  # 解析中的中医术语计数
                    analysis_tcm_terms_list = list(analysis_tcm_terms_counter.keys())  # 解析中的中医术语列表

                    prob_label_A, prob_1_A = calculate_score(analysis_sentence, response_sentence, model, tokenizer)
                    prob_label_reverse_A, prob_1_reserve_A = calculate_score(response_sentence, analysis_sentence,
                                                                             model, tokenizer)
                    prob = (prob_1_A + prob_1_reserve_A) / 2
                    ################## TCM score ####################
                    f1_term_score = f1_score_tcm_term(analysis_tcm_terms_list, analysis_terms, response_sentence, tcm_term_db)
                    f1_term_score_counter = tcm_score_f1(analysis_tcm_terms_counter, analysis_terms_counter, response_sentence, tcm_term_db)
                    f1_score.append(f1_term_score)
                    prob_score.append(prob)
                    f1_score_counter.append(f1_term_score_counter)
                f1_score = np.array(f1_score)

                ####不加f1 score#######
                average_prob_score = statistics.mean(prob_score)
                prob_score_list.append(average_prob_score)

                prob_score = np.array(prob_score)
                f1_score_counter = np.array(f1_score_counter)
                # 对列表进行归一化
                try:
                    normalized_f1_score = softmax(f1_score)
                    normalized_f1_score_counter = softmax(f1_score_counter)
                except:
                    print(response_sentence_list)
                    print(f1_score)
                    sys.exit()

                #####直接与F1 score相乘#####
                analysis_sentence_dotf1 = np.sum(f1_score * prob_score) / prob_score.size
                tcm_score_dotf1.append(analysis_sentence_dotf1)
                ###### 计算相乘并相加的结果，加权平均
                analysis_sentence_score = np.sum(normalized_f1_score * prob_score)
                tcm_score.append(analysis_sentence_score)
                # 用Counter计算的score
                analysis_sentence_score_counter = np.sum(normalized_f1_score_counter * prob_score)
                tcm_score_counter.append(analysis_sentence_score_counter)
            else:
                tcm_score.append(0)
                tcm_score_counter.append(0)
                tcm_score_dotf1.append(0)
                prob_score_list.append(0)
        # 长度惩罚项：1 - max(0, 差z) / 解析
        # length_penalty = math.exp(1 / math.exp(abs(len(sentence) - len(sample)) / len(sentence)) - 1)
        if len(sample) < len(sentence):
            if len(sample) > 1:
                length_penalty = math.exp(1 - math.log(len(sentence)) / math.log(len(sample)))
            elif len(sample) == 1:
                length_penalty = math.exp(1 - math.log(len(sentence)) / math.log(len(sample) + 1))
            else:
                length_penalty = 0
        else:
            # length_penalty = 1
            length_penalty = math.exp(1 - math.log(len(sample)) / math.log(len(sentence)))
        scores_per_response = statistics.mean(tcm_score) * length_penalty
        scores1.append(scores_per_response)
        # 用Counter计算的score
        scores_per_response_counter = statistics.mean(tcm_score_counter) * length_penalty
        scores1_counter.append(scores_per_response_counter)
        ############不加f1 score
        scores1_nof1.append(statistics.mean(prob_score_list))
        #####直接与F1 score相乘#####
        scores_per_response_dotf1 = statistics.mean(tcm_score_dotf1)
        scores1_dotf1.append(scores_per_response_dotf1)
    scores_per_doc = statistics.mean(scores1)
    scores_per_doc_counter = statistics.mean(scores1_counter)
    # print("解析与回答之间的TCM Score:", scores_per_doc)
    print("解析与回答之间的TCM Score，用Counter计算:", scores_per_doc_counter)
    # print("解析与回答之间的NLI Score，不加f1", statistics.mean(scores1_nof1))
    # print("解析与回答之间的NLI Score，直接与f1相乘", statistics.mean(scores1_dotf1))
    return scores_per_doc_counter


def predict(querys: List[str], sentences: List[str], sampled_passages: List[str], model, tokenizer, query_flag=False):
    num_sentences = len(sentences)
    num_samples = len(sampled_passages)
    document_true = 0
    tmnli_acc_num = 0
    tmnli_response_acc_num = 0
    scores1 = list()   # 计算LLMs生成的解析与标准解析之间的分数
    acc_response_score1, acc_score1 = list(), list()   # 分别计算LLMs生成的解析与QWA之间的分数、标准解析与QWA之间的分数
    acc_response_score1_counter, acc_score1_counter = list(), list()  # _counter分别计算LLMs生成的解析与QWA之间的分数、标准解析与QWA之间的分数
    if query_flag is False:  # 不考虑query判断语义相似性
        print(num_samples == num_samples == len(querys))
        for query_i, query in enumerate(querys):
            # 统计标准解析中的中医术语
            query_terms = list(jieba.cut(query))
            query_terms_counter = Counter(query_terms)
            query_tcm_terms_list = [term for term in query_terms if term in tcm_term_db]
            query_tcm_terms_counter = Counter(query_tcm_terms_list)
            query_tcm_terms_list = list(query_tcm_terms_counter.keys())  # 解析中的中医术语列表
            for sent_i, sentence in enumerate(sentences):   # 解析
                for sample_i, sample in enumerate(sampled_passages):
                    if query_i == sent_i == sample_i:
                        if query_i == 0:
                            print(f"sentence: {sentence}")
                            print(f"sample: {sample}")
                        ############################################################
                        # 分句
                        response_sentence_list = split_sentences(sample)
                        analysis_sentence_list = split_sentences(sentence)
                        # response_sentence_score, analysis_sentence_score = 0.0, 0.0
                        response_f1_score, analysis_f1_score = [], []
                        response_prob_score, analysis_prob_score = [], []
                        # counter
                        response_f1_score_counter, analysis_f1_score_counter = [], []
                        for analysis_sentence in analysis_sentence_list:  # 分句

                            # # 统计标准解析中的中医术语
                            # analysis_tcm_terms = list(jieba.cut(analysis_sentence))
                            # analysis_tcm_terms_list = [term for term in analysis_tcm_terms if term in tcm_term_db]
                            # analysis_tcm_terms_list = list(set(analysis_tcm_terms_list))  # 解析中的中医术语列表

                            prob_label_A, prob_1_A = calculate_score(query, analysis_sentence, model, tokenizer)
                            prob_label_reverse_A, prob_1_reserve_A = calculate_score(analysis_sentence, query, model,
                                                                                     tokenizer)
                            analysis_prob = (prob_1_A + prob_1_reserve_A) / 2
                            # # 加和平均
                            # analysis_sentence_score += analysis_prob
                            # 加权平均
                            f1_term_score_analysis = f1_score_tcm_term(query_tcm_terms_list, query_terms, analysis_sentence, tcm_term_db)
                            analysis_f1_score.append(f1_term_score_analysis)
                            analysis_prob_score.append(analysis_prob)

                            f1_term_score_analysis_counter = tcm_score_f1(query_tcm_terms_counter, query_terms_counter, analysis_sentence, tcm_term_db)
                            analysis_f1_score_counter.append(f1_term_score_analysis_counter)

                        analysis_f1_score = np.array(analysis_f1_score)
                        analysis_prob_score = np.array(analysis_prob_score)

                        analysis_f1_score_counter = np.array(analysis_f1_score_counter)

                        # 对列表进行归一化
                        normalized_analysis_f1_score = softmax(analysis_f1_score)
                        normalized_analysis_f1_score_counter = softmax(analysis_f1_score_counter)

                        # 长度惩罚项：1 - max(0, 差) / 解析
                        # length_penalty = math.exp(1 / math.exp(abs(len(sentence) - len(query)) / len(query)) - 1)
                        if len(sentence) < len(query):
                            length_penalty = math.exp(1 - math.log(len(query)) / math.log(len(sentence)))
                        else:
                            # length_penalty = 1
                            length_penalty = math.exp(1 - math.log(len(sentence)) / math.log(len(query)))

                        # 计算相乘并相加的结果
                        analysis_sentence_score = np.sum(normalized_analysis_f1_score * analysis_prob_score) * length_penalty
                        analysis_sentence_score_counter = np.sum(normalized_analysis_f1_score_counter * analysis_prob_score) * length_penalty
                        # analysis_sentence_score = analysis_sentence_score / len(analysis_sentence_list)
                        for response_sentence in response_sentence_list:  # 分句

                            # # 统计LLMs解析中的中医术语
                            # response_tcm_terms = list(jieba.cut(response_sentence))
                            # response_tcm_terms_list = [term for term in response_tcm_terms if term in tcm_term_db]
                            # response_tcm_terms_list = list(set(response_tcm_terms_list))  # 解析中的中医术语列表

                            prob_label_B, prob_1_B = calculate_score(query, response_sentence, model, tokenizer)
                            prob_label_reverse_B, prob_1_reserve_B = calculate_score(response_sentence, query, model, tokenizer)
                            response_prob = (prob_1_B + prob_1_reserve_B) / 2
                            # # 加和平均
                            # response_sentence_score += response_prob
                            # 加权平均
                            f1_term_score_response = f1_score_tcm_term(query_tcm_terms_list, query_terms, response_sentence, tcm_term_db)
                            response_f1_score.append(f1_term_score_response)
                            response_prob_score.append(response_prob)
                            # counter
                            f1_term_score_response_counter = tcm_score_f1(query_tcm_terms_counter, query_terms_counter, response_sentence, tcm_term_db)
                            response_f1_score_counter.append(f1_term_score_response_counter)
                        response_f1_score = np.array(response_f1_score)
                        response_prob_score = np.array(response_prob_score)
                        # counter
                        response_f1_score_counter = np.array(response_f1_score_counter)
                        # 对列表进行归一化
                        normalized_response_f1_score = softmax(response_f1_score)
                        normalized_response_f1_score_counter = softmax(response_f1_score_counter)
                        # 长度惩罚项：1 - max(0, 差) / 解析
                        # length_penalty = math.exp(1 / math.exp(abs(len(sample) - len(query)) / len(query)) - 1)
                        if len(sample) < len(query):
                            length_penalty = math.exp(1 - math.log(len(query)) / math.log(len(sample)))
                        else:
                            # length_penalty = 1   # 只惩罚短的
                            length_penalty = math.exp(1 - math.log(len(sample)) / math.log(len(query)))
                        # 计算相乘并相加的结果
                        response_sentence_score = np.sum(normalized_response_f1_score * response_prob_score) * length_penalty
                        response_sentence_score_counter = np.sum(normalized_response_f1_score_counter * response_prob_score) * length_penalty

                        # response_sentence_score = response_sentence_score / len(response_sentence_list)
                        acc_score1.append(analysis_sentence_score)  # 标准解析
                        acc_response_score1.append(response_sentence_score)  # LLMs解析

                        # counter
                        acc_score1_counter.append(analysis_sentence_score_counter)  # 标准解析_counter
                        acc_response_score1_counter.append(response_sentence_score_counter)  # LLMs解析_counter

                        #######################################################################
                        # 不分句
                        # prob_label_A, prob_1_A = calculate_score(query, sentence, model, tokenizer)
                        # prob_label_reverse_A, prob_1_reserve_A = calculate_score(sentence, query, model,
                        #                                                          tokenizer)
                        # prob_label_B, prob_1_B = calculate_score(query, sample, model, tokenizer)
                        # prob_label_reverse_B, prob_1_reserve_B = calculate_score(sample, query, model,
                        #                                                          tokenizer)
                        # # 与QWA之间的计算
                        # acc_score1.append((prob_1_A + prob_1_reserve_A) / 2)  # 标准解析
                        # acc_response_score1.append((prob_1_B + prob_1_reserve_B) / 2)  # LLMs解析
                        # if prob_label_A == 0 and prob_label_reverse_A == 0:
                        #     tmnli_acc_num += 1
                        # if prob_label_B == 0 and prob_label_reverse_B == 0:
                        #     tmnli_response_acc_num += 1

                        # 标准解析与LLMs解析之间的计算
                        prob_label, prob_1 = calculate_score(sentence, sample, model, tokenizer)
                        prob_label_reverse, prob_1_reserve = calculate_score(sample, sentence, model, tokenizer)
                        if prob_label == 0 and prob_label_reverse == 0:
                            document_true += 1
                        scores1.append((prob_1 + prob_1_reserve) / 2)
    else:  # 若query与A相关且query与B相关，那么A与B相关---暂时不使用
        print(num_samples == num_samples==len(querys))
        for query_i, query in enumerate(querys):
            for sent_i, sentence in enumerate(sentences):
                for sample_i, sample in enumerate(sampled_passages):
                    if query_i == sent_i == sample_i:
                        prob_label_A, prob_1_A = calculate_score(query, sentence, model, tokenizer)
                        prob_label_reverse_A, prob_1_reserve_A = calculate_score(sentence, query, model, tokenizer)
                        prob_label_B, prob_1_B = calculate_score(query, sample, model, tokenizer)
                        prob_label_reverse_B, prob_1_reserve_B = calculate_score(sample, query, model, tokenizer)
                        if prob_label_A == 0 and prob_label_reverse_A == 0:
                            tmnli_acc_num += 1
                        if prob_label_B == 0 and prob_label_reverse_B == 0:
                            tmnli_response_acc_num += 1
                        if prob_label_A == 0 and prob_label_reverse_A == 0 and prob_label_B == 0 and prob_label_reverse_B == 0:
                            document_true += 1
                        ref_score = (prob_1_B + prob_1_reserve_B) / 2
                        if prob_label_A != 0 or prob_label_reverse_A != 0:
                            prob_label, prob_1 = calculate_score(sentence, sample, model, tokenizer)
                            prob_label_reverse, prob_1_reserve = calculate_score(sample, sentence, model, tokenizer)
                            if prob_label == 0 and prob_label_reverse == 0:
                                document_true += 1
                                ref_score = (prob_1 + prob_1_reserve) / 2
                        scores1.append(ref_score)
    print(scores1)
    scores_per_doc = statistics.mean(scores1)
    print("NLI score:", scores_per_doc)
    print("NLI relevant true:", document_true/num_sentences)   # 用标签进行计算，推理出相关的解析占总测试集的比率

    analysis_scores_per_doc = statistics.mean(acc_score1)
    print("解析与QWA之间的相关性分数:", analysis_scores_per_doc)
    response_scores_per_doc = statistics.mean(acc_response_score1)
    print("LLMs生成的解析与QWA之间的相关性分数:", response_scores_per_doc)

    # counter
    analysis_scores_per_doc_counter = statistics.mean(acc_score1_counter)
    print("Counter: 解析与QWA之间的相关性分数:", analysis_scores_per_doc_counter)
    response_scores_per_doc_counter = statistics.mean(acc_response_score1_counter)
    print("Counter: LLMs生成的解析与QWA之间的相关性分数:", response_scores_per_doc_counter)

    # print("解析的 acc in testdata:", tmnli_acc_num / num_sentences)
    # print("LLMs生成的解析的 acc in testdata:", tmnli_response_acc_num / num_sentences)
    # return scores_per_doc, document_true/num_sentences

def nli_score(model_name, model, tokenizer):
    test_data = read_file(model_name)['example']
    query_list = read_file("")['example']
    # querys = [query_list[i]["QWA"] for i in range(len(query_list))]
    # querys = []
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    # predict(querys, sentences=references,  sampled_passages=predictions, model=model, tokenizer=tokenizer)
    n_score = predict_analysys_response(sentences=references,  sampled_passages=predictions, model=model, tokenizer=tokenizer)
    # print("解析不分句的情况")
    # nli_response_analysys(sentences=references, sampled_passages=predictions, model=model, tokenizer=tokenizer)
    return n_score


def nli_score_query(model_name, model, tokenizer):
    test_data = read_file(model_name)['example']
    query_list = read_file("")['example']
    querys = [query_list[i]["QWA"] for i in range(len(query_list))]
    # querys = []
    references = [test_data[i]["analysis"] for i in range(len(test_data))]
    predictions = [test_data[i]["response"] for i in range(len(test_data))]
    predict(querys, sentences=references,  sampled_passages=predictions, model=model, tokenizer=tokenizer)

import openpyxl
import glob
import os
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

if __name__ == "__main__":
    rouge1_list, rougeL_list, sari_list, bert_list, bart_list, tcmscore_list = [], [], [], [], [], []
    model_name_list = ["gpt-4", "gpt-3.5-turbo", "chatglm", "Huatuo", "LLaMa", "zhongjing"]
    model_name_id = 0
    human_test_res = {key: [] for key in model_name_list}
    for model_name in model_name_list:
        print(model_name)
        # r_1, r_l = rouge_score(model_name)
        # rouge1_list.append(r_1)
        # rougeL_list.append(r_l)
        # sari_list.append(sari_score(model_name))
        # bert_list.append(bert_score(model_name))
        # bart_list.append(bart_Score(model_name))
        #
        # model_name_or_path = "/data2/.../ChatWK-main/Deberta-V3-base-tmnli-QAC"
        # print(model_name_or_path)
        # config = AutoConfig.from_pretrained(
        #     model_name_or_path,
        #     num_labels=3,
        #     finetuning_task="mnli",
        #     trust_remote_code=False
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_name_or_path, use_fast=not False, trust_remote_code=False
        # )
        # # print(tokenizer)
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     f"{model_name_or_path}",
        #     from_tf=bool(".ckpt" in model_name_or_path),
        #     config=config,
        #     ignore_mismatched_sizes=False,
        # )
        # model = model.to(f"cuda:{gpu_id}")
        # tcmscore_list.append(nli_score(model_name, model, tokenizer))
        # 定义文件夹路径
        folder_path = './model_test/human_test'

        # 获取文件夹中所有的 .xlsx 文件路径
        xlsx_files = glob.glob(os.path.join(folder_path, '*KK*.xlsx'))
        # 遍历每个 .xlsx 文件并读取内容
        for file_path in xlsx_files:
            print(f"Reading data from {file_path}:")
            # 打开 Excel 文件
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            # 遍历 Excel 表格的行
            id = 0
            for row in sheet.iter_rows(min_row=3, values_only=True):
                if id % 7 == model_name_id:
                    model_score = []
                    for col in range(4, 9):  # 读取第5-9列的数据
                        if row[col] is None:
                            print("_________________")
                        model_score.append(row[col])
                    human_test_res[model_name].append(model_score)
                id += 1
        model_name_id += 1
        if model_name_id == 3:
            model_name_id += 1
    rouge1_list, rougeL_list, sari_list, bert_list, bart_list, tcmscore_list = (
    np.array(rouge1_list), np.array(rougeL_list),
    np.array(sari_list), np.array(bert_list),
    np.array(bart_list), np.array(tcmscore_list)
    )
    metric = ["准确性", "中医专业性", "辨证论治逻辑性", "客观性", "完整性", "平均"]
    for j in range(len(metric)):
        human_list = []
        print("*" * 100, metric[j])
        for key, value in human_test_res.items():
            value = value
            flattened_data = [item for sublist in value for item in sublist]
            # 使用 numpy 计算平均值
            average_value = np.mean(flattened_data)
            metric_score = list()
            for i in range(len(metric)-1):
                elements = [sublist[i] for sublist in value]
                # print(metric[i])
                # print(elements)
                average_elements = np.mean(elements)
                metric_score.append(str(average_elements))
                if i == j:
                    human_list.append(average_elements)
            metric_score_text = "\t".join(metric_score)
            print(f"{key}\t{metric_score_text}\t{average_value}")
            if j == len(metric) - 1:
                human_list.append(average_value)

        human_list = np.array(human_list)
        correlation_coefficient1, p_value1 = pearsonr(rouge1_list, human_list)
        print("rouge1:", correlation_coefficient1, p_value1)
        correlation_coefficient2, p_value2 = pearsonr(rougeL_list, human_list)
        # print(rouge1_list)
        print("rougeL:", correlation_coefficient2, p_value2)
        correlation_coefficient3, p_value3 = pearsonr(sari_list, human_list)
        # print(sari_list)
        print("SARI:", correlation_coefficient3, p_value3)
        correlation_coefficient4, p_value4 = pearsonr(bert_list, human_list)
        print("Bert:", correlation_coefficient4, p_value4)
        correlation_coefficient5, p_value5 = pearsonr(bart_list, human_list)
        print("Bart:", correlation_coefficient5, p_value5)
        correlation_coefficient6, p_value6 = pearsonr(tcmscore_list, human_list)
        print("TCMScore:", correlation_coefficient6, p_value6)
        # print(tcmscore_list)
        # print(human_list)
        # value1 = cohen_kappa_score(rouge1_list, human_list)
        # print("rouge1:", value1)
        # value2 = cohen_kappa_score(rougeL_list, human_list)
        # # print(rouge1_list)
        # print("rougeL:", value2)
        # value3 = cohen_kappa_score(sari_list, human_list)
        # # print(sari_list)
        # print("SARI:", value3)
        # value4 = cohen_kappa_score(bert_list, human_list)
        # print("Bert:", value4)
        # value5 = cohen_kappa_score(bart_list, human_list)
        # print("Bart:", value5)
        # value6 = cohen_kappa_score(tcmscore_list, human_list)
        # print("TCMScore:", value6)




    # print("gpt-4")
    # rouge_score('gpt-4')
    # print("gpt-3.5")
    # rouge_score('gpt-3.5-turbo')
    # print("chatglm")
    # rouge_score('chatglm')
    # # print("ShenNong")
    # # rouge_score('ShenNong')
    # print("LLaMa")
    # rouge_score('LLaMa')
    # print("Huatuo")
    # rouge_score('Huatuo')
    # # print("qwen")
    # # rouge_score('qwen')
    # print("zhongjing")
    # rouge_score('zhongjing')
    #
    # print("gpt-4")
    # bert_score('gpt-4')
    # print("gpt-3.5")
    # bert_score('gpt-3.5-turbo')
    # print("chatglm")
    # bert_score('chatglm')
    # # print("BertScore: ShenNong")
    # # bert_score('ShenNong')
    # print("BertScore: LLaMa")
    # bert_score('LLaMa')
    # print("BertScore: Huatuo")
    # bert_score('Huatuo')
    # # print("BertScore: qwen")
    # # bert_score('qwen')
    #
    # print("BertScore: zhongjing")
    # bert_score('zhongjing')
    #
    # print("gpt-4")
    # bart_Score('gpt-4')
    # print("gpt-3.5")
    # bart_Score('gpt-3.5-turbo')
    # print("chatglm")
    # bart_Score('chatglm')
    # print("chatglm")
    # bart_Score('chatglm')
    # # print("BartScore: ShenNong")
    # # bart_Score("ShenNong")
    # print("BartScore: Huatuo")
    # bart_Score("Huatuo")
    # print("BartScore: LLaMa")
    # bart_Score("LLaMa")
    # # print("BartScore: qwen")
    # # bart_Score("qwen")
    #
    # print("BartScore: zhongjing")
    # bart_Score("zhongjing")
    #
    # print("gpt-4")
    # factcc_score('gpt-4')
    # print("gpt-3.5")
    # factcc_score('gpt-3.5-turbo')
    # print("chatglm")
    # factcc_score('chatglm')
    #
    # print("gpt-4")
    # sari_score('gpt-4')
    # print("gpt-3.5")
    # sari_score('gpt-3.5-turbo')
    # print("chatglm")
    # sari_score('chatglm')
    # print("SARI: ShenNong")
    # sari_score('ShenNong')
    # print("SARI: Huatuo")
    # sari_score('Huatuo')
    # print("SARI: LLaMa")
    # sari_score('LLaMa')
    #
    # # print("SARI: qwen")
    # # sari_score('qwen')
    #
    # print("SARI: zhongjing")
    # sari_score('zhongjing')

    # model_name_or_path = "/data2/.../ChatWK-main/Deberta-V3-base-tmnli-QAC"
    # print(model_name_or_path)
    # config = AutoConfig.from_pretrained(
    #     model_name_or_path,
    #     num_labels=3,
    #     finetuning_task="mnli",
    #     trust_remote_code=False
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name_or_path, use_fast=not False, trust_remote_code=False
    # )
    # # print(tokenizer)
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     f"{model_name_or_path}",
    #     from_tf=bool(".ckpt" in model_name_or_path),
    #     config=config,
    #     ignore_mismatched_sizes=False,
    # )
    # model = model.to(f"cuda:{gpu_id}")
    # print("NLI score:")
    #
    # print("*长度惩罚项 长短不一")
    # print("长度惩罚项-log")
    #
    # # print("gpt-4 query")
    # # nli_score_query('gpt-4', model, tokenizer)
    # #
    # print("gpt-4")
    # nli_score('gpt-4', model, tokenizer)
    #
    # print("gpt-3.5")
    # nli_score('gpt-3.5-turbo', model, tokenizer)
    # print("chatglm")
    # nli_score('chatglm', model, tokenizer)
    # #
    # # print("ShenNong")
    # # nli_score('ShenNong', model, tokenizer)
    # print("Huatuo")
    # nli_score('Huatuo', model, tokenizer)
    # print("LlaMa")
    # nli_score('LLaMa', model, tokenizer)
    #
    # # print("qwen")
    # # nli_score('qwen', model, tokenizer)
    #
    # print("zhongjing")
    # nli_score('zhongjing', model, tokenizer)
    #
    # print("tmnli_search")
    # nli_score('tmnli_search', model, tokenizer)


    # print("bart: tmnli_search")
    # bart_Score('tmnli_search-bing')
    # print("bert: tmnli_search")
    # bert_score('tmnli_search-bing')
    # print("SARI: tmnli_search")
    # sari_score('tmnli_search-bing')

    # print("bart: bing_search_top1")
    # bart_Score('bing_search_top1')
    # print("bert: bing_search_top1")
    # bert_score('bing_search_top1')
    # print("SARI: bing_search_top1")
    # sari_score('bing_search_top1')
