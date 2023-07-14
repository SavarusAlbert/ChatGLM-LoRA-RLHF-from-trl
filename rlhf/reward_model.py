import torch
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score, accuracy_score


def reward_model(preds, labels, task_dataset):
    scores = []
    classification_task = ['CHIP-CTC', 'IMCS-V2-DAC', 'CHIP-STS', 'KUAKE-QTR', 'KUAKE-QIC', 'KUAKE-QQR', 'KUAKE-IR']
    multi_choice_task = ['CHIP-CDN']
    ner_task = ['CMeEE-V2', 'IMCS-V2-NER', 'CHIP-MDCFNPC', 'IMCS-V2-SR']
    CMeIE_task = ['CMeIE']
    CHIP_task = ['CHIP-CDEE']
    MRG_task = ['IMCS-V2-MRG']
    MedDG_task = ['MedDG']

    for pred, label, task in zip(preds, labels, task_dataset):
        pred = pred.replace(":", "：", 100).replace(",", "，", 100).replace(";", "；", 100)
        label = label.replace(":", "：", 100).replace(",", "，", 100).replace(";", "；", 100)
        if task in classification_task:
            scores.append(torch.tensor(1)) if pred.strip() == label.strip() else scores.append(torch.tensor(0))
        elif task in multi_choice_task:
            pred_ans = [w.strip() for w in pred.split("\n")[-1].split("，") if len(w.strip()) > 0]
            label_ans = [w.strip() for w in label.split("\n")[-1].split("，") if len(w.strip()) > 0]
            
            score = calculate_F1_score(pred_ans, label_ans)
            scores.append(torch.tensor(score))
        elif task in ner_task:
            pred_ans = pred.split("\n")
            label_ans = label.split("\n")

            pred_ans_list, label_ans_list = pred_label_list(pred_ans, label_ans)

            score = calculate_F1_score(label_ans_list, pred_ans_list)
            scores.append(torch.tensor(score))
        elif task in CMeIE_task:
            pred_ans = pred.split("\n")
            label_ans = label.split("\n")

            pred_ans_list, label_ans_list = pred_label_list(pred_ans, label_ans, special_symbol="。")
            
            score = calculate_F1_score(label_ans_list, pred_ans_list)
            scores.append(torch.tensor(score))
        elif task in CHIP_task:
            pred_ans = pred.split("\n")
            label_ans = label.split("\n")

            score = calculate_F1_score(label_ans, pred_ans)
            scores.append(torch.tensor(score))
        elif task in MRG_task:
            pred_ans = pred.split("\n")
            label_ans = label.split("\n")
            max_len = len(label_ans)
            pred_ans = pred_ans[:max_len] + [''] * (max_len - len(pred_ans))

            score_list = []
            score_list.append(pred_ans[0] == label_ans[0])
            for p, l in zip(pred_ans[1:], label_ans[1:]):
                if p.split("：")[0] == l.split("：")[0] and min(len(p.split("：")), len(l.split("："))) > 1:
                    score_list.append(compute_similarity(p.split("：")[1], l.split("：")[1]) > 0.8)
                else:
                    score_list.append(False)
            score = accuracy_score([True] * max_len, score_list)           
            scores.append(torch.tensor(score))
        elif task in MedDG_task:
            score = compute_similarity(pred, label)
            scores.append(score.clone().detach())

    return [torch.clamp(s, 0, 1).float() for s in scores]            


def compute_similarity(pred, label):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": [],
    }
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))
    rouge = Rouge()
    if hypothesis and reference:
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]
    else:
        result = {'rouge-1': {'r': 0., 'p': 0., 'f': 0.},
                    'rouge-2': {'r': 0., 'p': 0., 'f': 0.},
                    'rouge-l': {'r': 0., 'p': 0., 'f': 0.}}

    for k, v in result.items():
        score_dict[k].append(round(v["f"] * 100, 4))
    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
    score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    
    def reward_function(score_dict):
        rouge_1_weight = 0.1
        rouge_2_weight = 0.1
        rouge_l_weight = 0.5
        bleu_4_weight = 0.3

        scores = []
        for i in range(len(score_dict["rouge-1"])):
            rouge_1_score = score_dict["rouge-1"][i]
            rouge_2_score = score_dict["rouge-2"][i]
            rouge_l_score = score_dict["rouge-l"][i]
            bleu_4_score = score_dict["bleu-4"][i]
            
            average_score = (rouge_1_weight * rouge_1_score + rouge_2_weight * rouge_2_score + rouge_l_weight * rouge_l_score + bleu_4_weight * bleu_4_score) / (rouge_1_weight + rouge_2_weight + rouge_l_weight + bleu_4_weight)
            
            normalized_score = (average_score - 0) / (100 - 0)
            scores.append(torch.tensor(normalized_score))
        
        return scores

    return reward_function(score_dict)[0]


def calculate_F1_score(pred_list, label_list):
    pred_len = len(pred_list)
    label_len = len(label_list)

    pred_list += [''] * (label_len - pred_len)
    label_list += [''] * (pred_len - label_len)

    return f1_score(y_true=label_list, y_pred=pred_list, average="weighted")


def pred_label_list(pred_ans, label_ans, special_symbol = "，"):
    pred_ans_list = []
    label_ans_list = []

    for p in pred_ans:
        if len(p.split("：")) == 1:
            pred_ans_list.append(p.split("：")[0] + "")
        if len(p.split("：")) > 1:
            for _ in p.split("：")[1].split(special_symbol):
                pred_ans_list.append(p.split("：")[0] + _)

    for l in label_ans:
        if len(l.split("：")) == 1:
            label_ans_list.append(l.split("：")[0] + "")
        if len(l.split("：")) > 1:
            for _ in l.split("：")[1].split(special_symbol):
                label_ans_list.append(l.split("：")[0] + _)

    return pred_ans_list, label_ans_list
    