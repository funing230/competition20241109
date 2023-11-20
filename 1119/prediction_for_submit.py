import os
import random
from util import *
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForPreTraining, BertModel
from Lookahead.optimizer import Lookahead
from attacks import FGM, PGD
vocab_dict_file = 'vocab_dict.json'
from NeZha.model.modeling_nezha import NeZhaPreTrainedModel, NeZhaModel
from NeZha.model.configuration_nezha import NeZhaConfig

def predict(model, data_loader, device='cuda'):
    model.eval()

    preds_list = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            outputs = model(input_ids=data['input_ids'].to(device).long(),
                            attention_mask=data['attention_mask'].to(device).long())

            preds = outputs
            preds = F.sigmoid(preds)
            # print(preds)
            preds_list.append(preds.cpu().detach())

    preds_list = np.concatenate(preds_list)
    print('preds_list:', preds_list.shape)

    return preds_list



test = read_data('./dataset/testA.json')

if os.path.exists(vocab_dict_file):
    with open(vocab_dict_file, 'r', encoding='utf-8') as file:
        vocab_dict = json.load(file)
else:
    train = read_data('./dataset/train.json')
    word_dict = get_dict(train)
    print('original vocab size:', len(word_dict))
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = special_tokens + list(word_dict.keys())
    vocab_dict = {v: k for k, v in enumerate(vocab)}

    with open(vocab_dict_file, 'w', encoding='utf-8') as file:
        json.dump(vocab_dict, file, ensure_ascii=False)

test_text=get_all_text(test)


BATCH_SIZE = 64

test_dataset = BERTDataset_Finetune(test, vocab_dict, True, 256)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
config = NeZhaConfig.from_pretrained('pretrain/FUNINF_nezha_config.json')
model = NewNeZha(config)
# 加载已保存的模型状态
saved_model_path = 'model_nezha_final_0.350_fold_0_1_0.9982_3.1971.pth' #加载模型不用从新开始训练--------------
model.load_state_dict(torch.load(saved_model_path))
model = model.to(device)
result = predict(model, test_loader)
# 提交最终结果
test_ids = test['id'].tolist()
predicted_labels = [np.argmax(pred) for pred in result]
# 创建DataFrame并保存为CSV
sub = pd.DataFrame({'id': test_ids, 'label': predicted_labels})
sub.to_csv('result_1119.csv', encoding='utf-8', index=False)