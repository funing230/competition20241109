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
from transformers.models.bert.modeling_bert import (
    BertOutput,
    BertPooler,
    BertSelfOutput,
    BertIntermediate,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPreTrainingHeads,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
# 屏蔽所有警告
import warnings
warnings.filterwarnings('ignore')
SEEDS = [0, 68, 82, 1989, 75, 2021, 3033, 192, 385, 2022]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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
# Predict test
def predict(model, data_loader, device='cuda'):
    model.eval()

    preds_list = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            outputs = model(input_ids=data['input_ids'].to(device).long(),
                            attention_mask=data['attention_mask'].to(device).long())

            preds = outputs
            preds = F.softmax(preds,dim=1)
            # print(preds)
            preds_list.append(preds.cpu().detach())

    preds_list = np.concatenate(preds_list)
    print('preds_list:', preds_list.shape)

    return preds_list


# Construct model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Start training
NUM_EPOCHS = 200 #5------------------------------------
best_auc = -100
criterion = nn.CrossEntropyLoss()  # BCEWithLogitsLoss()

config = NeZhaConfig.from_pretrained('pretrain/FUNINF_nezha_config.json', num_labels=36)

for i in range(1):
    set_seed(SEEDS[i])

    train = read_data('./dataset/train.json')
    train_text=get_all_text(train)
    # X_text = train_text

    # train, valid = train_test_split(X_text, test_size=0.1, random_state=42)


    train_dataset = BERTDataset_Finetune(train_text, vocab_dict, True, 256)
    # valid_dataset = BERTDataset_Finetune(valid, vocab_dict, True, 256)
    test_dataset = BERTDataset_Finetune(test, vocab_dict, True, 256)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NewNeZha_k(config)
    model.load_state_dict(torch.load('model/model_nezha_250_4.234.pth'))
    # model.load_state_dict(torch.load('model_nezha_299_0.241.pth'))
    model = model.to(device)
    # print(model)

    # LookAhead
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-7)
    optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

    # fgm
    fgm = FGM(model)
    K = 3

    for epoch in range(NUM_EPOCHS):
        losses = []
        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()

            labels = data['bert_label'].to(device).float()

            outputs = model(input_ids=data['input_ids'].to(device).long(),
                            attention_mask=data['attention_mask'].to(device).long())

            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels).mean()
            # print(loss)
            losses.append(loss.cpu().detach().numpy())

            loss.backward()

            fgm.attack(epsilon=0.1)  # 在embedding上添加对抗扰动
            adv_outputs = model(input_ids=data['input_ids'].to(device).long(),
                                attention_mask=data['attention_mask'].to(device).long())

            temp=np.argmax(adv_outputs.cpu().detach().numpy())
            temp2=np.argmax(labels.cpu().detach().numpy())


            loss_adv = criterion(adv_outputs, labels).mean()
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

            optimizer.step()
            model.zero_grad()

            tqdm.write(f'350 fold:{i} epoch:{epoch} train loss:{np.mean(losses):.3f}')


        if np.mean(losses) < 0.35 or epoch % 10 == 0:
            torch.save(model.state_dict(),
                       'output/model_nezha_final_0.350_fold_{}_{}_{:.4f}.pth'.format(i, epoch, np.mean(losses)))

            result = predict(model, test_loader)
            # 提交最终结果
            test_ids = test['id'].tolist()
            predicted_labels = [np.argmax(pred) for pred in result]
            # 创建DataFrame并保存为CSV
            sub = pd.DataFrame({'id': test_ids, 'label': predicted_labels})
            sub.to_csv('output/sumit.loss_{:.4f}.csv'.format(np.mean(losses)), encoding='utf-8', index=False)




