import nltk
from jieba import cut
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sklearn
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
# from gensim.models import Word2Vec
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import random
import json
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
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
from NeZha.model.modeling_nezha import NeZhaPreTrainedModel, NeZhaModel
from NeZha.model.configuration_nezha import NeZhaConfig
import torch
from torch import nn

#读取数据json
def read_data(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # 将JSON数据转换为DataFrame
    dataset = pd.DataFrame(data_list)
    # dataset = dataset.values
    return dataset
#分解单词 title  assignee  abstract

def token_data(dataset):
    tokenized_data = []
    for item in dataset:
        # t1=item
        # t2=item[1]
        # t3=cut(item[1])
        title_tokens = list(cut(item[1]))  # 分词标题
        assignee_tokens = list(cut(item[2]))  # 分词专利权人
        abstract_tokens = list(cut(item[3]))  # 分词摘要
        tokenized_data.append({
            # "id":item[0],
            "title": title_tokens,
            "assignee": assignee_tokens,
            "abstract": abstract_tokens,
            # "label_id":item[4]
        })
    return tokenized_data
#数据清洗


def process_data(messages):
    with open('./dataset/stop_words.txt', 'r', encoding='utf-8') as file:
        chinese_stop_words = set([line.strip() for line in file])
    processed_corpus = []
    for text in messages:
        # 去除中文标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 中文分词
        words = jieba.cut(text)
        # 过滤停用词并拼接成字符串
        processed_text = ' '.join([word for word in words if word not in chinese_stop_words])
        # 将处理后的文本添加到列表中
        if processed_text != '':
            processed_corpus.append(processed_text)
        # processed_corpus.append([item for item in processed_text if len(item) != 0])
    return processed_corpus

# Construct word dict
def get_dict(train):
    train_list = token_data(train.values)
    sentences = [process_data(item["title"])
                 + process_data(item["assignee"])
                 + process_data(item["abstract"]) for item in train_list]
    cleaned_nested_list = [[word for word in sublist if word != ''] for sublist in sentences]
    words_dict = defaultdict(int)
    for i in tqdm(range(len(cleaned_nested_list))):
        text = cleaned_nested_list[i]
        for c in text:
            words_dict[c] += 1
    return words_dict

def get_all_text(data):
    data['combined_text'] = data['title'] + " " + data['assignee'] + " " + data['abstract']
    for i in range(len(data)):
        data.at[i, 'combined_text'] = re.sub(r'[^\w\s]', '', str(data.at[i, 'combined_text']))

        return data


class BERTDataset(Dataset):
    def __init__(self, corpus, vocab, test_flag, seq_len=128):
        self.vocab = vocab
        self.seq_len = seq_len
        self.texts = corpus
        self.test_flag = test_flag
        self.ratio = self.pn(3)

    def __len__(self):
        return len(self.texts)

    def get_sentence(self, idx):
        # 假设corpus中的文本列名为 'combined_text'
        t = self.texts['combined_text'].iloc[idx]
        return t

    # 每个片段长度n（最大为3），根据概率公式计算得到。比如1-gram、2-gram、3-gram的的概率分别为6/11、3/11、2/11
    # Masked-ngram-LM reference: albert https://blog.csdn.net/weixin_37947156/article/details/101529943
    def pn(self, n):
        denominator = 0
        for i in range(1, n + 1):
            denominator += 1.0 / i
        ratio = []
        for i in range(1, n + 1):
            ratio.append(1.0 / i / denominator)
        return ratio

    def random_word(self, sentence):
        # tokens = sentence.split()
        # k = len(tokens)
        tokens = list(jieba.cut(sentence))  # 使用jieba进行分词
        k = len(tokens)

        i = 0
        flag = False
        output_label = []
        while i < k:
            prob = random.random()
            if self.test_flag == False and flag == False and prob < 0.15:  # 15%的词被mask
                prob /= 0.15
                if prob < 0.8:  # 被mask的词中，80%用[MASK]token代替
                    prob /= 0.8
                    if prob <= self.ratio[0]:
                        n_gram = 1
                    elif prob <= self.ratio[0] + self.ratio[1]:
                        n_gram = 2
                    else:
                        n_gram = 3

                    j = i + n_gram
                    while i < j and i < k:
                        token = tokens[i]
                        tokens[i] = self.vocab['[MASK]']
                        output_label.append(self.vocab.get(token, self.vocab['[UNK]']))  # 得到token对应的id
                        # get()函数的第二个参数：字典中不包含该键时，get函数函数返回第二个参数
                        i += 1

                    flag = True  # 防止连续mask,不然可能整个句子都被mask了
                elif prob < 0.9:  # 10%的词用其他随机一个词代替
                    token = str(random.randrange(5, len(self.vocab)))
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    # print(token, tokens[i])
                    output_label.append(self.vocab.get(token, self.vocab['[UNK]']))

                    flag = False
                    i += 1
                else:  # 10%的词不变
                    # -100就表示不对该词进行mask
                    token = tokens[i]
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(-100)

                    flag = False
                    i += 1
            else:
                token = tokens[i]
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)

                flag = False
                i += 1
        if len(tokens) != len(output_label):
            print('error!', k, len(tokens), len(output_label))

        return tokens, output_label

    def __getitem__(self, idx):
        # Get original input
        t = self.get_sentence(idx)

        # 1-gram to 3-gram
        t_random, t_label = self.random_word(t)

        # targets
        # label_id = self.texts['label_id'].iloc[idx]  # 假设标签列名为 'label_id'

        # Construct bert input
        t = [self.vocab['[CLS]']] + t_random + [self.vocab['[SEP]']]

        # t_label = [self.vocab['[CLS]']] + t_label + [self.vocab['[SEP]']]
        t_label = [-100] + t_label + [self.vocab['[SEP]']]

        bert_input = t[:self.seq_len]
        bert_label = t_label[:self.seq_len]
        # print(t, t_label)

        # Padding
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]

        bert_input.extend(padding)
        bert_label.extend(padding_label)

        # 构造输出
        output = {"input_ids": np.array(bert_input),
                  'attention_mask': np.array(attention_mask),
                  "labels": np.array(bert_label)}  # 假设您想将标签直接作为输出


        return output

class NewNeZha(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Pretrain
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores

        # Finetune
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # return logits

class BERTDataset_for_test(Dataset):
    def __init__(self, corpus, vocab, test_flag, seq_len=128):
        self.vocab = vocab
        self.seq_len = seq_len
        self.texts = corpus
        self.test_flag = test_flag
        self.ratio = self.pn(3)

    def __len__(self):
        return len(self.texts)

    def get_sentence(self, idx):
        # 假设corpus中的文本列名为 'combined_text'
        t = self.texts['combined_text'].iloc[idx]
        return t

    # 每个片段长度n（最大为3），根据概率公式计算得到。比如1-gram、2-gram、3-gram的的概率分别为6/11、3/11、2/11
    # Masked-ngram-LM reference: albert https://blog.csdn.net/weixin_37947156/article/details/101529943
    def pn(self, n):
        denominator = 0
        for i in range(1, n + 1):
            denominator += 1.0 / i
        ratio = []
        for i in range(1, n + 1):
            ratio.append(1.0 / i / denominator)
        return ratio

    def random_word(self, sentence):
        tokens = sentence.split()
        k = len(tokens)

        i = 0
        flag = False
        output_label = []
        while i < k:
            prob = random.random()
            if self.test_flag == False and flag == False and prob < 0.15:  # 15%的词被mask
                prob /= 0.15
                if prob < 0.8:  # 被mask的词中，80%用[MASK]token代替
                    prob /= 0.8
                    if prob <= self.ratio[0]:
                        n_gram = 1
                    elif prob <= self.ratio[0] + self.ratio[1]:
                        n_gram = 2
                    else:
                        n_gram = 3

                    j = i + n_gram
                    while i < j and i < k:
                        token = tokens[i]
                        tokens[i] = self.vocab['[MASK]']
                        output_label.append(self.vocab.get(token, self.vocab['[UNK]']))  # 得到token对应的id
                        # get()函数的第二个参数：字典中不包含该键时，get函数函数返回第二个参数
                        i += 1

                    flag = True  # 防止连续mask,不然可能整个句子都被mask了
                elif prob < 0.9:  # 10%的词用其他随机一个词代替
                    token = str(random.randrange(5, len(self.vocab)))
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    # print(token, tokens[i])
                    output_label.append(self.vocab.get(token, self.vocab['[UNK]']))

                    flag = False
                    i += 1
                else:  # 10%的词不变
                    # -100就表示不对该词进行mask
                    token = tokens[i]
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(-100)

                    flag = False
                    i += 1
            else:
                token = tokens[i]
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)

                flag = False
                i += 1
        if len(tokens) != len(output_label):
            print('error!', k, len(tokens), len(output_label))

        return tokens, output_label

    def __getitem__(self, idx):
        # 处理输入文本
        t = self.get_sentence(idx)
        t_random, _ = self.random_word(t)  # 在预测时，只需处理输入文本，不用考虑标签

        # 构建BERT输入
        t = [self.vocab['[CLS]']] + t_random + [self.vocab['[SEP]']]
        bert_input = t[:self.seq_len]

        # 填充
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]
        bert_input.extend(padding)

        output = {"input_ids": np.array(bert_input), 'attention_mask': np.array(attention_mask)}
        return output


class NewNeZha_k(NeZhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Pretrain
        '''
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores
        '''

        # Finetune
        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # print(outputs[0].shape, outputs[0][:,0,:].shape)
        # cls = self.dropout(outputs[0][:,0,:])
        logits = self.classifier(outputs[0][:, 0, :])
        return logits


class BERTDataset_Finetune(Dataset):
    def __init__(self, corpus, vocab, test_flag, seq_len=128):
        self.vocab = vocab
        self.seq_len = seq_len
        self.texts = corpus
        self.test_flag = test_flag
        self.ratio = self.pn(3)

    def __len__(self):
        return len(self.texts)

    def get_sentence(self, idx):
        # 假设corpus中的文本列名为 'combined_text'
        t = self.texts['combined_text'].iloc[idx]
        return t

    # 每个片段长度n（最大为3），根据概率公式计算得到。比如1-gram、2-gram、3-gram的的概率分别为6/11、3/11、2/11
    # reference: albert https://blog.csdn.net/weixin_37947156/article/details/101529943
    def pn(self, n):
        denominator = 0
        for i in range(1, n + 1):
            denominator += 1.0 / i
        ratio = []
        for i in range(1, n + 1):
            ratio.append(1.0 / i / denominator)
        return ratio

    def random_word(self, sentence):
        # tokens = sentence.split()
        # k = len(tokens)
        tokens = list(jieba.cut(sentence))  # 使用jieba进行分词
        k = len(tokens)

        i = 0
        flag = False
        output_label = []
        while i < k:
            prob = random.random()
            if self.test_flag == False and flag == False and prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    prob /= 0.8
                    if prob <= self.ratio[0]:
                        n_gram = 1
                    elif prob <= self.ratio[0] + self.ratio[1]:
                        n_gram = 2
                    else:
                        n_gram = 3

                    j = i + n_gram
                    while i < j and i < k:
                        token = tokens[i]
                        tokens[i] = self.vocab['[MASK]']
                        output_label.append(self.vocab.get(token, self.vocab['[UNK]']))
                        i += 1

                    flag = True
                elif prob < 0.9:
                    token = str(random.randrange(5, len(self.vocab)))
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(self.vocab.get(token, self.vocab['[UNK]']))

                    flag = False
                    i += 1
                else:
                    # Tokens with indices set to -100 are ignored (masked)
                    token = tokens[i]
                    tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                    output_label.append(-100)

                    flag = False
                    i += 1
            else:
                # Tokens with indices set to -100 are ignored (masked)
                token = tokens[i]
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                output_label.append(-100)

                flag = False
                i += 1
        if len(tokens) != len(output_label):
            print('error!', k, len(tokens), len(output_label))

        return tokens, output_label

    def __getitem__(self, idx):
        # Get original input
        t = self.get_sentence(idx)

        # 1-gram to 3-gram
        t_random, t_label = self.random_word(t)

        # targets(test has no targets)
        try:
            # temp=self.texts
            # temp=self.texts.label_id
            # temp=self.texts.label_id.iloc[idx]
            label_list = [self.texts.label_id.iloc[idx]]
        except:
            label_list = []

        # Construct bert input
        t = [self.vocab['[CLS]']] + t_random + [self.vocab['[SEP]']]

        # t_label = [self.vocab['[CLS]']] + t_label + [self.vocab['[SEP]']]
        # t_label = [-100] + t_label + [self.vocab['[SEP]']]

        bert_input = t[:self.seq_len]

        # Pretrain
        # bert_label = t_label[:self.seq_len]
        # bert_label.extend(padding_label)
        # print(t, t_label)

        # Finetune
        bert_label = [0] * 36
        for i in label_list:
            bert_label[int(i)] = 1

        # Padding
        padding = [self.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        # padding_label = [-100 for _ in range(self.seq_len - len(bert_input))]
        attention_mask = len(bert_input) * [1] + len(padding) * [0]

        bert_input.extend(padding)

        output = {"input_ids": np.array(bert_input),
                  'attention_mask': np.array(attention_mask),
                  "bert_label": np.array(bert_label)}

        return output

