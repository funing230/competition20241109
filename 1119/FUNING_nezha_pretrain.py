import os
from Lookahead.optimizer import Lookahead
from attacks import FGM, PGD
from util import *
from NeZha.model.configuration_nezha import NeZhaConfig
import warnings
import json
vocab_dict_file = 'vocab_dict.json'
# 屏蔽所有警告
warnings.filterwarnings('ignore')
# wordpiece bpe  21128  20000


# Load dataset
# train = pd.read_csv('./track1_data/pretrain.tsv', sep=',', names=['id', 'text', 'label'], dtype=str)
# 假设 read_data 返回的是DataFrame
train = read_data('./dataset/train.json')

if os.path.exists(vocab_dict_file):
    with open(vocab_dict_file, 'r', encoding='utf-8') as file:
        vocab_dict = json.load(file)
else:
    word_dict = get_dict(train)
    print('original vocab size:', len(word_dict))
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = special_tokens + list(word_dict.keys())
    vocab_dict = {v: k for k, v in enumerate(vocab)}

    with open(vocab_dict_file, 'w', encoding='utf-8') as file:
        json.dump(vocab_dict, file, ensure_ascii=False)

train_text=get_all_text(train)

pretrain_dataset = BERTDataset(corpus=train_text[['combined_text', 'label_id']], vocab=vocab_dict, test_flag=False, seq_len=256)##128


BATCH_SIZE = 64
train_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Construct model, loss function, optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()

config = NeZhaConfig.from_pretrained('pretrain/FUNINF_nezha_config.json')

model = NewNeZha(config)

# model.load_state_dict(torch.load('pretrain/model_nezha_large_pre_270_0.591.pth'))
model = model.to(device)
# print(model)

# LookAhead优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)
optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

# fgm
fgm = FGM(model)
K = 3

# Start training
NUM_EPOCHS = 300
best_auc = 0.
for epoch in range(NUM_EPOCHS):
    losses = []
    model.train()
    for data in tqdm(train_loader):
        optimizer.zero_grad()

        labels = data['labels'].to(device).long()

        outputs = model(input_ids=data['input_ids'].to(device).long(),
                        attention_mask=data['attention_mask'].to(device).long())
        # print(outputs.shape, labels.shape)

        mask = (labels != -100)
        loss = criterion(outputs[mask].view(-1, len(vocab_dict)), labels[mask].view(-1))
        losses.append(loss.cpu().detach().numpy())

        loss.backward()

        optimizer.step()
        model.zero_grad()

        tqdm.write(f'epoch:{epoch} train loss:{np.mean(losses):.3f}')

    if np.mean(losses) < 0.35 or epoch % 10 == 0:
        torch.save(model.state_dict(), 'model/model_nezha_{}_{:.3f}.pth'.format(epoch, np.mean(losses)))

