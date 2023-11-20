import os
from util import *
from NeZha.model.configuration_nezha import NeZhaConfig
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
vocab_dict_file = 'vocab_dict.json'
config = NeZhaConfig.from_pretrained('pretrain/FUNINF_nezha_config.json')

model = NewNeZha(config)  # 创建模型实例，确保配置与保存时一致
model.load_state_dict(torch.load('pretrain/model_nezha_224_0.137.pth'))

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

# 创建测试数据集
BATCH_SIZE = 128
test_dataset = BERTDataset_for_test(corpus=test_text[['combined_text']], vocab=vocab_dict, test_flag=False, seq_len=128)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
model.eval()  # 确保模型处于评估模式
predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        try:
            input_ids = batch['input_ids'].to(device)  # 移动到相同的设备
            attention_mask = batch['attention_mask'].to(device)  # 移动到相同的设备
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"An error occurred: {e}")
            outputs = torch.zeros_like(batch['input_ids'])  # 创建一个与输入相同形状的零张量

        predictions.extend(outputs.cpu().numpy())

# 假设predictions是模型输出的概率
# 转换为类别标签（这里假设是多分类问题）
predicted_labels = [np.argmax(pred) for pred in predictions]

# 获取测试数据集的ID
test_ids = test['id'].tolist()

# 创建DataFrame并保存为CSV
sub = pd.DataFrame({'id': test_ids, 'label': predicted_labels})
sub.to_csv('result_1118.csv', encoding='utf-8', index=False)
