import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
import time

# spacy 无法下载 解决办法 pip --default-timeout=10000 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz
# 分词器
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

# 切分数据集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 构建词向量
MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

# 整理数据集
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size=BATCH_SIZE,
                                                           device=device)


# 定义RNN
class RNN(nn.Module):
    def __init__(self, imput_dim, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(imput_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out


# 定义维度
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
# model = nn.RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model = model.to(device)

# 计算二元交叉熵
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# 优化器
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.02)
# optimizer = torch.optim.Adam(model.parameters(), betas=(0.7, 0.995), lr=0.005)


# 计算 accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# 训练方法
def train(iterator):
    epoch_loss = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        # batch.text 就是上面forward函数的参数text，压缩维度是为了和batch.label维度一致
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 测试方法
def evaluate(iterator):
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_acc += acc.item()
            epoch_loss += loss.item()
    return epoch_acc / len(iterator),epoch_loss / len(iterator)


# 计算消耗时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# 测试
for epoch in range(10):
    start_time = time.time()
    train_loss = train(train_iterator)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    torch.save(model.state_dict(), 'jn_rnn')

    print(
        'Epoch: %d |train loss: %.3f |cost: %d m %d s' % (
            epoch + 1, train_loss, epoch_mins, epoch_secs))

    test_acc,test_loss = evaluate(test_iterator)
    print(
        'Epoch: %d |evaluate loss: %.2f |evaluate accuracy: %.2f' % (epoch + 1, test_loss, test_acc * 100))
