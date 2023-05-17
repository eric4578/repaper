import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random

from get_data import getData


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=16, out_features=8)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=8, out_features=4)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=4, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# 加载数据
X_train, Y_train = getData('train_data.txt')
X_train, _, Y_train, _ = train_test_split(X_train, Y_train, test_size=0.3, random_state=7)
X_val, Y_val = getData('eva_data.txt')

# 封装数据集和标签
train_data = TensorDataset(X_train, Y_train)
val_data = TensorDataset(X_val, Y_val)

# 定义超参数和优化器
batch_size = 256
learning_rate = 0.0001
num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 定义训练函数和验证函数
def train(model, criterion, optimizer, data_loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += y_batch.size(0)
        total_correct += (torch.round(outputs).squeeze() == y_batch).sum().item()

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)

    return accuracy, avg_loss


def validate(model, criterion, data_loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            total_samples += y_batch.size(0)
            y_true += list(y_batch.cpu().numpy())
            y_pred += list(torch.round(outputs).squeeze().cpu().numpy())
            total_correct += (torch.round(outputs).squeeze() == y_batch).sum().item()

        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(data_loader)
        cm = confusion_matrix(y_true, y_pred)

    return accuracy, avg_loss, cm

if __name__  == "__main__":
    # 训练模型并输出结果
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    for epoch in range(num_epochs):
        train_acc, train_loss = train(model, criterion, optimizer, train_loader)
        val_acc, val_loss, cm = validate(model, criterion, val_loader)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        if epoch >= 498:
            print(cm)

    # 保存模型
    torch.save(model.state_dict(), 'path/to/model.pt')