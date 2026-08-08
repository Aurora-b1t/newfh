import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================
# 1. 准备数据与 DataLoader
# ==========================
# 模拟生成一些简单的分类数据（假设输入特征维度为20，共4个类别，1000个样本）
input_dim = 20
class_num = 4
num_samples = 1000

# 随机生成特征和标签
x_data = torch.randn(num_samples, input_dim).float()
y_data = torch.randint(0, class_num, (num_samples,)).long()

# 将特征和标签组合成 PyTorch 数据集
dataset = TensorDataset(x_data, y_data)

# 使用 DataLoader 包装数据集，设置批量大小和是否打乱数据
# shuffle=True 表示每个训练轮次打乱数据，有助于模型更好收敛
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================
# 2. 搭建神经网络模型
# ==========================
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        # 定义简单的全连接层
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播过程
        x = self.relu(self.fc1(x))
        output = self.fc2(x)  # 输出层不加激活函数，CrossEntropyLoss 内部包含 softmax
        return output

# 实例化模型
model = SimpleClassifier(input_dim, class_num)

# ==========================
# 3. 设置损失函数和优化器
# ==========================
# 多分类任务通常使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用 Adam 优化器，学习率设置为 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================
# 4. 模型训练
# ==========================
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0
    # 设置模型为训练模式
    model.train()
    
    for x_batch, y_batch in dataloader:
        # 前向传播：获取模型预测结果
        outputs = model(x_batch)
        
        # 计算损失
        loss = criterion(outputs, y_batch)
        
        # 反向传播与参数更新
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数
        
        total_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

# ==========================
# 5. 模型评估
# ==========================
# 设置模型为评估模式
model.eval()
correct = 0
total = 0

# 评估时关闭梯度计算，节省内存并加速
with torch.no_grad():
    for x_batch, y_batch in dataloader:
        outputs = model(x_batch)
        # 获取预测类别（概率最大的索引）
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f'模型在数据集上的准确率: {100 * correct / total:.2f}%')