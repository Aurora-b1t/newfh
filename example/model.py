import torch

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import gzip
import itertools

device = torch.device('cuda')

num_train = 60000  # 60k train examples
num_test = 10000  # 10k test examples


BATCH_SIZE = 100

#数据标准化
class StandardScaler(object):
    def __init__(self):
        pass
    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

# ... 截断正态分布初始化 ...#
def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int = 7, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        #权重矩阵是一个 3D Tensor 维度为: (7个模型, 输入维度, 输出维度)
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size, hidden_size=200, learning_rate=1e-3, use_decay=False):
        super(EnsembleModel, self).__init__()
        self.hidden_size = hidden_size
        # 定义 4 层隐藏层
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay

        self.output_dim = reward_size
        # Add variance output
        # 第 5 层是输出层，输出维度乘 2，因为前一半是均值 (mean)，后一半是对数方差 (logvar)
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        # 定义方差的上下界
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.apply(init_weights) # 调用外部的截断正态分布初始化
        self.swish = Swish()

    def forward(self, x, ret_log_var=False):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)
        # 提取前一半作为均值预测
        mean = nn5_output[:, :, :self.output_dim]
        # 利用 softplus 函数的平滑特性，将网络输出的原始 logvar 强行限制在 [min_logvar, max_logvar] 之间。
        logvar = self.max_logvar - F.softplus(self.max_logvar - nn5_output[:, :, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        #L_{decay} = \frac{\lambda}{2} \sum w^2
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        # inv_var = 1 / var (即 1 / e^logvar)
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            # (y - mu)^2 / (var)
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        #给 max_logvar 加正数惩罚 (促使它变小)，给 min_logvar 减去正数 (促使它变大)
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        # print('loss:', loss.item())
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.shape, torch.mean(param.grad), param.grad.flatten()[:5])
        self.optimizer.step()


class EnsembleDynamicsModel():
    def __init__(self, network_size, elite_size, state_size, action_size, reward_size=1, hidden_size=200, use_decay=False):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay)
        self.scaler = StandardScaler()

    def train(self, inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update # 容忍连续多少个 Epoch 没有提升
        self._epochs_since_update = 0 # 连续未提升的计数器
        self._state = {}
        # _snapshots 字典用于记录 7 个子模型各自的【历史最佳表现】。
        # 格式为 {模型序号: (最佳Epoch数, 历史最小验证集误差)}。
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        # --- 开始切分验证集 (Holdout Set) ---
        # 根据设定的比例 (如 0.2)，计算需要抽出多少条数据作为验证集
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        # np.random.permutation 生成一个从 0 到 数据总量-1 的随机乱序数组
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]
        # 切片操作：从开头到 num_holdout 的数据作为验证集 (Holdout)
        # 从 num_holdout 到最后的数据作为真正用于计算梯度的训练集 (Train)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]
        #数据处理
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        #holdout_inputs 的形状是 2D (holdout_size, dim)变为变为 (network_size=7, holdout_size, dim)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])

        for epoch in itertools.count():
            # 为 7 个子网络分别生成 7 种完全不同的训练集，形状为 (7, 训练集大小) 的 2D 矩阵
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.network_size)])
            # train_idx = np.vstack([np.arange(train_inputs.shape[0])] for _ in range(self.network_size))
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                # 形状为 (7, batch_size)
                idx = train_idx[:, start_pos: start_pos + batch_size]
                train_input = torch.from_numpy(train_inputs[idx]).float().to(device)
                train_label = torch.from_numpy(train_labels[idx]).float().to(device)
                losses = []
                # 1. 前向传播：同时算出 7 个模型预测的均值和对数方差
                mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                # 2. 计算损失：计算高斯负对数似然
                loss, _ = self.ensemble_model.loss(mean, logvar, train_label)
                self.ensemble_model.train(loss)
                losses.append(loss)

            with torch.no_grad():
                holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                # 返回的 holdout_mse_losses 是一个长度为 7 的一维数组，代表 7 个模型的真实预测偏差。
                _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                # np.argsort 对 7 个误差值从小到大排序，返回的是排序后的原数组“索引”。
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                # 截取误差最小的前 elite_size (如 5) 个模型的索引，保存下来。
                self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                # 调用外部的 _save_best 方法，比对历史最佳记录，判断是否连续多个 Epoch 都没有超过 1% 的实质性提升。
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break
            # print('epoch: {}, holdout mse losses: {}'.format(epoch, holdout_mse_losses))

    def _save_best(self, epoch, holdout_losses):
        #记录在这一轮里，7个模型中有没有任何一个取得了进步？
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            # 从 _snapshots 字典中，拿出第 i 个模型历史最佳的记录。
            # _snapshots[i] 存的是一个元组 (最好成绩出现的epoch, 历史最低误差)。
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        #使用训练时拟合好的 Scaler，将外部传入的 (State, Action) 归一化
        inputs = self.scaler.transform(inputs)
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(device)
            #ret_log_var=False，要求网络输出真实的物理方差，而不是对数方差
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.network_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        #ensemble_mean 和 ensemble_var 的形状恢复为完整的 (7, total_inputs, dim)
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            # 直接原封不动地返回 7 个模型的预测结果，形状为 (7, reward_size, dim)
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


def get_data(inputs_file_path, labels_file_path, num_examples):
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8) / 255.0
        inputs = data.reshape(num_examples, 784)

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_examples)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int8)




