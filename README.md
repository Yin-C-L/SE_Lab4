# 实验4：MobileNet & ShuffleNet 实验报告

- 姓名：尹传林
- 学号：24110041036
- 学期：2026 年秋
- 课程：软件工程原理与实践
- 实验页面：https://oucai.club/classes/se/lab04
- 代码仓库：https://github.com/Yin-C-L/SE_Lab4
- 运行环境：Google Colab (GPU, PyTorch)

本仓库对应实验4的两部分代码作业：
1. 1.4 SENet & CBAM：在 ResNet18/CIFAR10 基线中加入 SE 通道注意力模块
2. 2. 代码作业：HybridSN 高光谱图像分类（网络部分自行实现）

---

## 1.4 SENet & CBAM

### 1.4.1 SENet 原理

SENet（Squeeze-and-Excitation Network，Momenta，ImageNet 2017 冠军）是一种通道注意力机制。它认为卷积特征图的各个通道重要性不同，应当让网络自适应地学习每个通道的权重：

1. Squeeze：对每个通道做全局平均池化（GAP），把 H×W 的特征图压缩成一个实数，得到通道描述子。
2. Excitation：经过两个全连接层（先降维再升维，中间 ReLU，最后 Sigmoid），为每个通道生成 0~1 的权重。
3. Reweight：把这个权重乘回原特征图的对应通道，完成通道维度的自适应校准。

代码上对应一个 SEBlock：

```python
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 1.4.2 改动

在原 ResNet18 的 BasicBlock 中，于 `conv2+bn2` 之后、shortcut 相加之前插入 SEBlock：

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)
        ...
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

其余部分（数据增强、dataloader、训练超参、优化器）与原 ResNet18 基线完全一致，只训练 10 个 epoch，方便和基线（wiki 上为 85.96%）对比。

### 1.4.3 实验结果

训练日志：

```
Epoch: 1 Minibatch:     1 loss: 2.342
Epoch: 1 Minibatch:   101 loss: 1.499
Epoch: 1 Minibatch:   201 loss: 1.271
Epoch: 1 Minibatch:   301 loss: 1.149
Epoch: 2 Minibatch:     1 loss: 1.015
...
Epoch: 10 Minibatch:     1 loss: 0.263
Epoch: 10 Minibatch:   101 loss: 0.341
Epoch: 10 Minibatch:   201 loss: 0.166
Epoch: 10 Minibatch:   301 loss: 0.301
Finished Training
```

测试准确率：

```
Accuracy of the network on the 10000 test images: 85.74 %
```

分析：加了 SE 模块后 10 epoch 达到 85.74%，与原基线 85.96% 基本持平。SE 模块只增加了极少量参数（两个 FC 层），但它的收益通常需要更长训练、更大数据量才能体现；本实验只跑 10 epoch，受随机初始化和训练轮数限制，两者在误差范围内。如果把 epoch 加到 30~50，SE 版本一般会比基线高 1~2 个百分点，这也和论文里 SE-ResNet 相对 ResNet 的提升幅度一致。

对应代码文件：`lab04_1_4_SE_ResNet18_CIFAR10.ipynb`

---

## 2. HybridSN 高光谱图像分类

### 2.1 3D 卷积与 2D 卷积的区别

高光谱图像是一个三维数据块：高度 H、宽度 W、光谱波段数 D（Indian Pines 原始 200 维）。

- 2D 卷积：卷积核只在空间 H、W 两个维度上滑动，不涉及光谱维度。它把每个波段当作独立通道处理，适合普通 RGB 图像，但无法建模相邻波段之间的光谱相关性。
- 3D 卷积：卷积核在 H、W、D 三个维度上都滑动，一次卷积同时捕获空间邻域信息和相邻波段的光谱信息，输出仍是一个三维立方体。HybridSN 先用三层 3D 卷积做 spectral-spatial 联合特征提取，再把结果 reshape 成二维特征图接 2D 卷积做进一步空间特征学习，兼顾两者优点且控制了计算量。

### 2.2 网络结构（自行实现部分）

输入 patch：(N, 1, 30, 25, 25)，即 PCA 降到 30 个波段、25×25 空间邻域、1 个通道。

```python
class HybridSN(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridSN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, (7, 3, 3))
        self.conv2 = nn.Conv3d(8, 16, (5, 3, 3))
        self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))
        self.conv4 = nn.Conv2d(32 * 18, 64, (3, 3))
        self.fc1 = nn.Linear(64 * 17 * 17, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = F.relu(self.conv1(x))      # (B, 8, 24, 23, 23)
        out = F.relu(self.conv2(out))    # (B,16, 20, 21, 21)
        out = F.relu(self.conv3(out))    # (B,32, 18, 19, 19)
        out = out.view(-1, 32*18, 19, 19)  # reshape -> (B,576,19,19)
        out = F.relu(self.conv4(out))    # (B,64,17,17)
        out = out.reshape(out.size(0), -1)  # flatten 18496
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out
```

维度变化和实验指导完全一致：3D 卷积后光谱维从 30 压到 18，把 32×18 当作 576 个通道接 2D 卷积，最后 Flatten 成 18496 维向量，接 256→128→16 的全连接分类头，两个全连接层之间都加 0.4 的 Dropout。

### 2.3 数据流程

Indian Pines 数据集原始形状 (145,145,200)，标签 (145,145)。先用 PCA 把 200 个波段压到 30 个主成分，再对每个像素取 25×25 邻域 patch，剔除标签为 0 的背景像素后共 10249 个样本；按 10%/90% 划分训练/测试集（1024 / 9225），最后 transpose 成 PyTorch 3D 卷积要求的 (N, C, D, H, W) 格式：

```
Hyperspectral data shape:  (145, 145, 200)
Label shape:  (145, 145)
Data shape after PCA:  (145, 145, 30)
Data cube X shape:  (10249, 25, 25, 30)
Data cube y shape:  (10249,)
Xtrain shape:  (1024, 25, 25, 30)
Xtest  shape:  (9225, 25, 25, 30)
after transpose: Xtrain shape:  (1024, 1, 30, 25, 25)
after transpose: Xtest  shape:  (9225, 1, 30, 25, 25)
```

训练超参：Adam，lr=0.001，batch_size=128，共 100 epoch。

### 2.4 实验结果

训练后期 loss 已收敛到接近 0：

```
[Epoch: 95]   [loss avg: 2.9196]   [current loss: 0.0021]
[Epoch: 96]   [loss avg: 2.8898]   [current loss: 0.0007]
[Epoch: 97]   [loss avg: 2.8619]   [current loss: 0.0036]
[Epoch: 98]   [loss avg: 2.8346]   [current loss: 0.0162]
[Epoch: 99]   [loss avg: 2.8069]   [current loss: 0.0201]
[Epoch: 100]  [loss avg: 2.7807]   [current loss: 0.0176]
Finished Training
```

测试集 classification_report：

```
              precision    recall  f1-score   support

         0.0     0.8864    1.0000    0.9398        39
         1.0     0.9816    0.9045    0.9415      1298
         2.0     0.9372    0.9603    0.9486       731
         3.0     0.9796    0.8930    0.9343       215
         4.0     0.9398    0.9776    0.9583       447
         5.0     0.9373    0.9865    0.9613       667
         6.0     1.0000    0.1923    0.3226        26
         7.0     0.9954    0.9977    0.9966       438
         8.0     0.7895    0.9375    0.8571        16
         9.0     0.9260    0.9600    0.9427       874
        10.0     0.9816    0.9683    0.9749      2208
        11.0     0.9297    0.9192    0.9244       532
        12.0     0.8634    0.9568    0.9077       185
        13.0     0.9323    0.9982    0.9641      1117
        14.0     0.9494    0.9193    0.9341       347
        15.0     0.9683    0.7176    0.8243        85

    accuracy                         0.9536      9225
   macro avg     0.9373    0.8931    0.8958      9225
weighted avg     0.9549    0.9536    0.9527      9225
```

整体 OA = 95.36%，和实验参考 96.53% 处于同一水平（差异来自随机种子和小类样本数极少，比如第 6 类只有 26 个测试样本、recall 仅 0.19，拉低了 macro avg）。逐像素预测的分类效果图在本地 Colab notebook 最后一格用 spectral.imshow 渲染。

对应代码文件：`lab04_2_HybridSN_IP.ipynb`

---

## 3. 思考题

### 3.1 训练 HybridSN 每次结果不一样，为什么？

主要有以下随机来源：

1. 网络权重的初始化是随机的（默认 kaiming 初始化）。
2. `DataLoader` 在 `shuffle=True` 时每个 epoch 打乱顺序不同。
3. Dropout 在训练时随机失活神经元。
4. 虽然 `train_test_split` 固定了 random_state=345，但 GPU 上 cuDNN 的卷积/归约操作本身是非确定性的。
5. 训练集只有 1024 个样本（10%），数据量小，这些随机因素被放大，所以每次 OA 可能在 94%~97% 之间波动。要复现需要同时固定 torch、numpy、CUDA 的随机种子并开启 deterministic 模式。

### 3.2 如何进一步提升高光谱图像分类性能？

可以从几个方向入手：

- 网络结构：在 3D-2D 卷积之间加入 SE/CBAM 注意力，或使用更深的 HybridSN、ResNet 式残差连接；
- 训练策略：加入学习率衰减（StepLR/CosineAnnealing）、Weight Decay、Label Smoothing；用 AdamW 代替 Adam；
- 数据：增大训练集比例或对 patch 做翻转/旋转数据增强；PCA 主成分数从 30 调到更优值；
- 正则化：调整 Dropout 比例、加入 Early Stopping 防止过拟合；
- 后处理：对预测图做 MRF / 全连接 CRF 平滑；多模型投票集成；
- 也可以换用更新的 3D-2D 混合网络（如 SSRN、SpectralNet）。

### 3.3 depth-wise conv 和分组卷积的区别与联系？

联系：两者都是分组卷积思想，目的是减少参数量和计算量。

区别：

- 分组卷积（Grouped Conv）：把输入通道分成 g 组，每组用 k 个独立卷积核，输出通道数 = g×k。MobileNetV1 之前 AlexNet/ResNeXt 用的就是分组卷积。
- depth-wise 卷积是分组卷积的极端情况：令 g = 输入通道数，即每个通道单独用一个卷积核卷积，输出通道数 = 输入通道数，不做通道间融合。
- depth-wise 卷积本身不融合跨通道信息，所以 MobileNet 在它后面接一个 1×1 的 pointwise 卷积来组合通道，二者合起来就是 depthwise separable convolution，这是 MobileNetV1/V2 的核心。

### 3.4 SENet 的注意力是不是可以加在空间位置上？

可以。SENet 只建模了通道维度的注意力，CBAM（Convolutional Block Attention Module）就在 SE 的通道注意力之后并联了一个空间注意力模块：沿通道维做 max-pooling 和 avg-pooling，得到两个 H×W 的二维图，拼接后用 7×7 卷积生成单通道空间权重图，再 Sigmoid 乘回原特征。这样同时关注"哪些通道重要"和"空间上哪些位置重要"。在高光谱、医学图像等任务上加入空间注意力通常还能再涨点。

### 3.5 ShuffleNet 中通道 shuffle 如何用代码实现？

分组卷积会让不同组的特征信息完全隔离，梯度和信息无法跨组流动。通道 shuffle 的做法是把卷积输出的维度 reshape 成 (batch, g, n/g, h, w)，在组维内做转置打乱，再 flatten 回去：

```python
def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x
```

这样下一层分组卷积的输入通道就来自不同组，实现了跨组信息流通。

---

## 复现说明

- 两个 notebook 都在 Google Colab 上运行，运行时选 GPU。
- 数据由 notebook 第一个 cell 自动从 GitHub 镜像下载 Indian_pines_corrected.mat / Indian_pines_gt.mat。
- 关键超参：CIFAR10 实验 lr=1e-3、batch=128、10 epoch；HybridSN 实验 lr=1e-3、batch=128、100 epoch、patch=25、PCA=30。
