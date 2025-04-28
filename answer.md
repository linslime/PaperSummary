- [LLM 基础](#-LLM-基础)
    - [公式](##-公式)
        - [DPO 公式](###-DPO-公式)
        - [Layer Norm 公式](###-Layer-Norm-公式)
        - [Batch Norm 公式](###-Batch-Norm-公式)
# LLM 基础
## 公式
### DPO 公式

DPO 是一种直接根据偏好数据优化模型的策略，其目标是最大化模型在偏好对上的胜率。

假设：
- $\pi_{\theta}$ 是当前策略（policy），参数为 $\theta$。
- $\pi_{\text{ref}}$ 是参考策略（reference policy）。
- 给定偏好对 $(x, y_{\text{chosen}}, y_{\text{rejected}})$，其中 $y_{\text{chosen}}$ 是偏好选中的响应，$y_{\text{rejected}}$ 是被拒绝的响应。

1. **奖励差异（Reward Difference）**:

DPO 假设奖励与对数概率有关，定义奖励差为：

$$
r = \log \pi_{\theta}(y_{\text{chosen}} \mid x) - \log \pi_{\theta}(y_{\text{rejected}} \mid x)
$$

2. **DPO 损失函数**:

DPO 的优化目标是最大化选中响应相对于拒绝响应的优势，使用二元交叉熵形式定义损失：

$$
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_{\text{chosen}}, y_{\text{rejected}})} \left[ \log \sigma\left( r - \beta \cdot r_{\text{ref}} \right) \right]
$$

其中：
- $\sigma(\cdot)$ 是 sigmoid 函数。
- $\beta$ 是一个平衡当前策略与参考策略差异的超参数。
- $r_{\text{ref}}$ 是参考策略的奖励差：

$$
r_{\text{ref}} = \log \pi_{\text{ref}}(y_{\text{chosen}} \mid x) - \log \pi_{\text{ref}}(y_{\text{rejected}} \mid x)
$$
可以直观理解为：  
让 $\pi_{\theta}$ 在被选中的回答上比被拒绝的回答有更高的概率，同时受参考策略 $\pi_{\text{ref}}$ 的影响进行正则化。

### Softmax 函数公式

Softmax 函数通常用于将一个向量的实数值转换为一个概率分布。

给定一个输入向量 $\mathbf{z} = (z_1, z_2, \dots, z_K)$，Softmax 函数定义为：

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

其中：
- $z_i$ 是输入向量中的第 $i$ 个元素，
- $K$ 是向量的总长度，
- 输出满足归一化性质，即所有输出之和为 $1$。

Softmax 输出性质
- 每个 $\text{Softmax}(z_i) \in (0, 1)$。
- 所有输出值之和满足：
  $$
  \sum_{i=1}^{K} \text{Softmax}(z_i) = 1
  $$

应用场景
- 多分类问题中的概率预测（如分类器最后一层输出）。
- 注意力机制中的权重归一化。


### Layer Norm 公式
___
Layer Normalization 的计算步骤如下：

给定输入向量 $\mathbf{x} = (x_1, x_2, \dots, x_H)$，其中 $H$ 是特征的数量。

1. **计算均值（Mean）**:
   $$
   \mu = \frac{1}{H} \sum_{i=1}^{H} x_i
   $$

2. **计算方差（Variance）**:
   $$
   \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
   $$

3. **归一化（Normalization）**:
   $$
   \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$

4. **缩放和平移（Scale and Shift）**:
   $$
   y_i = \gamma_i \hat{x}_i + \beta_i
   $$

其中：
- $\epsilon$ 是一个很小的正数，防止除零错误。
- $\gamma$ 和 $\beta$ 是可学习的参数，用于恢复模型的表达能力。

最终输出向量为 $\mathbf{y} = (y_1, y_2, \dots, y_H)$。

### Batch Norm 公式
___
Batch Normalization 的计算步骤如下：

给定一批输入样本 $\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)}\}$，其中每个样本 $\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \dots, x_H^{(i)})$，$m$ 是 batch size，$H$ 是特征维度。

对于每个特征维度 $j$：

1. **计算均值（Mean）**:
   $$
   \mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
   $$

2. **计算方差（Variance）**:
   $$
   \sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
   $$

3. **归一化（Normalization）**:
   $$
   \hat{x}_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
   $$

4. **缩放和平移（Scale and Shift）**:
   $$
   y_j^{(i)} = \gamma_j \hat{x}_j^{(i)} + \beta_j
   $$

其中：
- $\epsilon$ 是一个很小的正数，防止除零错误。
- $\gamma_j$ 和 $\beta_j$ 是可学习的缩放与平移参数。

最终输出为一批归一化且经过仿射变换的样本。