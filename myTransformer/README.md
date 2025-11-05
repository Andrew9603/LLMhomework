# 项目简介
本项目基于手工实现的 Transformer 模型，旨在帮助开发者深入理解 Transformer 的核心组件，包括多头注意力（Multi-Head Attention）、位置编码（Positional Encoding）、残差连接与层归一化（Residual + LayerNorm）、位置前馈网络（Position-wise Feed-Forward Network）等。  

通过在小规模文本建模任务（字符级文本生成、IWSLT 2017 英德翻译任务）上的实验，本项目验证了各组件的功能与必要性，并提供了完整的训练、评估及消融实验流程。  


# 硬件要求

| 配置   | 建议                           |
| ---- | ---------------------------- |
| GPU  | NVIDIA RTX 3060 或更高（6GB+ 显存） |
| CPU  | 8核以上                         |
| 内存   | ≥ 16GB                       |
| 运行时长 | 单次实验约 30~60 分钟（取决于层数与样本量）    |

# 数据集

默认使用 IWSLT2017 英德翻译数据集：
训练集：10000 条样本

验证集：2000 条样本

英文序列长度限制：64

德文序列长度限制：32

文本预处理使用 Hugging Face MarianTokenizer 进行分词和编码。
`load_dataset("iwslt2017", "iwslt2017-en-de")`


# 运行命令
进入项目目录后，执行：
`bash scripts/run.sh`

# 可选参数：

| 参数名            | 默认值  | 说明                         |
| -------------- | ---- | -------------------------- |
| `--seed`       | 42   | 固定随机种子以保证可复现性              |
| `--epochs`     | 10   | 训练轮数                       |
| `--batch_size` | 64   | 批大小                        |
| `--num_layers` | 3    | Encoder/Decoder 层数（支持消融实验） |
| `--limit`      | None | 限制训练样本数，用于快速调试             |

# 实验结果
训练完成后，会在 results/ 目录下自动生成：
train_loss_curves.png —— 各层数配置下的训练曲线
val_loss_curves.png —— 各层数配置下的验证曲线
final_loss_curves.png —— 训练 & 验证曲线并列展示
loss_一层编码器解码器.csv、loss_二层编码器解码器.csv 等 —— 每个模型的 loss 记录
![img.png](img.png)
# 消融实验
本实验比较了不同编码器/解码器层数（1、2、3层）的影响，
保持其他超参数一致，观察收敛速度与最终验证误差的变化。

| 模型层数 | 最终验证 Loss (↓) | 收敛轮数 |
| ---- | ------------- | ---- |
| 1 层  | 3.21          | 6    |
| 2 层  | 2.78          | 8    |
| 3 层  | 2.60          | 9    |
# 复现实验的 Exact 命令
`python train.py --seed 42 --epochs 10 --batch_size 64 --num_layers 3 --limit 20000`



作者：闫本旭
日期：2025年11月

版本：v1.0
