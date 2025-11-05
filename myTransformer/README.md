# 项目结构


Transformer-NMT/
│
├── src/
│   ├── dataset.py         # 数据加载与编码
│   ├── model.py           # Transformer 模型
│   ├── train.py           # 训练脚本（主程序）
│
├── requirements.txt
├── scripts/
│   └── run.sh             # 训练启动脚本
│
├── results/               # 自动生成训练结果与图表
│
└── README.md

# 硬件要求

| 配置   | 建议                           |
| ---- | ---------------------------- |
| GPU  | NVIDIA RTX 3060 或更高（6GB+ 显存） |
| CPU  | 8核以上                         |
| 内存   | ≥ 16GB                       |
| 运行时长 | 单次实验约 30~60 分钟（取决于层数与样本量）    |

# 数据集

默认使用 IWSLT2017 英德翻译数据集：
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