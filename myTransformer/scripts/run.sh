
# run.sh: 一键运行训练和生成结果

set -e  # 遇到错误停止

echo "Step 1: 安装依赖..."
pip install -r ../requirements.txt

echo "Step 2: 开始训练..."
python src/train.py

echo "Step 3: 完成训练，结果保存在 ../results/"
