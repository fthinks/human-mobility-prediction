import pickle
from collections import Counter
import numpy as np
import os
import sys

# 添加 geobleu 模块路径（根据你的目录结构调整）
geobleu_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../geobleu/geobleu'))
sys.path.append(geobleu_src_path)

from seq_eval import calc_geobleu_single

# 数据路径（根据你实际使用的城市更换）
data_path = 'data/preprocessed_city_A.pkl'

if not os.path.exists(data_path):
    print(f"[错误] 未找到文件：{data_path}")
    exit(1)

# 加载数据
with open(data_path, 'rb') as f:
    samples = pickle.load(f)

print(f"✅ 成功加载样本总数：{len(samples)}")

# 只保留前400个样本用于测试
samples = samples[:400]

scores = []

for idx, sample in enumerate(samples):
    X = sample['X']
    Y = sample['Y']

    recent_coords = [(x, y) for (d, t, x, y) in X if d >= 57]
    if not recent_coords:
        continue

    most_common = Counter(recent_coords).most_common(1)[0][0]
    pred = [most_common for _ in range(len(Y))]

    gt_xy = [(x, y) for (_, _, x, y) in Y]

    pred_seq = [(61 + i // 96, i % 96, x, y) for i, (x, y) in enumerate(pred)]
    gt_seq   = [(61 + i // 96, i % 96, x, y) for i, (x, y) in enumerate(gt_xy)]

    try:
        score = calc_geobleu_single(pred_seq, gt_seq)
        scores.append(score)
    except Exception as e:
        print(f"[警告] 样本 {idx} 评分失败：{e}")

    if idx % 100 == 0:
        print(f"已处理 {idx} / {len(samples)} 个样本，当前平均分：{np.mean(scores):.4f}")

if scores:
    avg_score = np.mean(scores)
    print(f"\n🎯 Baseline GEO-BLEU 平均得分（前400样本）：{avg_score:.4f}")
else:
    print("❌ 没有成功评估任何样本。")
