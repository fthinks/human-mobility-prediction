import pandas as pd
import os
import pickle
from tqdm import tqdm

# 路径
DATA_PATH = 'data/city_A_challengedata.csv'
SAVE_PATH = 'data/preprocessed_city_A.pkl'

# 读取数据
df = pd.read_csv(DATA_PATH)
print("原始数据量：", len(df))

# 按照用户、d（天）、t（时间片）排序
df = df.sort_values(by=['uid', 'd', 't'])

# 分组并构造样本
samples = []
user_group = df.groupby('uid')

for uid, user_df in tqdm(user_group, desc="处理用户"):
    # 以天为单位整理轨迹
    day_group = user_df.groupby('d')
    day_traj = {day: group[['t', 'x', 'y']].values.tolist() for day, group in day_group}
    
    # 判断是否有足够的天数：至少75天
    if len(day_traj) < 75:
        continue
    
    # 构造样本（X: 前60天，Y: 第61~75天）
    days_sorted = sorted(day_traj.keys())
    X = []
    Y = []
    for d in days_sorted[:60]:
        for txy in day_traj[d]:
            X.append((d,) + tuple(txy))  # 加入天数
    for d in days_sorted[60:75]:
        for txy in day_traj[d]:
            Y.append((d,) + tuple(txy))
    
    # 加入样本
    if X and Y:
        samples.append({'uid': uid, 'X': X, 'Y': Y})

print("构造样本数量：", len(samples))

# 保存为 pickle 文件
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(samples, f)

print(f"已保存处理后的数据至：{SAVE_PATH}")
