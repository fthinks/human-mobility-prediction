import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('data/city_A_challengedata.csv', sep=',')

# 打印列名和前几行内容
print("列名：", df.columns.tolist())
print(df.head())

# 简单统计用户数
print("总用户数：", df['uid'].nunique())

# 轨迹点统计
print("总轨迹点数：", len(df))

# 每用户轨迹长度分布
user_lengths = df.groupby('uid').size()
print(user_lengths.describe())

# 画出几个用户的轨迹散点图
sample_uids = df['uid'].unique()[:3]
for uid in sample_uids:
    user_data = df[df['uid'] == uid]
    plt.scatter(user_data['x'], user_data['y'], s=2)
    plt.title(f'User {uid} Trajectory')
    plt.xlabel('X grid')
    plt.ylabel('Y grid')
    plt.show()
