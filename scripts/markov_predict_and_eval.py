import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
from geobleu.seq_eval import calc_geobleu_single

# ==== 加载数据 ====
with open("data/preprocessed_city_A.pkl", "rb") as f:
    data = pickle.load(f)  # data 是 list，元素是 dict，包含 'X', 'Y', 'uid'

print(f"✅ 加载样本总数：{len(data)}")

# ==== 构建 Markov 转移矩阵 ====
transition_counts = defaultdict(Counter)

for sample in tqdm(data, desc="构建转移概率"):
    traj = sample['X']  # 轨迹是列表，元素格式 (day, t, x, y)
    for i in range(len(traj) - 1):
        cur_pos = (traj[i][2], traj[i][3])     # 当前坐标 (x, y)
        next_pos = (traj[i+1][2], traj[i+1][3])  # 下一坐标 (x, y)
        transition_counts[cur_pos][next_pos] += 1

# ==== 转换为概率矩阵 ====
transition_probs = {}
for cur_pos, counter in transition_counts.items():
    total = sum(counter.values())
    transition_probs[cur_pos] = {k: v / total for k, v in counter.items()}

print(f"✅ 构建完成，共有 {len(transition_probs)} 个位置的转移概率")

# ==== 预测函数 ====
def predict_trajectory(start_point, target_trajectory):
    """
    start_point: (day, t, x, y) 起始点
    target_trajectory: 目标轨迹，用于获取正确的时间序列
    返回与target_trajectory长度相同的预测轨迹
    """
    predicted = []
    current_pos = (start_point[2], start_point[3])  # 当前位置 (x, y)
    
    for i in range(len(target_trajectory)):
        # 使用目标轨迹的时间信息
        target_day = target_trajectory[i][0]
        target_t = target_trajectory[i][1]
        
        # 预测下一个位置
        if current_pos in transition_probs:
            next_pos = max(transition_probs[current_pos].items(), key=lambda x: x[1])[0]
        else:
            next_pos = current_pos  # 如果没有转移信息，保持原位置
        
        # 构建预测点，使用目标轨迹的时间
        predicted_point = (target_day, target_t, next_pos[0], next_pos[1])
        predicted.append(predicted_point)
        
        # 更新当前位置
        current_pos = next_pos
    
    return predicted

# ==== 评估 ====
num_eval = 400
scores = []
sample_results = []  # 存储每个样本的详细结果

print(f"开始评估前 {num_eval} 个样本...")
print("=" * 80)
print(f"{'样本ID':<8} {'UID':<15} {'轨迹长度':<8} {'GEO-BLEU得分':<12} {'状态'}")
print("=" * 80)

for i in range(min(num_eval, len(data))):
    sample = data[i]
    x = sample['X']
    y = sample['Y']
    uid = sample.get('uid', f'sample_{i}')  # 获取用户ID，如果没有则使用样本索引
    
    # 使用X轨迹最后一个点的位置信息作为起点
    start_point = x[-1]
    
    # 预测轨迹，确保时间信息与y一致
    pred = predict_trajectory(start_point, y)
    
    try:
        score = calc_geobleu_single(pred, y)
        scores.append(score)
        status = "✅ 成功"
        
        # 记录样本结果
        sample_result = {
            'sample_id': i,
            'uid': uid,
            'trajectory_length': len(y),
            'score': score,
            'status': 'success'
        }
        sample_results.append(sample_result)
        
        # 输出每个样本的得分
        print(f"{i:<8} {str(uid):<15} {len(y):<8} {score:<12.6f} {status}")
        
    except Exception as e:
        status = f"❌ 失败: {str(e)[:30]}..."
        
        # 记录失败样本
        sample_result = {
            'sample_id': i,
            'uid': uid,
            'trajectory_length': len(y),
            'score': None,
            'status': f'failed: {str(e)}'
        }
        sample_results.append(sample_result)
        
        print(f"{i:<8} {str(uid):<15} {len(y):<8} {'N/A':<12} {status}")
        
        # 调试信息（可选，如果需要详细错误信息）
        if len(pred) > 0 and len(y) > 0:
            print(f"    预测首点：{pred[0]}")
            print(f"    真实首点：{y[0]}")
    
    # 每100个样本显示统计信息
    if (i + 1) % 100 == 0:
        avg_score = sum(scores) / len(scores) if scores else 0
        success_rate = len(scores) / (i + 1) * 100
        print("-" * 80)
        print(f"📊 进度统计 [{i + 1}/{min(num_eval, len(data))}]:")
        print(f"   当前平均分：{avg_score:.6f}")
        print(f"   成功率：{success_rate:.1f}% ({len(scores)}/{i + 1})")
        print("-" * 80)

print("=" * 80)

# ==== 最终结果统计 ====
if scores:
    final_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    
    print(f"\n🎯 最终统计结果：")
    print(f"   总样本数：{min(num_eval, len(data))}")
    print(f"   成功评估：{len(scores)} 个")
    print(f"   失败样本：{min(num_eval, len(data)) - len(scores)} 个")
    print(f"   成功率：{len(scores) / min(num_eval, len(data)) * 100:.1f}%")
    print(f"   平均得分：{final_score:.6f}")
    print(f"   最高得分：{max_score:.6f}")
    print(f"   最低得分：{min_score:.6f}")
    
    # 显示得分分布
    print(f"\n📈 得分分布：")
    score_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    
    for low, high in score_ranges:
        count = sum(1 for s in scores if low <= s < high)
        if count > 0:
            print(f"   [{low:.1f}, {high:.1f}): {count} 个样本 ({count/len(scores)*100:.1f}%)")
    
else:
    print("\n❌ 没有成功评估的样本")

# ==== 保存详细结果到文件（可选） ====
print(f"\n💾 是否保存详细结果到文件？")
save_results = input("输入 'y' 保存到 'markov_evaluation_results.txt': ").lower().strip()

if save_results == 'y':
    with open('markov_evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("Markov Baseline 评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        for result in sample_results:
            f.write(f"样本 {result['sample_id']:03d}: ")
            f.write(f"UID={result['uid']}, ")
            f.write(f"长度={result['trajectory_length']}, ")
            if result['score'] is not None:
                f.write(f"得分={result['score']:.6f}\n")
            else:
                f.write(f"状态={result['status']}\n")
        
        f.write(f"\n总结统计:\n")
        if scores:
            f.write(f"平均得分: {sum(scores)/len(scores):.6f}\n")
            f.write(f"成功率: {len(scores)/len(sample_results)*100:.1f}%\n")
    
    print("✅ 结果已保存到 'markov_evaluation_results.txt'")

# ==== 绘制得分曲线 ====
import matplotlib.pyplot as plt

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建图形
plt.figure(figsize=(12, 6))

# 子图1: 得分曲线
plt.subplot(1, 2, 1)
plt.plot(range(len(scores)), scores, 'b-', alpha=0.7)
plt.title('GEO-BLEU得分曲线 (按样本顺序)')
plt.xlabel('样本序号')
plt.ylabel('GEO-BLEU得分')
plt.grid(True, linestyle='--', alpha=0.5)

# 子图2: 得分直方图
plt.subplot(1, 2, 2)
plt.hist(scores, bins=20, color='g', alpha=0.7, edgecolor='black')
plt.title('GEO-BLEU得分分布')
plt.xlabel('GEO-BLEU得分')
plt.ylabel('样本数量')
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局并保存
plt.tight_layout()
plt.savefig('markov_evaluation_scores.png', dpi=300, bbox_inches='tight')
print("✅ 得分图表已保存到 'markov_evaluation_scores.png'")
