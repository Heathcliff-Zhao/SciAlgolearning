import numpy as np
import matplotlib.pyplot as plt

# 城市坐标
cities = np.array(
    [[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556]]
)

# 计算两城市间的距离
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# 计算路径长度
def total_distance(path):
    return sum(distance(cities[path[i]], cities[path[(i+1) % len(path)]]) for i in range(len(path)))

# 生成邻近解：随机交换两个城市的顺序
def get_neighbor(path):
    a, b = np.random.choice(len(path), 2, replace=False)
    path[a], path[b] = path[b], path[a]
    return path

# 模拟退火算法
def simulated_annealing(cities):
    current_temp = 100.0  # 初始温度
    min_temp = 1.0  # 最低温度
    alpha = 0.5  # 冷却率
    
    current_path = np.random.permutation(cities.shape[0])  # 初始解
    current_distance = total_distance(current_path)
    
    while current_temp > min_temp:
        neighbor = get_neighbor(current_path.copy())
        neighbor_distance = total_distance(neighbor)
        
        if neighbor_distance < current_distance or np.random.rand() < np.exp((current_distance - neighbor_distance) / current_temp):
            current_path = neighbor
            current_distance = neighbor_distance
        
        current_temp *= alpha
    
    return current_path, current_distance

# 解决TSP问题
best_path, best_distance = simulated_annealing(cities)
print("Best path:", best_path)
print("Best distance:", best_distance)

import itertools

def naive_solution(cities):
    # 生成所有可能的路径
    all_possible_paths = list(itertools.permutations(range(cities.shape[0])))
    
    # 计算每条路径的总距离
    all_distances = [total_distance(path) for path in all_possible_paths]
    
    # 找到总距离最短的路径
    best_path_index = np.argmin(all_distances)
    best_path = all_possible_paths[best_path_index]
    best_distance = all_distances[best_path_index]
    
    return best_path, best_distance

# 暴力解法
best_path_naive, best_distance_naive = naive_solution(cities)
print("Best path (naive):", best_path_naive)
print("Best distance (naive):", best_distance_naive)

# 绘制优化路径
plt.figure(figsize=(10, 5))
plt.plot(cities[:, 0], cities[:, 1], 'o', color='red')  # 城市位置
for i in range(len(best_path)):
    plt.plot([cities[best_path[i], 0], cities[best_path[(i+1) % len(best_path)], 0]], 
             [cities[best_path[i], 1], cities[best_path[(i+1) % len(best_path)], 1]], color='blue')  # 路径
    
# 绘制最佳路径
for i in range(len(best_path_naive)):
    plt.plot([cities[best_path_naive[i], 0], cities[best_path_naive[(i+1) % len(best_path_naive)], 0]], 
             [cities[best_path_naive[i], 1], cities[best_path_naive[(i+1) % len(best_path_naive)], 1]], color='green')  # 路径
    
plt.title('Traveling Salesman Problem')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
# plt.show()
plt.savefig('TSP.png')

