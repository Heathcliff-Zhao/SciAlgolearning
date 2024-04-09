在旅行商问题（TSP, Traveling Salesman Problem）中，旅行商希望访问$N$个城市，每个城市访问一次，并最终返回出发城市，目标是找到总旅程最短的路径。

### 问题设定

- 假设有一个小规模的TSP问题，即有4个城市，它们的坐标分别是：A(0,0)，B(1,0)，C(1,1)，D(0,1)。
- 目标是找到访问这4个城市的最短路径。

### 模拟退火算法步骤

1. **初始化**：随机生成一个访问城市的顺序作为初始解。
2. **目标函数**：计算路径的总长度。
3. **邻近解的生成**：通过交换路径中两个城市的顺序来生成邻近解。
4. **接受准则**：如果邻近解比当前解更好，就接受它；如果更差，也有一定概率接受，以避免陷入局部最优。
5. **冷却计划**：逐渐降低温度，直至满足停止条件。

下面是使用Python实现的这个TSP问题的模拟退火解决方案：

```python
import numpy as np
import math
import matplotlib.pyplot as plt

# 城市坐标
cities = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])

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
    alpha = 0.99  # 冷却率
    
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

# 绘制最佳路径
plt.figure(figsize=(10, 5))
plt.plot(cities[:, 0], cities[:, 1], 'o', color='red')  # 城市位置
for i in range(len(best_path)):
    plt.plot([cities[best_path[i], 0], cities[best_path[(i+1) % len(best_path)], 0]], 
             [cities[best_path[i], 1], cities[best_path[(i+1) % len(best_path)], 1]], color='blue')  # 路径
plt.show()
```

由于随机性，每次运行可能得到不同的结果，但对于如此小的城市集，很可能每次都能找到最优解。