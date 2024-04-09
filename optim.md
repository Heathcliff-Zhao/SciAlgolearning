使用模拟退火算法来解决一个优化问题：最小化一个多变量函数。这类问题在工程和数据科学领域中非常常见。假设我们想要找到以下函数的全局最小值：

$$f(x, y) = x^2 + y^2 + 10 \sin(x) + 10 \sin(y)$$

这个函数有多个局部最小值，使得寻找全局最小值较为困难，这是模拟退火算法擅长的问题类型。

### 模拟退火算法步骤

1. **初始化**：随机选择一个点作为初始解。
2. **目标函数**：计算给定点的函数值。
3. **邻近解的生成**：在当前点的周围随机选择一个新点。
4. **接受准则**：如果新点的函数值比当前点的函数值低，则接受新点；否则，也有一定概率接受，这个概率取决于两点的函数值差和当前的温度。
5. **冷却计划**：逐渐降低温度，直至满足停止条件。

下面是使用Python实现的模拟退火算法，用于寻找该函数的全局最小值：

```python
import numpy as np

# 定义目标函数
def f(x, y):
    return x**2 + y**2 + 10 * np.sin(x) + 10 * np.sin(y)

# 模拟退火算法
def simulated_annealing_for_function():
    current_temp = 100.0  # 初始温度
    min_temp = 1.0  # 最低温度
    alpha = 0.99  # 冷却率
    
    current_point = np.random.rand(2) * 20 - 10  # 在[-10, 10]范围内随机选择初始点
    current_value = f(*current_point)
    
    while current_temp > min_temp:
        # 在当前点附近随机选择新点
        step = (np.random.rand(2) - 0.5) * 10
        new_point = current_point + step * current_temp / 100
        new_value = f(*new_point)
        
        # 接受准则
        if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / current_temp):
            current_point = new_point
            current_value = new_value
        
        current_temp *= alpha  # 降温
    
    return current_point, current_value

# 执行算法
best_point, best_value = simulated_annealing_for_function()
print(f"Best point: {best_point}")
print(f"Best value: {best_value}")
```

通过模拟退火算法寻找函数 $f(x, y) = x^2 + y^2 + 10 \sin(x) + 10 \sin(y)$ 的全局最小值。这种方法特别适用于那些具有多个局部最小值的复杂函数，传统的梯度下降方法可能会陷入局部最小值。