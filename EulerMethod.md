欧拉方法是求解常微分方程（ODEs）的最简单数值方法之一。它基于在给定点处的斜率来预测函数在下一点的值。对于一阶微分方程 $\frac{dy}{dt} = f(t, y)$，欧拉方法的迭代公式为：

$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$

其中 $y_{n+1}$ 是下一个点的函数值，$y_n$ 是当前点的函数值，$h$ 是步长，$f(t_n, y_n)$ 是在点 $(t_n, y_n)$ 处微分方程的斜率。

使用欧拉方法来求解之前的微分方程 $\frac{dy}{dt} = -y$，取初始条件 $y(0) = 1$，并在 $t = 0$ 到 $t = 5$ 的区间内求解，步长 $h$ 取为 0.01。

下面是使用Python实现的欧拉方法：

```python
def f(t, y):
    return -y  # 定义微分方程

def euler_step(t, y, h):
    return y + h * f(t, y)  # 欧拉方法的迭代步

# 初始条件
y0 = 1
t0 = 0
tf = 5
h = 0.01  # 步长

# 时间和解向量
t_values = [t0]
y_values = [y0]

# 使用欧拉方法求解ODE
t = t0
y = y0
while t < tf:
    y = euler_step(t, y, h)
    t += h
    t_values.append(t)
    y_values.append(y)

# 打印最后几个值看看在t=5时的解
print("t-values:", t_values[-5:])
print("y-values:", y_values[-5:])
```

这段代码通过迭代地应用欧拉方法，计算了在每个时间步长 $h$ 下函数的近似值，并存储了这些值以便于后续的分析或绘图。由于欧拉方法是一种显式方法，可能在某些问题上不够稳定，特别是当步长 $h$ 较大时。