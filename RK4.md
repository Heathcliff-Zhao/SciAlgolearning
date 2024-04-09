求解以下一阶常微分方程作为例子：

$$\frac{dy}{dt} = f(t, y) = ay$$

其中，$a$是一个常数。我们知道这个方程的解是指数函数，但现在将通过RK4方法来数值求解。

RK4方法的一般公式为：

$$
\begin{align*}
k_1 &= f(t_i, y_i) \\
k_2 &= f(t_i + \frac{h}{2}, y_i + \frac{h}{2}k_1) \\
k_3 &= f(t_i + \frac{h}{2}, y_i + \frac{h}{2}k_2) \\
k_4 &= f(t_i + h, y_i + hk_3) \\
y_{i+1} &= y_i + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) \\
\end{align*}
$$

其中，$h$是步长，$t_i$是当前的时间，$y_i$是当前的$y$值，$f(t, y)$是给定的微分方程。

用RK4方法求解上述微分方程，假设$a = -1$，初始条件为$y(0) = 1$，求解从$t = 0$到$t = 5$的解。

```python
def f(t, y, a=-1):
    return a * y

def rk4_step(t, y, h, a=-1):
    k1 = f(t, y, a)
    k2 = f(t + h/2, y + h/2 * k1, a)
    k3 = f(t + h/2, y + h/2 * k2, a)
    k4 = f(t + h, y + h * k3, a)
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Initial conditions
y0 = 1
t0 = 0
tf = 5
h = 0.01  # Step size

# Time and solution vectors
t_values = [t0]
y_values = [y0]

# Solving the ODE
t = t0
y = y0
while t < tf:
    y = rk4_step(t, y, h)
    t += h
    t_values.append(t)
    y_values.append(y)

# Let's print the last few values to see the solution at t = 5
print("t-values:", t_values[-5:])
print("y-values:", y_values[-5:])
```

选择的步长$h$影响求解的精度，步长越小，求解结果越精确，计算量也相应增加。