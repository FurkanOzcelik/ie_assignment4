import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def func_to_calculate(x):
    return 10 + 0.01 * x - 0.1 * x**2 + 0.8 * np.cos(3*x)


a = 0.53
b = 1.56
x = (a+b)/2
epsilon = 0.0001
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
table = pd.DataFrame(columns=['Point A', 'Point B', 'Point X', 'Value X', 'Point x+epsilon','Value x+epsilon'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

f_x = np.linspace(0.53,1.56,100)

# the function, which is y = x^2 here
y = 10 + 0.01 * f_x - 0.1 * f_x**2 + 0.8 * np.cos(3*f_x)
# plot the function
plt.plot(f_x,y, 'r')

while b - a > epsilon:
    x = (a+b)/2
    s1 = func_to_calculate(x+epsilon)
    s2 = func_to_calculate(x)
    table = table.append({'Point A':a, 'Point B':b, 'Point X':x, 'Value X':s2, 'Point x+epsilon':x+epsilon, 'Value x+epsilon':s1}, ignore_index=True)
    if s1 <= s2:
        a = x
    else:
        b = x

    plt.scatter(x, s2, marker=',', s=15, color="darkblue")
print(table)
print(a)
print(b)
print(x)
print(func_to_calculate(x))


plt.show()