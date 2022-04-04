import numpy as np
import random
import matplotlib.pyplot as plt
##初始化,N为种群规模，n为染色体长度
def init(N,n):
    C = []
    for i in range(N):
        c = []
        for j in range(n):
            a = np.random.randint(0,2)
            c.append(a)
        C.append(c)
    return C


##评估函数
# x(i)取值为1表示被选中，取值为0表示未被选中
# w(i)表示各个分量的重量，v（i）表示各个分量的价值，w表示最大承受重量
def fitness(C,N,n,W,V,w):
    S = []##用于存储被选中的下标
    F = []## 用于存放当前该个体的最大价值
    for i in range(N):
        s = []
        h = 0  # 重量
        f = 0  # 价值
        for j in range(n):
            if C[i][j]==1:
                if h+W[j]<=w:
                    h=h+W[j]
                    f = f+V[j]
                    s.append(j)
        S.append(s)
        F.append(f)
    return S,F

##适应值函数,B位返回的种族的基因下标，y为返回的最大值
def best_x(F,S,N):
    y = 0
    x = 0
    B = [0]*N
    for i in range(N):
        if y<F[i]:
            x = i
        y = F[x]
        B = S[x]
    return B,y

## 计算比率
def rate(x):
    p = [0] * len(x)
    s = 0
    for i in x:
        s += i
    for i in range(len(x)):
        p[i] = x[i] / (s+1)
    return p

## 选择
def chose(p, X, m, n):
    X1 = X
    r = np.random.rand(m)
    for i in range(m):
        k = 0
        for j in range(n):
            k = k + p[j]
            if r[i] <= k:
                X1[i] = X[j]
                break
    return X1

##交配
def match(X, m, n, p):
    r = np.random.rand(m)
    k = [0] * m
    for i in range(m):
        if r[i] < p:
            k[i] = 1
    u = v = 0
    k[0] = k[0] = 0
    for i in range(m):
        if k[i]:
            if k[u] == 0:
                u = i
            elif k[v] == 0:
                v = i
        if k[u] and k[v]:
            # print(u,v)
            q = np.random.randint(n - 1)
            # print(q)
            for i in range(q + 1, n):
                X[u][i], X[v][i] = X[v][i], X[u][i]
            k[u] = 0
            k[v] = 0
    return X

##变异
def vari(X, m, n, p):
    for i in range(m):
        for j in range(n):
            q = np.random.rand()
            if q < p:
                X[i][j] = np.random.randint(0,2)

    return X

m = 8##规模
N = 500  ##迭代次数
Pc = 0.8 ##交配概率
Pm = 0.05##变异概率
V =[89, 59, 19, 43, 100, 72, 44, 16, 7, 64]
W =[95, 75, 23, 73, 50, 22, 6, 57, 89, 98]
n = len(W)##染色体长度
w = 388

C = init(m, n)
S,F  = fitness(C,m,n,W,V,w)
B ,y = best_x(F,S,m)
Y =[y]
for i in range(N):
    p = rate(F)
    C = chose(p, C, m, n)
    C = match(C, m, n, Pc)
    C = vari(C, m, n, Pm)
    S, F = fitness(C, m, n, W, V, w)
    B1, y1 = best_x(F, S, m)
    if y1 > y:
        y = y1
    Y.append(y)
print("最大值为：",y)

plt.plot(Y)
plt.show()
