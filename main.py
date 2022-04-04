from time import *
from tkinter import *
import tkinter.filedialog
from tkinter.ttk import Combobox
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
LOG_NUM = 0

toop = Tk()
toop.title("软件工程")  # 定义标题
toop.geometry("700x500+200+50")
frame1 = Frame(toop, height=100)  # 选择框架
frame2 = Frame(toop, width=700, height=100)  # 功能框架
frame3 = Frame(toop, width=400, height=300)  # 输出框架
frame4 = Frame(toop, width=400, height=300)  # 日志框架
frame5 = Frame(toop, width=100, height=200)  # 辅助功能框架
frame1.grid(row=0, padx=20, pady=10)
frame2.grid(row=1, padx=50)
frame3.grid(row=2, padx=20)
frame4.grid(row=3, padx=20)
frame5.grid(row=4, padx=20)

path = StringVar()  # 地址存储

#选择文件
def xz():
    m = tkinter.filedialog.askopenfilename()
    if path != '':
        path.set(m)
    return path

# 选择框架
Entry(frame1, width=50, textvariable=path).grid(row=0, column=0, padx=10, pady=10)
btn1 = Button(frame1, text='请选择文件', command=xz)
btn1.grid(row=0, column=1, padx=10, pady=5)
cv = tkinter.StringVar()
com = Combobox(frame1, textvariable=cv)
com.grid(row=0, column=2, padx=10, pady=5)
com["value"] = ("请选择算法", "贪心算法", "动态规划法", "回溯法", "遗传算法")
com.current(0)

# a = input()
# path = "C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\" + a

# 数据处理
def Initial():
    global path
    datas = pd.read_csv(path, sep=' ', header=0)
    datas.columns = ['Weight', 'Value']
    # 提取数据表Weight列
    array1 = pd.to_numeric(datas["Weight"])
    weight = array1.tolist()
    # 提取数据表Value列
    array2 = pd.to_numeric(datas["Value"])
    price = array2.tolist()
    # 提取数据表中背包总量w，物品数量n
    with open(path, "r") as f:
        r = f.readlines()
    str1 = r[0]
    str_list = str1.split()
    w = str_list[0]
    n = str_list[1]
    return w, n, weight, price, datas

# 绘制散点图
def Scatter():
    global path
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x, y = np.loadtxt(path, delimiter=' ', unpack=True, skiprows=1)
    plt.plot(x, y, 'o')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.title('数据散点图')
    plt.show()

# 贪心法
begin_time1 = time()
def Get_data():
    fun1 = Initial()
    C = int(fun1[0])
    item = list(zip(fun1[2], fun1[3]))
    return item, C
def Density(item):
    number = len(item)
    data = np.array(item)
    data_list = [0] * number  # 初始化列表
    for i in range(number):
        data_list[i] = (data[i, 1]) / (data[i, 0])  # 得出性价比列表
    data_set = np.array(data_list)
    idex = np.argsort(-1 * data_set)  # 按降序排列
    return idex
def Greedy(item, C, idex):
    number = len(item)
    status = [0] * number  # 初始化10个元素的列表
    total_weight = 0
    total_value = 0
    for i in range(number):
        if item[idex[i], 0] <= C:
            total_weight += item[idex[i], 0]
            total_value += item[idex[i], 1]
            status[idex[i]] = 1  # 选中的置为1
            C -= item[idex[i], 0]
        else:
            continue
    return total_value, status
def Function1():
    item0, C = Get_data()
    item = np.array(item0)
    idex_Density = Density(item)
    results_Density = Greedy(item, C, idex_Density)
    # print("----------贪心法----------")
    # print("最大价值为：")
    # print(results_Density[0])
    f1=results_Density[0]
    # print("解向量为：")
    # print(results_Density[1])
    end_time1 = time()
    run_time1 = end_time1-begin_time1
    return f1,run_time1
    # print ('该循环程序运行时间：',run_time1)

# 动态规划法
begin_time2 = time()
def bag(n, c, w, v):
    value = [[0 for j in range(c + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            if j < w[i - 1]:
                value[i][j] = value[i - 1][j]
            else:
                value[i][j] = max(value[i - 1][j], value[i - 1][j - w[i - 1]] + v[i - 1])
    return value
def show(n, c, w, value):
    # print("----------动态规划法----------")
    # print('最大价值为:')
    dt=value[n][c]
    # print(value[n][c])
    return dt
    # x = [0 for i in range(n)]
    # j = c
    # for i in range(n, 0, -1):
    #     if value[i][j] > value[i - 1][j]:
    #         x[i - 1] = 1
    #         j -= w[i - 1]
    # print('背包中所装物品为:')
    # for i in range(n):
    #     if x[i]:
    #         print('第', i+1, '个,', end='')

def Function2():
    n = int(Initial()[1])
    c = int(Initial()[0])
    w = list(Initial()[2])
    v = list(Initial()[3])
    price = list(bag(n,c,w,v))
    a=show(n, c, w, price)
    end_time2 = time()
    run_time2 = end_time2 - begin_time2
    # dt=value[n][c]
    # print('该循环程序运行时间：', run_time2)
    return a[0],run_time2

# 性价比非递增排序
def Proportion_Sort():
    global path
    datas = pd.read_csv(path, sep=' ', header=0)
    datas.columns = ['Weight', 'Value']
    datas['Proportion'] = datas.apply(lambda x: x['Value'] / x['Weight'], axis=1)
    datas['Proportion'] = datas['Proportion'].apply(lambda x: round(x, 3))
    datas.sort_values(by='Proportion', inplace=True, ascending=False)
    print(datas)

#回溯法
begin_time3 = time()
fun3 = Initial()
n = int(fun3[1])
c = int(fun3[0])
w = list(fun3[2])
v = list(fun3[3])
maxw = 0
maxv = 0
bag = [0] * n
bags = []
bestbag = None
def conflict(k):
    global bag, w, c
    if sum([y[0] for y in filter(lambda x: x[1] == 1, zip(w[:k + 1], bag[:k + 1]))]) > c:
        return True
    return False
def backpack(k):
    global bag, maxv, maxw, bestbag
    if k == n:
        cv = get_a_pack_value(bag)
        cw = get_a_pack_weight(bag)
        if cv > maxv:
            maxv = cv
            bestbag = bag[:]
        if cv == maxv and cw < maxw:
            maxw = cw
            bestbag = bag[:]
    else:
        for i in [1, 0]:
            bag[k] = i
            if not conflict(k):
                backpack(k + 1)
def get_a_pack_weight(bag):
    global w
    return sum([y[0] for y in filter(lambda x: x[1] == 1, zip(w, bag))])
def get_a_pack_value(bag):
    global v
    return sum([y[0] for y in filter(lambda x: x[1] == 1, zip(v, bag))])
def Function3():
    backpack(0)
    # print("----------回溯法----------")
    print("最大价值为：")
    # print(get_a_pack_value(bestbag))
    b=get_a_pack_value(bestbag)
    # return b
    # print("解向量为：")
    # print(bestbag)
    end_time3 = time()
    run_time3 = end_time3 - begin_time3
    # print('该循环程序运行时间：', run_time3)
    return b,run_time3


N = 500  ##迭代次数
Pc = 0.8  ##交配概率
Pm = 0.15  ##变异概率
## 遗传算法：
# 初始化,N为种群规模，n为染色体长度
def init(N, n):
    C = []
    for i in range(N):
        c = []
        for j in range(n):
            a = np.random.randint(0, 2)
            c.append(a)
        C.append(c)
    return C
def fitness(C, N, n, W, V, w):
    S = []
    F = []
    for i in range(N):
        s = []
        h = 0  # 重量
        f = 0  # 价值
        for j in range(n):
            if C[i][j] == 1:
                if h + W[j] <= w:
                    h = h + W[j]
                    f = f + V[j]
                    s.append(j)
        S.append(s)
        F.append(f)
    return S, F
def best_x(F, S, N):
    y = 0
    x = 0
    B = [0] * N
    for i in range(N):
        if y < F[i]:
            x = i
        y = F[x]
        B = S[x]
    return B, y


## 计算比率
def rate(x):
    p = [0] * len(x)
    s = 0
    for i in x:
        s += i
    for i in range(len(x)):
        p[i] = x[i] / s
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
                X[i][j] = np.random.randint(0, 2)

    return X


# 性价比非递增排序
def Proportion_Sort():
    global path
    datas = pd.read_csv(path, sep=' ', header=0)
    datas.columns = ['Weight', 'Value']
    datas['Proportion'] = datas.apply(lambda x: x['Value'] / x['Weight'], axis=1)
    datas['Proportion'] = datas['Proportion'].apply(lambda x: round(x, 3))
    datas.sort_values(by='Proportion', inplace=True, ascending=False)
    p=datas['Proportion']
    return p


# 日志动态打印：
def pr_log_text(logmsg):
    global LOG_NUM
    current_time = obtain_time()
    logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
    if LOG_NUM <= 7:
        log_txt.insert(END, logmsg_in)
        LOG_NUM = LOG_NUM + 1
    else:
        log_txt.delete(1.0, 2.0)
        log_txt.insert(END, logmsg_in)


# 获取当前时间
def obtain_time():
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return current_time


def clear():
    result_text.delete(1.0, END)
    log_txt.delete(1.0, END)
    px.delete(0, END)


def choose():  # 处理事件
    a0 = []
    a1 = []
    m = Initial()
    for line in m[0]:
        line = line.split(' ')  # 以空格划分
        a0.append(line)
    for line in m[1]:
        line = line.split(' ')
        a1.append(line)

    c = int(a1[0][0])
    num = int(a1[0][1])

    a = np.array(a0)
    a = a.astype(int)
    w = (a[:, 0])
    v = (a[:, 1])
    w = np.array(w)
    v = np.array(v)
    w = w.astype(int)
    v = v.astype(int)
    item = list(zip(w, v))

    if com.get() == "贪心算法":


        tx=Function1()
        # print("求解时间为：")
        # print(time_sum)
        result = "贪心算法：最优解：" + str(tx[0]) + "，" + "求解时间：" + str(tx[1])
        file = open('C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\zrx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.0, '最优解为：')
        result_text.insert(1.8, tx[0])
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, tx[1])
        pr_log_text("使用贪心算法求解成功")

    elif com.get() == "动态规划算法":
        # time_start = timer()
        dt=Function2()
        # print(ret2)
        # time_end = timer()
        # time_sum = time_end - time_start
        # print("求解时间为：")
        # print(time_sum)
        result = "动态规划算法：最优解：" + str(dt[0]) + "，" + "求解时间：" + str(dt[1])
        file = open('C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\zrx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.0, '最优解为：')
        result_text.insert(1.8, dt[0])
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, dt[1])
        pr_log_text("使用动态规划算法求解成功")

    elif com.get() == "回溯法":
        # time_start = timer()
        hs=Function3()
        # print(bestV)
        # time_end = timer()
        # time_sum = time_end - time_start
        # print("求解时间为：")
        # print(time_sum)
        result = "回溯算法：最优解：" + str(hs[0]) + "，" + "求解时间：" + str(hs[1])
        file = open('C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\zrx.txt', 'w')
        file.write(result)
        file.close()
        result_text.delete(1.0, END)
        result_text.insert(1.8, hs[0])
        result_text.insert(2.0, '\n')
        result_text.insert(2.0, '运行时间为：')
        result_text.insert(2.8, hs[1])
        pr_log_text("使用回溯法求解成功")

    # elif com.get() == "遗传算法":
    #     time_start = timer()
    #     n = len(w)
    #     C = init(num, n)
    #     S, F = fitness(C, num, n, w, v, c)
    #     B, y = best_x(F, S, num)
    #     Y = [y]
    #     for i in range(N):
    #         p = rate(F)
    #         C = chose(p, C, num, n)
    #         C = match(C, num, n, Pc)
    #         C = vari(C, num, n, Pm)
    #         S, F = fitness(C, num, n, w, v, c)
    #         B1, y1 = best_x(F, S, num)
    #         if y1 > y:
    #             y = y1
    #         Y.append(y)
    #     print("遗传算法，最大价值为：")
    #     print(y)
    #     time_end = timer()
    #     time_sum = time_end - time_start
    #     print("求解时间为：")
    #     print(time_sum)
    #     result = "遗传算法：最优解：" + str(bestV) + "，" + "求解时间：" + str(time_sum)
    #     file = open('C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\zrx.txt', 'w')
    #     file.write(result)
    #     file.close()
    #     result_text.delete(1.0, END)
    #     result_text.insert(1.0, '最优解为：')
    #     result_text.insert(1.8, y)
    #     result_text.insert(2.0, '\n')
    #     result_text.insert(2.0, '运行时间为：')
    #     result_text.insert(2.8, time_sum)
    #     pr_log_text("使用遗传法求解成功")

# 排序
def sort():
    a0 = []
    descending = []
    # 计算重量与价值的比值
    m = Initial()
    for line in m[0]:
        line = line.split(' ')
        a0.append(line)
    for i in range(len(a0)):
        F0 = int(a0[i][0])
        S0 = int(a0[i][1])
        T0 = F0 / S0
        m = round(T0,3)
        descending.append(m)
    descending.sort(reverse=True)
    print(descending)
    for item in descending:
        px.insert(END, item)
    pr_log_text("按照重量/质量非递增排列成功")
# 功能框架
btn2 = Button(frame2, text='求解问题', command=choose, width=15, height=1)
btn2.grid(row=0, column=0, padx=20, pady=10)
btn3 = Button(frame2, text='绘制散点图', command=Scatter, width=15, height=1)
btn3.grid(row=0, column=3, padx=20, pady=10)
btn4 = Button(frame2, text='重量比排序', command=sort, width=15, height=1)
btn4.grid(row=0, column=5, padx=20, pady=10)
btn5 = Button(frame2, text='上传文件', command=xz, width=15, height=1)
btn5.grid(row=0, column=7, padx=20, pady=10)

# 结果框架
Label(frame3, text="结果：").grid(row=0, column=0)
result_text = Text(frame3, width=90, height=3)  # 结果输出框
result_text.grid(row=1, column=0, columnspan=30, padx=8)
Label(frame3, text="排序结果：").grid(row=2, column=0)
px = Listbox(frame3, width=90, height=5)  # 日志输出框
px.grid(row=3, column=0, columnspan=30, padx=8)
Label(frame4, text="日志: ").grid(row=0, column=0)
log_txt = Text(frame4, width=90, height=7)  # 日志输出框
log_txt.grid(row=1, column=0, columnspan=30, padx=8)

# 退出框架
btn6 = Button(frame5, text='清空', command=clear, width=15, height=1)
btn6.grid(row=0, column=0, padx=20, pady=10)
btn7 = Button(frame5, text='退出', command=toop.quit, width=15, height=1)
btn7.grid(row=0, column=1, padx=20, pady=10)
toop.mainloop()