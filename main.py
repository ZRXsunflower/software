import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import *

print("请输入要处理的数据文件名：")
a = input()
path = "C:\\Users\\86199\\Desktop\\01背包测试数据\\测试数据\\" + a

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

# 性价比非递增排序
def Proportion_Sort():
    global path
    datas = pd.read_csv(path, sep=' ', header=0)
    datas.columns = ['Weight', 'Value']
    datas['Proportion'] = datas.apply(lambda x: x['Value'] / x['Weight'], axis=1)
    datas['Proportion'] = datas['Proportion'].apply(lambda x: round(x, 3))
    datas.sort_values(by='Proportion', inplace=True, ascending=False)
    print(datas)

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
    print("----------贪心法----------")
    print("最大价值为：")
    print(results_Density[0])
    print("解向量为：")
    print(results_Density[1])
    end_time1 = time()
    run_time1 = end_time1-begin_time1
    print ('该循环程序运行时间：',run_time1)


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
    print("----------动态规划法----------")
    print('最大价值为:')
    print(value[n][c])
    x = [0 for i in range(n)]
    j = c
    for i in range(n, 0, -1):
        if value[i][j] > value[i - 1][j]:
            x[i - 1] = 1
            j -= w[i - 1]
    print('背包中所装物品为:')
    for i in range(n):
        if x[i]:
            print('第', i+1, '个,', end='')
def Function2():
    n = int(Initial()[1])
    c = int(Initial()[0])
    w = list(Initial()[2])
    v = list(Initial()[3])
    price = list(bag(n,c,w,v))
    show(n, c, w, price)
    end_time2 = time()
    run_time2 = end_time2 - begin_time2
    print('该循环程序运行时间：', run_time2)

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
    print("----------回溯法----------")
    print("最大价值为：")
    print(get_a_pack_value(bestbag))
    print("解向量为：")
    print(bestbag)
    end_time3 = time()
    run_time3 = end_time3 - begin_time3
    print('该循环程序运行时间：', run_time3)

if __name__ == "__main__":
    print(Initial()[4])
    while(1):
        print("===================================================")
        option = int(input('请选择您的操作:\n1、按性价比进行非递增排序\n2、绘制散点图\n3、贪心算法\n4、动态规划法\n5、回溯法\n6、将结果输出为txt文件'))
        if option == 1:
            Proportion_Sort()
        if option == 2:
            Scatter()
        if option == 3:
            Function1()
        if option == 4:
            Function2()
        if option == 5:
            Function3()

