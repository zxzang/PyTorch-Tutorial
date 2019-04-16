"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
numpy
"""
import torch
import numpy as np
import random
from commctrl import RB_BEGINDRAG

# details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)

data = [[-1], [-2], [1], [2]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor),      # [1 2 1 2]
    '\ntorch dim: ', tensor.dim() 
)

# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 32-bit floating point
# print(
#     '\nabs',
#     '\ntorch: ', torch.abs(tensor.t()),      # [1 2 1 2]
#     '\ntorch dim: ', tensor.t().dim() 
# )

# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor),      # [1 2 1 2]
    '\ntorch dim: ', tensor.dim() 
)

# sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
# 矩阵相乘， 矩阵大小需满足： (i, n)x(n, j)
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor),   # [[7, 10], [15, 22]]
    '\ntorch2: ', tensor.mm(tensor)          # [[7, 10], [15, 22]]
)
# incorrect method
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]] 矩阵相乘
    #'\ntorch: ', tensor.dot(tensor)     # this will convert tensor to [1,2,3,4], you'll get 30.0
    # 新版本中(>=0.3.0), 关于 tensor.dot() 有了新的改变, 它只能针对于一维的数组.  向量点乘(内积)、数量积、标量积 dot product; scalar product
    
    '\ntorch: ', torch.dot(tensor.view(4), tensor.view(4)), # tensor(30.) 向量点乘(内积)
    '\ntorch2: ', torch.dot(tensor.resize((4)), tensor.resize((4))),  # tensor(30.) 向量点乘(内积)
    # https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension
)

# 对应点相乘，x.mul(y) ，即点乘操作，点乘不求和操作，又可以叫作Hadamard product
print('\nmatrix Hadamard product')
data2 = [[1,2], [3,4], [5, 6]]
tensor2 = torch.FloatTensor(data2)
print('\ntorch: ', tensor2.mul(tensor2) )

print(tensor)
print(tensor2)
# print('\ntorch2: ', tensor2.mm(tensor) )
print('\ntorch222: ', torch.mm(tensor2, tensor) )
# print('\ntorch2: ', tensor.mm(tensor2) ) # 错误，tensor与tensor2位置颠倒

data = [[1,2], [3,4]]
tensorA = torch.FloatTensor(data)  # 32-bit floating point
data = [[5,6], [7,8]]
tensorB = torch.FloatTensor(data)  # 32-bit floating point 
print(id(tensorB))          
def ChangeTensor( tensorA ):

    tensorA.copy_(tensorB)  #会改变
    #tensorA = tensorB.clone()    #不变
    #tensorA = tensorB        #不变

ChangeTensor(tensorA)
print (tensorA)
print(id(tensorA))
#print (tensorB)     

# random.randint(a, b)，用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b
# print (random.randint(20, 10) )  # 该语句是错误的。下限必须小于上限

# random.sample(sequence, k)，从指定序列中随机获取指定长度的片断。sample函数不会修改原有序列
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(list, 5)  # 从list中随机获取5个元素，作为一个片断返回
#slice = random.sample(list, 0)  # 返回[]
print (slice)

print('\n')

data = [[1,2], [3,4], [5,6]]
tensorA = torch.FloatTensor(data)  # 32-bit floating point
print(tensorA.tolist()[1])  



lists = [[] for i in range(3)]  # 创建的是多行三列的二维列表
for i in range(3):
    lists[0].append(i)
for i in range(5):
    lists[1].append(i)
for i in range(7):
    lists[2].append(i)
print("\nlists is:", lists)
# lists is: [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6]]

for i in lists:
    # i.reverse()
    for j in i:
        j += 1   # 无效  
        
for i in lists:
    # i.reverse()
    for j in range(len(i)):
        i[j] += 1   # 有效  
        
print("lists is:", lists)



x = 6
def func():
    global x #定义外部的x
    x = 1

func()
print (x)


print("\n")
a=[1,-1,2]
b=[1,-1,0]
c=[-1,1,0]
print(sorted(b) == sorted(c))
print(c)

aveWeightVector = [0.0] * (10)
print(aveWeightVector)

print("\n")
a = [[1,2],[3,4]]
print(np.array(a).shape)
print('grad_weight type:', type(a) )

a = {1: 2, 2: 2}
b = {1: 1, 3: 3}
#b.update(a)
b[2]=5
print (b)

print("\n")

print (random.randint(20, 20))

ra = np.random.randn(11)/np.sqrt(10)
print (ra )
ra[1:] = ra[1:] /2.0
print (ra )

print("\n")
#rb = np.zeros((2, 3))
rb = np.zeros(2)
print (rb )
print("数组的维度数目", rb.ndim) 
print("数组元素总数：", rb.size)
print('shape:', rb.shape)

def add(a, b):
    return a+b
print (add(1,2))

