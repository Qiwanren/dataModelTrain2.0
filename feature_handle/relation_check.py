'''
特征相关性检验
    工具介绍：SelectKBest ：根据给定的选择器，选择出前k个与标签最相关的特征 ，sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, *, k=10)
                    参数介绍：1、score_func: 可调用的函数输入两个数组X和y，并返回一对数组（分数，p-value）或带分数的单个数组。默认值为f_classif
                              2、k：int or “all”, optional, 默认=10 要选择的主要特征数（保留前几个最佳特征）。“ all”选项绕过选择，用于参数搜索
    1、卡方检验
    2、斯皮尔蒙检验
    3、

 构建新的特征
    1、PolynomialFeatures
        介绍：使用 sklearn.preprocessing.PolynomialFeatures 这个类可以进行特征的构造，构造的方式就是特征与特征相乘（自己与自己，自己与其他人），这种方式叫做使用多项式的方式。
              例如：有 a、b 两个特征，那么它的 2 次多项式的次数为 [1,a,b,a2,ab,b2]。
        参数介绍：
              degree：控制多项式的次数；
              interaction_only：默认为 False，如果指定为 True，那么就不会有特征自己和自己结合的项，组合的特征中没有 a^2 和 b^2；
              include_bias：默认为 True 。如果为 True 的话，那么结果中就会有 0 次幂项，即全为 1 这一列
        应用：
'''

'''
使用卡方检验选择特征
    1、卡方检验专用于分类算法，用于选择特征
'''
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np

def chosefeaturebycli2(X,y,k):
    from sklearn.feature_selection import SelectKBest, chi2
    print(X.shape)
    # X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    X_new = SelectKBest(chi2, k=k).fit_transform(X, y)
    print(X_new.shape)

def PolynomialFeaturesDemo():
    X = np.arange(6).reshape(3,2)
    print(X)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    res = poly.fit_transform(X)
    print(res)
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    res1 = poly.fit_transform(X)
    print(res1)


if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    #chosefeaturebycli2(X,y,20)
    PolynomialFeaturesDemo()