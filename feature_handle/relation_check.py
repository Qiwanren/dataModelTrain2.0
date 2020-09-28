'''
特征相关性检验
    工具介绍：SelectKBest ：根据给定的选择器，选择出前k个与标签最相关的特征 ，sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, *, k=10)
                    参数介绍：1、score_func: 可调用的函数输入两个数组X和y，并返回一对数组（分数，p-value）或带分数的单个数组。默认值为f_classif
                              2、k：int or “all”, optional, 默认=10 要选择的主要特征数（保留前几个最佳特征）。“ all”选项绕过选择，用于参数搜索
    1、卡方检验
    2、斯皮尔蒙检验
    3、
'''

'''
使用卡方检验选择特征
    1、卡方检验专用于分类算法，用于选择特征
'''
from sklearn.datasets import load_digits

def chosefeaturebycli2(X,y,k):
    from sklearn.feature_selection import SelectKBest, chi2
    print(X.shape)
    # X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    X_new = SelectKBest(chi2, k=k).fit_transform(X, y)
    print(X_new.shape)


if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    chosefeaturebycli2(X,y,20)