import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]  # 去掉了label为2，label只能二分，才可以。
n_samples, n_features = X.shape
# 增加噪声特征
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

cv = StratifiedKFold(n_splits=6)  # 导入该模型，后面将数据划分6份
classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)  # SVC模型 可以换作AdaBoost模型试试

# 画平均ROC曲线的两个参数
mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
mean_fpr = np.linspace(0, 1, 100)
cnt = 0
for i, (train, test) in enumerate(cv.split(X, y)):  # 利用模型划分数据集和目标变量 为一一对应的下标
    cnt += 1
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])  # 训练模型后预测每条样本得到两种结果的概率
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

    mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
    mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点

    roc_auc = auc(fpr, tpr)  # 求auc面积
    plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, roc_auc))  # 画出当前分割数据的ROC曲线

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线

mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.2f})'.format(mean_auc), lw=2)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()