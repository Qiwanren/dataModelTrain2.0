from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
import pandas as pd

from model_example.dataHandle import getresultData2

#X,y = load_iris(return_X_y=True)
#X_df = pd.DataFrame(X,columns=list("ABCD"))

X_df,y,result = getresultData2()
# 使用卡方检验选择特征
(chi2,pval) = chi2(X_df,y)

dict_feature = {}
for i,j in zip(X_df.columns.values,chi2):
    dict_feature[i]=j
#对字典按照values排序
ls = sorted(dict_feature.items(),key=lambda item:item[1],reverse=True)

print(ls)