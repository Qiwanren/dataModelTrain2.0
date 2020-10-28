import time

import pandas as pd

from model_example.dataHandle import getresultData

def readData(path):
    all_params = ['prov_id', 'user_id', 'cust_id', 'product_id', 'area_id', 'device_number', 'innet_months',
                  'service_type', 'cust_sex', 'cert_age',  # 9
                  'total_fee', 'jf_flux', 'fj_arpu', 'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura',
                  'total_times', 'total_nums', 'local_nums',  # 19
                  'roam_nums', 'in_cnt', 'out_cnt', 'in_dura', 'out_dura', 'heyue_flag', 'activity_type',
                  'is_limit_flag', 'product_type', 'is_5g_flag',  # 29
                  'brand', 'brand_flag', 'brand_detail', 'price', 'imei_duration', 'avg_duratioin', 'is_5g_city_flag',
                  'one_city_flag', 'shejiao_active_days',  # 38
                  'shejiao_visit_cnt', 'xinwen_active_days', 'xinwen_visit_cnt', 'shipin_active_days',
                  'shipin_visit_cnt', 'dshipin_active_days',  # 44
                  'dshipin_visit_cnt', 'zhibo_active_days', 'zhibo_visit_cnt', 'waimai_active_days', 'waimai_visit_cnt',
                  'ditudaohang_active_days',  # 50
                  'ditudaohang_visit_cnt', 'luntan_active_days', 'luntan_visit_cnt', 'shouji_shoping_active_days',
                  'shouji_shoping_visit_cnt',  # 55
                  'liulanqi_active_days', 'liulanqi_visit_cnt', 'wenhua_active_days', 'wenhua_visit_cnt',
                  'youxi_active_days', 'youxi_visit_cnt',  # 61
                  'yinyue_active_days', 'yinyue_visit_cnt', 'work_fze_active_days', 'work_fze_visit_cnt',
                  'jinrong_active_days', 'jinrong_visit_cnt',  # 67
                  'app_type_id', 'app_active_days', 'app_visit_dura', 'flag']  # 70

    train = pd.read_csv(filepath_or_buffer=path, sep=",", names=all_params, encoding='utf-8')
    return train

def getIntNumber(x):
    if x != 1.0 and x != 0.0:
        return None
    else:
        strs = str(x).split('.')
        return int(strs[0])

def writeDataToCsv(df, ans, path):
    df['score'] = ans
    # 将结果输出到文件
    df.to_csv(path)

## 处理类别对象，返回整数handleTypeFlag
def handleTypeFlag(df):
    df['flag'] = df['flag'].map(lambda x: getIntNumber(x))
    type_value = df['flag'].value_counts().index[0]
    df['flag'].fillna(type_value, inplace=True)
    return df

# knn算法
from sklearn.neighbors import KNeighborsClassifier

def model_knn(x_train,y_train,x_test,y_test):
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    ann = knn.predict(x_test)
    # record training set accuracy
    print("Training set score:{:.3f}".format(knn.score(x_train, y_train)))  # 精确到小数点后三位
    print("Test set score:{:.3f}".format(knn.score(x_test, y_test)))
    return ann


'''
测试逻辑回归算法
'''
# 逻辑回归算法
# 正则化参数C=1（默认值）的模型在训练集上准确度为78%，在测试集上准确度为77%。
from sklearn.linear_model import LogisticRegression

def model_LogisticRegression(x_train,y_train,x_test,y_test):
    #logreg = LogisticRegression().fit(x_train, y_train)
    #print("Training set score:{:.3f}".format(logreg.score(x_train, y_train)))  # 精确到小数点后三位
    #print("Test set score:{:.3f}".format(logreg.score(x_test, y_test)))

    # 而将正则化参数C设置为100时，模型在训练集上准确度稍有提高但测试集上准确度略降，
    # 说明较少正则化和更复杂的模型并不一定会比默认参数模型的预测效果更好。
    # 所以我们选择默认值C=1
    logreg100 = LogisticRegression(C=100)
    logreg100.fit(x_train, y_train)
    ans = logreg100.predict(x_test)
    print("Training set accuracy:{:.3f}".format(logreg100.score(x_train, y_train)))
    print("Test set accuracy:{:.3f}".format(logreg100.score(x_test, y_test)))
    return ans

# 决策树算法
from sklearn.tree import DecisionTreeClassifier

def model_DecisionTreeClassifier(x_train,y_train,x_test,y_test):
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(x_train, y_train)
    ans = tree.predict(x_test)
    print("Accuracy on training set:{:.3f}".format(tree.score(x_train, y_train)))
    print("Accuracy on test set:{:.3f}".format(tree.score(x_test, y_test)))
    return ans

# 随机森林
# 再用随机森林算法进行研究
from sklearn.ensemble import RandomForestClassifier

def model_RandomForestClassifier(x_train,y_train,x_test,y_test):
    rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
    rf1.fit(x_train, y_train)
    ans = rf1.predict(x_test)
    print("Accuracy on training set:{:.3f}".format(rf1.score(x_train, y_train)))
    print("Accuracy on test set:{:.3f}".format(rf1.score(x_test, y_test)))
    return ans

# 使用支持向量机算法
# SVM模型过拟合比较明显，虽然在训练集中有一个完美的表现，但是在测试集中仅仅有65%的准确度。
from sklearn.svm import SVC
def model_SVC(x_train,y_train,x_test,y_test):
    svc = SVC()
    svc.fit(x_train, y_train)
    ans = svc.predict(x_test)
    print("Accuracy on training set:{:.2f}".format(svc.score(x_train, y_train)))
    print("Accuracy on test set:{:.2f}".format(svc.score(x_test, y_test)))
    return ans

# 接下来使用深度学习算法
# 从结果中我们可以看到，多层神经网络（MLP）的预测准确度并不如其他模型表现的好，这可能是数据的尺度不同造成的。

from sklearn.neural_network import MLPClassifier

def model_MLPClassifier(x_train,y_train,x_test,y_test):
    mlp = MLPClassifier(random_state=42)
    mlp.fit(x_train, y_train)
    print("Accuracy on training set:{:.2f}".format(mlp.score(x_train, y_train)))
    print("Accuracy on test set:{:.2f}".format(mlp.score(x_test, y_test)))

if __name__ == '__main__':
    # 读取数据
    trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
    test_path = 'D:/data/python/work/qwr_woyinyue_basic_result4.txt'

    train_data = readData(trainFilePath)
    test_data = readData(test_path)

    x_train,y_train,train_result =  getresultData(train_data)
    x_test,y_test,test_result = getresultData(test_data)

    print("24小时格式：" + time.strftime("%H:%M:%S"))
    # knn算法
    #ans = model_knn(x_train, y_train, x_test, y_test)
    # 逻辑回归
    #ans = model_LogisticRegression(x_train, y_train, x_test, y_test)

    # 决策树
    #ans = model_DecisionTreeClassifier(x_train, y_train, x_test, y_test)

    # 随机森林
    #ans = model_RandomForestClassifier(x_train, y_train, x_test, y_test)

    # 支持向量机
    ans = model_SVC(x_train, y_train, x_test, y_test)

    # 深度学习
    #model_MLPClassifier(x_train,y_train, x_test, y_test)

    print("24小时格式：" + time.strftime("%H:%M:%S"))

    path = 'D:/data/python/work/result201022.csv'
    writeDataToCsv(test_result, ans, path)