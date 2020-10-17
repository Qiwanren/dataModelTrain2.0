# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
# File system manangement
import os
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics

path = 'D:/data/python/test/'
app_train = pd.read_csv(path + 'public.train.csv')
app_test = pd.read_csv(path + 'public.test.csv')

train_id=app_train[['ID']]
test_id=app_test[['ID']]

app_train_test = [app_train, app_test]
app_train_test = pd.concat(app_train_test)

app_train_test=app_train_test.mask(app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
app_train_test=app_train_test.fillna(method='ffill')

app_train_test.to_csv(path +'data_prc.csv',index=False)


app_train= train_id.merge(app_train_test, on='ID', how='left')
app_test= test_id.merge(app_train_test, on='ID', how='left')
app_test=app_test.drop(columns='发电量')

app_train.to_csv(path +'train_prc.csv',index = False)
app_test.to_csv(path +'test_prc.csv',index = False)

# -------------------------------------------------------------------------------------
def add_former_power(features):
    mean_power=features['平均功率']
    former_power=np.zeros(features.shape[0])  # 创建一个特定长度，值为零的数组
    for i in range(1,features.shape[0]):
        former_power[i]=mean_power[i-1]
    former_power[0]=mean_power[0]

    features.insert(1,'former_power',former_power)
    # features.to_csv('data/features_former_power.csv',index=None)

    return features

def add_dis2peak(data):
    peaks_IDs_unchecked = [90, 309, 466, 686, 844, 1046, 1209, 1402, 1597, 1775, 1976, 2163, 2349, 2540,
             2723, 2937, 3104, 3263, 3440, 3646, 3844, 3985, 4178, 4357, 4547, 4745, 4915,
             5175, 5387, 5607, 5803, 6018, 6229, 6443, 6635, 6816, 7017, 7206, 7432, 7633,
             7846, 8018, 8226, 8396, 8589, 8806, 8998, 9169, 9390, 9631, 9807, 10018, 10250,
             10432, 10645, 10825, 10998, 11230, 11428, 11632, 11850, 12054, 12264, 12476,
             12689, 12904, 13102, 13312, 13544, 13708, 13915, 14125, 14317, 14555, 14759,
             14952, 15173, 15399, 15612, 15813, 16036, 16240, 16436, 16651, 16838, 17028,
             17223, 17399, 17645, 17840]

    peak_IDs = []
    peak_index=[]
    ID=data['ID']
    for ID_i in peaks_IDs_unchecked:
        for j, ID_j in enumerate(ID.values):  ##  列举和枚举
            if ID_i >= ID_j and ID_i < ID[j+1]:
                peak_IDs.append(ID_j)
                peak_index.append(j)


    dis2peak = []
    peak_value=[]
    mean_power=data['平均功率']
    for id in data['ID']:
        mindis = np.abs(id - peak_IDs[0])
        peak_row=peak_index[0]
        for i,peak_id in enumerate(peak_IDs):
            if np.abs(id-peak_id)<mindis:
                mindis=np.abs(id-peak_id)
                peak_row=peak_index[i]
        dis2peak.append(mindis)
        peak_value.append(mean_power[peak_row])

    data.insert(1, 'dis2peak', dis2peak )
    data.insert(1, 'peak_value', peak_value )
    return data

def add_power_mean_std(data):
    mean_power=[]
    std_power=[]

    for dis in enumerate(data['dis2peak']):
        mean_power_i=np.mean(data[data['dis2peak']==dis[1]]['平均功率'])
        mean_power.append(mean_power_i)
        std_power_i=np.std(data[data['dis2peak']==dis[1]]['平均功率'])
        std_power.append(std_power_i)
    data.insert(17,'mean_power',mean_power)
    data.insert(17,'std_power',std_power)
    return data

def add_mean_board_temperature(data,T=210):
    board_temperature=data['板温']
    mean_board_temperature=[]
    for i in range(len(board_temperature)):
        temperature_i=[]
        if i<len(board_temperature)-T:
            temperature_i=np.sum(board_temperature[i:i+T])*1.0/T
        else:
            temperature_i=np.sum(board_temperature[i-T:i])*1.0/T
        mean_board_temperature.append(temperature_i)
    data.insert(1,'mean_board_temp',mean_board_temperature)
    return data

def add_wind(data,T=20):
    wind0=list(data['风速']*data['风向'])
    wind=[]
    for i in range(len(data['风向'])):
        wind_i=[]
        if i<T/2:
            wind_i=np.sum(wind0[i:i+T])*1.0/T
        elif i<len(data['风向'])-T/2:
            wind_i=np.sum(wind0[i-int(T/2):i+int(T/2)])*1.0/T
        else:
            wind_i=np.sum(wind0[i-T:i])*1.0/T
        wind.append(wind_i)
    data.insert(17,'wind',wind0)
    return data

def add_I_eta(data):
    data.insert(6,'I_eta',data['光照强度']*data['转换效率'])
    return data

def add_P_eta(data):
    data.insert(16,'P_eta',data['转换效率']*data['平均功率'])
    return data

def add_temp_diff(data):
    data.insert(10,'temp_diff',data['板温']-data['现场温度'])
    return data

def add_idc(data):
    data.insert(11,'idc_A',data['电流A']/(data['转换效率A']+0.001))
    data.insert(11,'idc_B',data['电流B']/(data['转换效率B']+0.001))
    data.insert(11,'idc_C',data['电流C']/(data['转换效率C']+0.001))
    return data

def add_vdc(data):
    data.insert(11,'vdc_A',data['电压A']/(data['转换效率A']+0.001))
    data.insert(11,'vdc_B',data['电压B']/(data['转换效率B']+0.001))
    data.insert(11,'vdc_C',data['电压C']/(data['转换效率C']+0.001))
    return data

def add_temp_diff_light(data):
    data.insert(10,'温差乘以光强',np.abs((data['板温']-data['现场温度'])*data['光照强度']))
    return data

def add_PN_I(data):
    I0=4.215*10**(-14)*np.exp(0.1539*data['板温'])
    q=1.6*10**(-9)
    K=0.86*10**(-4)
    U=data['电压A']
    T=data['板温']+273.15
    Id=I0*(np.exp((q*U)/(K*T))-1)
    data.insert(13,'PN_I',Id*10**16.5 )
    return data

def add_vdc_square(data):
    data.insert(14,'vdc_A_square',(data['电压A']/(data['转换效率A']+0.001))**2)
    data.insert(14,'vdc_B_square',(data['电压B']/(data['转换效率B']+0.001))**2)
    data.insert(14,'vdc_C_square',(data['电压C']/(data['转换效率C']+0.001))**2)
    return data

def add_idc_square(data):
    data.insert(17,'idc_A_square',(data['电流A']/(data['转换效率A']+0.001))**2)
    data.insert(17,'idc_B_square',(data['电流B']/(data['转换效率B']+0.001))**2)
    data.insert(17,'idc_C_square',(data['电流C']/(data['转换效率C']+0.001))**2)
    return data

def add_poly_features(data,column_names):
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))
    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features

def do_feature_project(features):
    print('添加与平均功率峰值的距离和平均功率峰值大小...')
    features=add_dis2peak(features)
    print('添加平均功率的均值和方差...')
    features=add_power_mean_std(features)
    print('正在保存特征...')
    features.to_csv('features.csv',index=None)
    print('特征构造完毕!总特征数量为：',features.shape[1])
    return features
# 基础特征工程
def method01():
    train_data = pd.read_csv(path + 'train_prc.csv')
    train_ID = train_data['ID']
    test_data = pd.read_csv(path + 'test_prc.csv')
    test_ID = test_data['ID']

    features = pd.read_csv(path + 'data_prc.csv')
    # print(features.head(10))

    features = do_feature_project(features)
    # features=pd.read_csv('data/features.csv')

    train_features = features[features['ID'].isin(train_ID)]
    train_features.reset_index(drop=True)
    # train_features.insert(train_features.shape[1],'发电量',train_data['发电量'].values)
    train_data = train_features

    test_data = features[features['ID'].isin(test_ID)]
    test_data.reset_index(drop=True)

    print('正在保存新的训练集和测试集...')
    train_data.to_csv(path + 'train_all_features2.csv', index=None)
    test_data.to_csv(path + 'test_all_features2.csv', index=None)
    print('正在保存新的训练集和测试集...')
    print('保存完毕')


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from numpy import nan as NA

def method02():
    app_train = pd.read_csv(path + 'train_all_features2.csv')
    print('Training data shape: ', app_train.shape)
    app_test = pd.read_csv(path + 'test_all_features2.csv')
    print('Testing data shape: ', app_test.shape)
    train_id = app_train[['ID']]
    test_id = app_test[['ID']]
    app_train_test = [app_train, app_test]
    app_train_test = pd.concat(app_train_test)
    app_train_test = app_train_test.mask(
        app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
    '''may change ffill'''
    app_train_test = app_train_test.fillna(method='ffill')
    app_train = train_id.merge(app_train_test, on='ID', how='left')
    app_test = test_id.merge(app_train_test, on='ID', how='left')
    app_test = app_test.drop(columns='发电量')
    app_train = app_train.drop(columns='peak_value')
    app_test = app_test.drop(columns='peak_value')

    '''a'''
    app_train['理论输出'] = app_train['光照强度'] * app_train['转换效率']
    app_test['理论输出'] = app_test['光照强度'] * app_test['转换效率']
    '''b'''
    app_train['温差'] = app_train['板温'] - app_train['现场温度']
    app_test['温差'] = app_test['板温'] - app_test['现场温度']
    '''c'''
    app_train['实际功率'] = app_train['转换效率'] * app_train['平均功率']
    app_test['实际功率'] = app_test['转换效率'] * app_test['平均功率']
    '''d'''
    # app_train['风力X风向']=app_train['风向']*app_train['风速']
    # app_test['风力X风向']=app_test['风向']*app_test['风速']

    app_train['实际温度'] = app_train['转换效率'] * app_train['现场温度']
    app_test['实际温度'] = app_test['转换效率'] * app_test['现场温度']
    '''开始瞎jb蒙'''
    app_train['电压差A'] = app_train['电流A'] - app_train['电流B']
    app_test['电压差A'] = app_test['电流A'] - app_test['电流B']

    app_train['cde'] = app_train['电压A'] / app_train['转换效率A']
    app_test['cde'] = app_test['电压A'] / app_test['转换效率A']
    app_train['cde1'] = app_train['电压B'] / app_train['转换效率B']
    app_test['cde1'] = app_test['电压B'] / app_test['转换效率B']
    app_train['cde2'] = app_train['电压C'] / app_train['转换效率C']
    app_test['cde2'] = app_test['电压C'] / app_test['转换效率C']

    # app_train['abk']=app_train['abk']*app_train['abk']
    # app_test['abk']=app_test['abk']*app_test['abk']
    # app_train['iuo']=app_train['光照强度']*np.cos((app_train['ID']))
    # app_test['iuo']=app_test['光照强度']*np.cos((app_test['ID']))
    app_train['cdex'] = app_train['cde'] * app_train['cde']
    app_test['cdex'] = app_test['cde'] * app_test['cde']
    app_train['cdex1'] = app_train['cde1'] * app_train['cde1']
    app_test['cdex1'] = app_test['cde1'] * app_test['cde1']
    app_train['cdex2'] = app_train['cde2'] * app_train['cde2']
    app_test['cdex2'] = app_test['cde2'] * app_test['cde2']

    # print(app_train['dis2peak'])
    app_train['C_1'] = app_train['dis2peak'] * app_train['dis2peak']
    app_test['C_1'] = app_test['dis2peak'] * app_test['dis2peak']

    app_train['C_2'] = app_train['dis2peak'] * app_train['光照强度']
    app_test['C_2'] = app_test['dis2peak'] * app_test['光照强度']

    # app_train['C_4']=app_train['功率A']/app_train['风速']
    # app_test['C_4']=app_test['功率A']/app_test['风速']

    # app_train['C_5']=app_train['功率B']/app_train['风速']
    # app_test['C_5']=app_test['功率B']/app_test['风速']
    #
    # app_train['C_6']=app_train['C_4']*app_train['C_4']
    # app_test['C_6']=app_test['C_4']*app_test['C_4']
    # app_train['C']=app_train['电流B']-app_train['电流C']
    # app_test['C']=app_test['电流B']-app_test['电流C']

    # app_train['C_3']=app_train['dis2peak']*app_train['平均功率']
    # app_test['C_3']=app_test['dis2peak']*app_test['平均功率']

    poly_features = app_train[['板温', '现场温度', '光照强度', '风速', '风向']]
    poly_features_test = app_test[['板温', '现场温度', '光照强度', '风速', '风向']]

    # imputer for handling missing values
    #from sklearn.preprocessing import Imputer
    imputer = SimpleImputer(missing_values=NA, strategy="median")

    poly_target = app_train['发电量']

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    poly_transformer = PolynomialFeatures(degree=2)

    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: ', poly_features.shape)

    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))

    # Add in the target
    poly_features['TARGET'] = poly_target

    # Find the correlations with the target
    poly_corrs = poly_features.corr()['TARGET'].sort_values()

    # Display most negative and most positive
    # print(poly_corrs)
    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))
    ''''''

    ''''''
    # Merge polynomial features into training dataframe
    poly_features['ID'] = app_train['ID']
    app_train_poly = app_train.merge(poly_features, on='ID', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['ID'] = app_test['ID']
    app_test_poly = app_test.merge(poly_features_test, on='ID', how='left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

    app_train_poly['发电量'] = poly_target
    app_train = app_train_poly
    app_test = app_test_poly

    # app_train.to_csv('xgb_poly_timetrain.csv',index=False)
    # app_test.to_csv('xgb_poly_timetest.csv',index=False)

    print('Training data with polynomial features shape: ', poly_features.shape)
    print('Testing data with polynomial features shape:  ', poly_features_test.shape)
    print('Training data with polynomial features shape: ', app_train.shape)
    print('Testing data with polynomial features shape:  ', app_test.shape)

    app_train['ID'] = train_id
    app_test['ID'] = test_id

    from sklearn.model_selection import KFold
    import gc
    def my_scorer(y_true, y_predicted, X_test):
        loss_train = np.sum((y_true - y_predicted) ** 2, axis=0) / (X_test.shape[0])  # RMSE
        loss_train = loss_train ** 0.5
        score = 1 / (1 + loss_train)
        return score

    def model(features, test_features, encoding='ohe', n_folds=4):

        # Extract the ids
        train_ids = features['ID']
        test_ids = test_features['ID']

        # Extract the labels for training
        labels = features['发电量']

        # Remove the ids and target
        features = features.drop(columns=['ID', '发电量'])
        test_features = test_features.drop(columns=['ID'])

        # One Hot Encoding
        if encoding == 'ohe':
            features = pd.get_dummies(features)
            test_features = pd.get_dummies(test_features)

            # Align the dataframes by the columns
            features, test_features = features.align(test_features, join='inner', axis=1)

            # No categorical indices to record
            cat_indices = 'auto'

        # Integer label encoding
        elif encoding == 'le':

            # Create a label encoder
            label_encoder = LabelEncoder()

            # List for storing categorical indices
            cat_indices = []

            # Iterate through each column
            for i, col in enumerate(features):
                if features[col].dtype == 'object':
                    # Map the categorical features to integers
                    features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                    test_features[col] = label_encoder.transform(
                        np.array(test_features[col].astype(str)).reshape((-1,)))

                    # Record the categorical indices
                    cat_indices.append(i)

        # Catch error if label encoding scheme is not valid
        else:
            raise ValueError("Encoding must be either 'ohe' or 'le'")

        print('Training Data Shape: ', features.shape)
        print('Testing Data Shape: ', test_features.shape)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np arrays
        features = np.array(features)
        test_features = np.array(test_features)

        # Create the kfold object
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)  # 进行交叉验证

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        # Empty array for test predictions
        test_predictions = np.zeros(test_features.shape[0])
        train_predictions = np.zeros(features.shape[0])

        # Empty array for out of fold validation predictions
        out_of_fold = np.zeros(features.shape[0])

        # Lists for recording validation and training scores
        # valid_scores = []
        train_scores = []

        # Iterate through each fold
        for train_indices, valid_indices in k_fold.split(features):
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]

            # Create the model-work
            model = xgb.XGBRegressor(objective='reg:linear', n_estimators=16000, min_child_weight=1, num_leaves=20,
                                     learning_rate=0.01, max_depth=6, n_jobs=20,
                                     subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)

            # Train the model-work
            model.fit(train_features, train_labels,
                      eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                      early_stopping_rounds=300, verbose=600)

            # Record the best iteration
            best_iteration = 16000

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / k_fold.n_splits

            # Make predictions
            test_predictions += model.predict(test_features) / k_fold.n_splits
            train_predictions += model.predict(features) / k_fold.n_splits
            # Record the out of fold predictions
            out_of_fold = model.predict(valid_features) / k_fold.n_splits

            # Record the best score
            train_score = my_scorer(valid_labels, out_of_fold, valid_features)

            # valid_scores.append(valid_score)
            train_scores.append(train_score)

            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()

        # Make the submission dataframe
        submission = pd.DataFrame({'ID': test_ids, '发电量': test_predictions})
        train_sub = pd.DataFrame({'ID': train_ids, '发电量': train_predictions})
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Overall validation score
        # valid_auc = roc_auc_score(labels, out_of_fold)

        # Add the overall scores to the metrics
        # valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))

        # Needed for creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')

        # Dataframe of validation scores
        metric = pd.DataFrame({'fold': fold_names,
                               'train': train_scores,
                               })

        return submission, feature_importances, metric, train_sub

    submission, fi, metric, train_sub = model(app_train, app_test)
    print('Baseline metrics')
    print(metric)
    xgb2lgb_train = pd.read_csv(path + 'train_all_features2.csv')
    xgb2lgb_test = pd.read_csv(path + 'test_all_features2.csv')
    xgb2lgb_train['xgb发电量'] = train_sub['发电量']
    xgb2lgb_test['xgb发电量'] = submission['发电量']
    xgb2lgb_train.to_csv(path + 'xgb2lgb_train.csv', index=False)
    xgb2lgb_test.to_csv(path + 'xgb2lgb_test.csv', index=False)

def method02_test():
    app_train = pd.read_csv(path + 'train_all_features2.csv')
    print('Training data shape: ', app_train.shape)
    app_test = pd.read_csv(path + 'test_all_features2.csv')
    print('Testing data shape: ', app_test.shape)
    train_id = app_train[['ID']]
    test_id = app_test[['ID']]
    app_train_test = [app_train, app_test]
    app_train_test = pd.concat(app_train_test)
    print(app_train_test.head(5))
    app_train_test = app_train_test.mask(
        app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))
    print('-----------------------------------------------------------')
    print(app_train_test.head(5))
    print(type(app_train_test))
    '''may change ffill'''

    '''
    app_train_test = app_train_test.fillna(method='ffill')
    app_train = train_id.merge(app_train_test, on='ID', how='left')
    app_test = test_id.merge(app_train_test, on='ID', how='left')
    app_test = app_test.drop(columns='发电量')
    app_train = app_train.drop(columns='peak_value')
    app_test = app_test.drop(columns='peak_value')
    '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

def method03():
    import warnings
    warnings.filterwarnings('ignore')
    import lightgbm as lgb
    app_train = pd.read_csv('xgb2lgb_train.csv')
    print('Training data shape: ', app_train.shape)
    app_test = pd.read_csv('xgb2lgb_test.csv')
    print('Testing data shape: ', app_test.shape)

    train_id = app_train[['ID']]
    test_id = app_test[['ID']]
    app_train_test = [app_train, app_test]
    app_train_test = pd.concat(app_train_test)
    app_train_test = app_train_test.mask(
        app_train_test.sub(app_train_test.mean()).div(app_train_test.std()).abs().gt(3))

    app_train_test = app_train_test.fillna(method='ffill')
    app_train = train_id.merge(app_train_test, on='ID', how='left')
    app_test = test_id.merge(app_train_test, on='ID', how='left')
    app_test = app_test.drop(columns='发电量')
    app_train = app_train.drop(columns='peak_value')
    app_test = app_test.drop(columns='peak_value')
    app_train['理论输出'] = app_train['光照强度'] * app_train['转换效率']
    app_test['理论输出'] = app_test['光照强度'] * app_test['转换效率']

    app_train['温差'] = app_train['板温'] - app_train['现场温度']
    app_test['温差'] = app_test['板温'] - app_test['现场温度']

    app_train['实际功率'] = app_train['转换效率'] * app_train['平均功率']
    app_test['实际功率'] = app_test['转换效率'] * app_test['平均功率']

    # app_train['风力X风向']=app_train['风向']*app_train['风速']
    # app_test['风力X风向']=app_test['风向']*app_test['风速']

    app_train['实际温度'] = app_train['转换效率'] * app_train['现场温度']
    app_test['实际温度'] = app_test['转换效率'] * app_test['现场温度']

    app_train['电压差A'] = app_train['电流A'] - app_train['电流B']
    app_test['电压差A'] = app_test['电流A'] - app_test['电流B']

    app_train['cde'] = app_train['电压A'] / app_train['转换效率A']
    app_test['cde'] = app_test['电压A'] / app_test['转换效率A']
    app_train['cde1'] = app_train['电压B'] / app_train['转换效率B']
    app_test['cde1'] = app_test['电压B'] / app_test['转换效率B']
    app_train['cde2'] = app_train['电压C'] / app_train['转换效率C']
    app_test['cde2'] = app_test['电压C'] / app_test['转换效率C']

    # app_train['abk']=app_train['abk']*app_train['abk']
    # app_test['abk']=app_test['abk']*app_test['abk']
    # app_train['iuo']=app_train['光照强度']*np.cos((app_train['ID']))
    # app_test['iuo']=app_test['光照强度']*np.cos((app_test['ID']))
    app_train['cdex'] = app_train['cde'] * app_train['cde']
    app_test['cdex'] = app_test['cde'] * app_test['cde']
    app_train['cdex1'] = app_train['cde1'] * app_train['cde1']
    app_test['cdex1'] = app_test['cde1'] * app_test['cde1']
    app_train['cdex2'] = app_train['cde2'] * app_train['cde2']
    app_test['cdex2'] = app_test['cde2'] * app_test['cde2']

    # print(app_train['dis2peak'])
    app_train['C_1'] = app_train['dis2peak'] * app_train['dis2peak']
    app_test['C_1'] = app_test['dis2peak'] * app_test['dis2peak']

    app_train['C_2'] = app_train['dis2peak'] * app_train['光照强度']
    app_test['C_2'] = app_test['dis2peak'] * app_test['光照强度']

    app_train['C_3'] = app_train['电流B'] - app_train['电流C']
    app_test['C_3'] = app_test['电流B'] - app_test['电流C']

    # app_train['实际温度1']=app_train['转换效率']*app_train['风向']
    # app_test['实际温度1']=app_test['转换效率']*app_test['风向']

    app_train['C_4'] = app_train['功率A'] / app_train['风速']
    app_test['C_4'] = app_test['功率A'] / app_test['风速']

    app_train['C_5'] = app_train['功率B'] / app_train['风速']
    app_test['C_5'] = app_test['功率B'] / app_test['风速']

    app_train['C_6'] = app_train['C_4'] * app_train['C_4']
    app_test['C_6'] = app_test['C_4'] * app_test['C_4']

    # app_train['C_7']=app_train['功率A']*app_train['风向']
    # app_test['C_7']=app_test['功率A']*app_test['风向']

    # app_train['实际温度3']=app_train['功率C']/app_train['风速']
    # app_test['实际温度3']=app_test['功率C']/app_test['风速']
    # app_train['C_9']=app_train['风向']*app_train['转换效率A']
    # app_test['C_9']=app_test['风向']*app_test['转换效率A']

    # app_train['实际温度3']=app_train['功率B']*app_train['风速']
    # app_test['实际温度3']=app_test['功率B']*app_test['风速']

    # app_train['cde5']=app_train['cde4'] *app_train['cde4']
    # app_test['cde5']=app_test['cde4'] * app_test['cde4']

    # app_train['C_3']=app_train['电流C']-app_train['电流A']
    # app_test['C_3']=app_test['电流C']-app_test['电流A']

    poly_features = app_train[['板温', '现场温度', '光照强度', '风速', '风向']]
    poly_features_test = app_test[['板温', '现场温度', '光照强度', '风速', '风向']]

    # imputer for handling missing values
    #imputer = Imputer(strategy='median')
    imputer = SimpleImputer(strategy="median")

    poly_target = app_train['发电量']

    # Need to impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=2)

    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    print('Polynomial Features shape: ', poly_features.shape)

    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))

    # Add in the target
    poly_features['TARGET'] = poly_target

    # Find the correlations with the target
    poly_corrs = poly_features.corr()['TARGET'].sort_values()

    # Display most negative and most positive
    # print(poly_corrs)
    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(['板温', '现场温度', '光照强度', '风速', '风向']))
    ''''''

    ''''''
    # Merge polynomial features into training dataframe
    poly_features['ID'] = app_train['ID']
    app_train_poly = app_train.merge(poly_features, on='ID', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['ID'] = app_test['ID']
    app_test_poly = app_test.merge(poly_features_test, on='ID', how='left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

    app_train_poly['发电量'] = poly_target
    app_train = app_train_poly
    app_test = app_test_poly

    # app_train.to_csv('xgb_poly_timetrain.csv',index=False)
    # app_test.to_csv('xgb_poly_timetest.csv',index=False)

    print('Training data with polynomial features shape: ', poly_features.shape)
    print('Testing data with polynomial features shape:  ', poly_features_test.shape)
    print('Training data with polynomial features shape: ', app_train.shape)
    print('Testing data with polynomial features shape:  ', app_test.shape)

    app_train['ID'] = train_id
    app_test['ID'] = test_id

    # app_train = app_train.drop(['1', '板温_y','现场温度_y', '光照强度_y', '风速_y', '风向_y'],axis=1)
    # app_test = app_test.drop(['1', '板温_y','现场温度_y', '光照强度_y', '风速_y', '风向_y'],axis=1)
    from sklearn.model_selection import KFold
    import gc

    def model(features, test_features, encoding='ohe', n_folds=4):

        # Extract the ids
        train_ids = features['ID']
        test_ids = test_features['ID']

        # Extract the labels for training
        labels = features['发电量']

        # Remove the ids and target
        features = features.drop(columns=['ID', '发电量'])
        test_features = test_features.drop(columns=['ID'])

        # One Hot Encoding
        if encoding == 'ohe':
            features = pd.get_dummies(features)

            test_features = pd.get_dummies(test_features)

            # Align the dataframes by the columns
            features, test_features = features.align(test_features, join='inner', axis=1)

            # No categorical indices to record
            cat_indices = 'auto'

        # Integer label encoding
        elif encoding == 'le':

            # Create a label encoder
            label_encoder = LabelEncoder()

            # List for storing categorical indices
            cat_indices = []

            # Iterate through each column
            for i, col in enumerate(features):
                if features[col].dtype == 'object':
                    # Map the categorical features to integers
                    features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                    test_features[col] = label_encoder.transform(
                        np.array(test_features[col].astype(str)).reshape((-1,)))

                    # Record the categorical indices
                    cat_indices.append(i)

        # Catch error if label encoding scheme is not valid
        else:
            raise ValueError("Encoding must be either 'ohe' or 'le'")

        print('Training Data Shape: ', features.shape)
        print('Testing Data Shape: ', test_features.shape)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np arrays
        features = np.array(features)
        test_features = np.array(test_features)

        # Create the kfold object
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        # Empty array for test predictions
        test_predictions = np.zeros(test_features.shape[0])
        train_predictions = np.zeros(features.shape[0])
        # Empty array for out of fold validation predictions
        out_of_fold = np.zeros(features.shape[0])

        # Lists for recording validation and training scores
        valid_scores = []
        train_scores = []

        # Iterate through each fold
        for train_indices, valid_indices in k_fold.split(features):
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]

            # Create the model-work
            model = lgb.LGBMRegressor(objective='regression', n_estimators=12000, min_child_samples=20, num_leaves=20,
                                      learning_rate=0.005, feature_fraction=0.8,
                                      subsample=0.5, n_jobs=-1, random_state=50)

            # Train the model-work
            model.fit(train_features, train_labels, eval_metric='rmse',
                      eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                      eval_names=['valid', 'train'], categorical_feature=cat_indices,
                      early_stopping_rounds=2000, verbose=600)

            # Record the best iteration
            best_iteration = model.best_iteration_

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / k_fold.n_splits

            # Make predictions
            test_predictions += model.predict(test_features, num_iteration=best_iteration) / k_fold.n_splits
            train_predictions += model.predict(features, num_iteration=best_iteration) / k_fold.n_splits
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict(valid_features, num_iteration=best_iteration) / k_fold.n_splits

            # Record the best score
            valid_score = model.best_score_['valid']['rmse']
            train_score = model.best_score_['train']['rmse']

            valid_scores.append(valid_score)
            train_scores.append(train_score)

            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()

        # Make the submission dataframe
        submission = pd.DataFrame({'ID': test_ids, '发电量': test_predictions})
        train_sub = pd.DataFrame({'ID': train_ids, '发电量': train_predictions})
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Overall validation score
        # valid_auc = roc_auc_score(labels, out_of_fold)

        # Add the overall scores to the metrics
        # valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
        valid_scores.append(np.mean(valid_scores))
        # Needed for creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')

        # Dataframe of validation scores
        metric = pd.DataFrame({'fold': fold_names,
                               'train': train_scores,
                               'valid': valid_scores})

        return submission, feature_importances, metric, train_sub

    submission, fi, metric, train_sub = model(app_train, app_test)
    print('Baseline metrics')
    print(metric)
    lgb2xgb_train = pd.read_csv('train_all_features2.csv')
    lgb2xgb_test = pd.read_csv('test_all_features2.csv')
    lgb2xgb_train['xgb发电量'] = train_sub['发电量']
    lgb2xgb_test['xgb发电量'] = submission['发电量']
    lgb2xgb_train.to_csv('lgb2xgb_train.csv', index=False)
    lgb2xgb_test.to_csv('lgb2xgb_test.csv', index=False)


if __name__ == '__main__':
    ##method02()
    ##method02_test()
    method03()




