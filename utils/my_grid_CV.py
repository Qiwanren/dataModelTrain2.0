'''
获取模型评估指标
    输入参数：
        1、model:模型算法实例
        2、params : 需要验证匹配的超参数  tree_pipe_params = {'max_depth':[1,3,5,7]}，多个参数之间用逗号分隔
        3、样本数据集 X
        4、类别字段 Y

'''
from sklearn.model_selection import GridSearchCV


def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,  # 要搜索的模型
                        params,  # 要尝试的参数
                        error_score=0,  # 如果报错，则结果为零
                        cv=5
                        )
    # 管道设计
    '''
    mean_impute = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                            ('classify', knn)
                            ])
    grid = GridSearchCV(mean_impute, knn_params)
    '''

    grid.fit(X, y)  # 拟合模型和参数
    # 经典的性能指示
    print('  Best Accuracy : {}'.format(grid.best_score_))
    # 得到最佳准确率的最佳参数
    print(' Best Parameters : {}'.format(grid.best_params_))
    # 拟合的平均时间（秒）
    print(" Average Time to Fit (s) ：{}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    # 预测的平均时间 （秒）,从该指标可以看出模型在真实世界的性能
    print(" Average Time to Score (s) : {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))