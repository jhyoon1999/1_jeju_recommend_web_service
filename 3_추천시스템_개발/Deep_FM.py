import pandas as pd
import tensorflow as tf
import tqdm

#%% Load Data and column information
data = pd.read_excel(r'train_data\train_data.xlsx')
data.shape
data.info()
data.head()
data.nunique()

column_info = pd.read_excel(r'train_data\column_info.xlsx')
column_info.shape
column_info.head()
column_info.nunique()

target_columns = ['user', 'item', 'label'] + column_info['column_name'].tolist()
print(target_columns)
data = data[target_columns]
data.info()
data.shape

#%% Make Explicit Feedback to Implicit Feedback
data['label'] = data['label'].apply(lambda x : 1 if x == 5 else 0)
data['label'].value_counts()

#%%Data Preprocessing
from libreco.data import random_split
from libreco.data import DatasetFeat
from libreco.evaluation import evaluate

train_set, eval_set = random_split(data, test_size= 0.1, seed = 42)
train_set.nunique()

#변수 정보 지정
user_col = list(column_info[column_info['feature1'] == 'user_col']['column_name'])
item_col = list(column_info[column_info['feature1'] == 'item_col']['column_name'])

sparse_col = list(column_info[column_info['feature2'] == 'sparse_col']['column_name'])
dense_col = list(column_info[column_info['feature2'] == 'dense_col']['column_name'])
multi_sparse_1 = list(column_info[column_info['feature2'] == 'multi_sparse_1']['column_name'])
multi_sparse_2 = list(column_info[column_info['feature2'] == 'multi_sparse_2']['column_name'])
multi_sparse_3 = list(column_info[column_info['feature2'] == 'multi_sparse_3']['column_name'])

len(user_col) + len(item_col) == len(sparse_col) + len(dense_col) + len(multi_sparse_1) + len(multi_sparse_2) + len(multi_sparse_3)

train_data, data_info = DatasetFeat.build_trainset(train_data=train_set,
                                                    user_col= user_col,
                                                    item_col = item_col,
                                                    sparse_col= sparse_col,
                                                    dense_col= dense_col,
                                                    multi_sparse_col= [multi_sparse_1, multi_sparse_2, multi_sparse_3],
                                                    pad_val= ["missing", "missing", "missing"])
#build testset
eval_data = DatasetFeat.build_evalset(eval_set)

#%%Build hyperparameter dataframe
from sklearn.model_selection import ParameterGrid
param_grid = {
    'loss_type': ["cross_entropy", "focal"],
    'embed_size': [16, 32, 50, 100, 150],
    'lr' : [1e-3, 1e-4, 1e-5],
    'drop_rate' : [0.0, 0.1, 0.2, 0.3],
    'n_epochs' : [10,30,50],
    'multi_sparse_combiner' : ['normal', 'sqrtn']
}

collectofgrid = []
for params in ParameterGrid(param_grid):
    collectofgrid.append(params)
grid_result = pd.DataFrame(collectofgrid)
grid_result.head()
grid_result.shape

metrics = [
        "loss",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc"
    ]

import numpy as np
grid_result['loss'] = np.NaN
grid_result['balanced_accuracy'] = np.NaN
grid_result['roc_auc'] = np.NaN
grid_result['pr_auc'] = np.NaN
grid_result.head()

#%% HyperParameter Tuning
from libreco.algorithms import DeepFM

for i in range(grid_result.shape[0]) :
    print(f'{i}/{grid_result.shape[0]}')
    parameter_info = grid_result.iloc[i]

    tf.compat.v1.reset_default_graph() # reset the default computational graph

    model = DeepFM(
        "ranking",
        data_info,
        loss_type=parameter_info['loss_type'],
        embed_size=parameter_info['embed_size'],
        n_epochs=parameter_info['n_epochs'],
        lr=parameter_info['lr'],
        lr_decay=False,
        reg=None,
        batch_size=256,
        use_bn=True,
        dropout_rate=parameter_info['drop_rate'],
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
        multi_sparse_combiner=parameter_info['multi_sparse_combiner']
    )

    model.fit(
        train_data,
        neg_sampling=False,
        verbose = 2,
        shuffle=True,
        eval_data = eval_data,
        metrics = metrics
    )

    result = evaluate(model = model, data = eval_data, 
            neg_sampling=False, metrics=metrics)

    grid_result.loc[i, 'loss'] = result['loss']
    grid_result.loc[i, 'balanced_accuracy'] = result['balanced_accuracy']
    grid_result.loc[i, 'roc_auc'] = result['roc_auc']
    grid_result.loc[i, 'pr_auc'] = result['pr_auc']