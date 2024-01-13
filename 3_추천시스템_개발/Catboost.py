#%% Load Data and Make data for train catboost
import pandas as pd
import numpy as np

user_features = pd.read_excel('train_data_catboost/user_features.xlsx')
user_features.columns
user_features.nunique()

item_features = pd.read_excel('train_data_catboost/item_features_target.xlsx')
item_features.columns
item_features.drop(['spot_id', 'VISIT_AREA_NM', 'rename', 'address_name',
                    'category_name', 'place_name', 'place_url', 'x', 'y',
                    'category_name_1','category_name_2','category_name_3', 
                    'category_name_4','category_name_5'], axis=1, inplace=True)
item_features.head()
item_features.nunique()
item_features.shape
item_features.drop_duplicates(inplace=True)
item_features.shape

ratings = pd.read_excel('train_data_catboost/ratings_recleaning.xlsx')
ratings.columns
ratings.head()
ratings.shape

data = pd.merge(ratings, user_features, on='TRAVEL_ID', how='left')
data = pd.merge(data, item_features, on='new_spot_id', how='left')
data.info()
data.shape
data.isna().sum().sum()

data.drop(['TRAVEL_ID','new_spot_id'], axis=1, inplace=True)
data.shape

#rating의 분포를 그려본다.
import matplotlib.pyplot as plt
plt.hist(data['rating'], bins=10)
plt.show()

pd.Series(data.columns).to_excel('train_data_catboost/data_columns_info_api.xlsx')
#%% Make training data
from sklearn.model_selection import train_test_split
from catboost import Pool

data.columns
data.shape
X = data.drop(['rating'], axis=1)
y = data['rating']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#load column information
columns_info = pd.read_excel('train_data_catboost/data_columns_info.xlsx')
columns_info.head()
columns_info.shape
X_train.shape
X_valid.shape

#index of categorical value
cat_features = columns_info[columns_info['category'] == 1]['index'].tolist()
X_train.iloc[:,cat_features].columns
X_train.iloc[:,cat_features].nunique().to_excel('train_data_catboost/check_category.xlsx')

#Make Pool
train_pool = Pool(X_train, y_train, cat_features=cat_features)
train_pool.get_feature_names()  
X_train.iloc[:,train_pool.get_cat_feature_indices()].columns

valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

#%%Basic Model
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=500,
                        learning_rate=0.05,
                        loss_function='RMSE',
                        eval_metric='RMSE',
                        use_best_model = True,
                        custom_metric = ['R2', 'MAE'])

model.fit(train_pool,
        plot = True,
        verbose = False,
        eval_set = valid_pool)

#%% Hyperparameter tuning using hyperopt
from hyperopt import hp, fmin, tpe
import numpy as np
from sklearn.metrics import mean_squared_error

def hyperopt_objective(params):
    print(params)
    params['iterations'] = int(params['iterations'])  # iterations 값을 정수로 변환
    model = CatBoostRegressor(**params, random_seed=42, task_type="CPU")
    model.fit(train_pool, verbose=0, eval_set=valid_pool)
    y_pred = model.predict(valid_pool)
    rmse = mean_squared_error(valid_pool.get_label(), y_pred, squared=False)
    return rmse

space = {
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  # 0.01에서 1 사이의 로그 균일 분포
    'iterations': hp.quniform('iterations', 100, 2000, 1),  # 100에서 2000 사이의 균일 분포 (정수 값)
    'depth': hp.choice('depth', [3, 4, 5, 6, 7, 8, 9, 10]),  # 가능한 깊이 값들 중 하나를 선택
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),  # 1에서 10 사이의 균일 분포
    'border_count': hp.choice('border_count', [32, 64, 128, 254])  # 가능한 border_count 값들 중 하나를 선택
}

best = fmin(hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

# best 값을 실제 값으로 변환
best['depth'] = [3, 4, 5, 6, 7, 8, 9, 10][best['depth']]
best['border_count'] = [32, 64, 128, 254][best['border_count']]
best['iterations'] = int(best['iterations'])

best_params = best.copy()
print(best_params)

#%% Estimate model performance using Cross-Validation
from catboost import cv
print(best_params)

best_params['loss_function'] = 'RMSE'
best_params['custom_metric'] = ['R2', 'MAE']

cv_data = cv(
    params = best_params,
    pool = Pool(X, label=y, cat_features=cat_features),
    fold_count=10,
    shuffle=True,
    partition_random_seed=0,
    plot=True,
    stratified=False,
    verbose=False
)

mean_rmse = cv_data['test-RMSE-mean'].iloc[-1]
mean_mae = cv_data['test-MAE-mean'].iloc[-1]
mean_r2 = cv_data['test-R2-mean'].iloc[-1]

print("Average RMSE:", mean_rmse)
print("Average MAE:", mean_mae)
print("Average R2:", mean_r2)

#%%Make final model
final_model = CatBoostRegressor(**best_params, random_seed=42)
final_model.fit(X, y, cat_features=cat_features, verbose=False, plot=True)

#%% Extract variable importance
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
feature_importances = final_model.get_feature_importance()

# Sort the feature importances in descending order and get the indices
sorted_indices = np.argsort(feature_importances)[::-1]
feature_importances[sorted_indices]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), np.array(final_model.feature_names_)[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

#%% Save the model
final_model.save_model('final_model.cbm')