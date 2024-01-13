#Variable Importance를 추출해 중요 변수들을 식별함으로써 모델 해석성을 향상시키고 불필요한 변수의 영향을 줄인다.
#%%1. 데이터 불러오기
import pandas as pd
import numpy as np
import mysql.connector


def fetch_table_data(cursor, table_name):
    query = f'SELECT * FROM {table_name}'
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    return pd.DataFrame(result, columns=columns)

def main():
    # 데이터베이스 연결 정보
    db_config = {
        'host': '****',
        'user': '****',
        'password': '****',
        'database': '****',
        'port': 30575
    }

    # MariaDB 연결
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        #user_features
        user_table_name = 'user_features_dummies'
        user_features = fetch_table_data(cursor, user_table_name)
        print("\nUser Data:")
        print(user_features.head())
        print(user_features.nunique())
        print(user_features.shape)

        #item_features
        item_table_name = 'item_features_dummies'
        item_features = fetch_table_data(cursor, item_table_name)
        print("\nItem Data:")
        print(item_features.head())
        print(item_features.nunique())
        print(item_features.shape)
        
        #ratings
        ratings_table_name = 'ratings_recleaning'
        ratings = fetch_table_data(cursor, ratings_table_name)
        print("\nRatings Data:")
        print(ratings.head())
        print(ratings.nunique())
        print(ratings.shape)
        
    finally:
        # 연결 종료
        cursor.close()
        conn.close()

main()

data = pd.merge(ratings, user_features, on='TRAVEL_ID', how='left')
data = pd.merge(data, item_features, on='new_spot_id', how='left')
data.drop(['TRAVEL_ID','new_spot_id'], axis=1, inplace=True)

#%%2. 학습 데이터를 생성한다.
from sklearn.model_selection import train_test_split
from catboost import Pool

X = data.drop(['rating'], axis=1)
y = data['rating']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#작성해놓은 변수 정보를 불러온 후, 범주형 변수를 식별할 수 있도록 한다.
columns_info = pd.read_excel(r'train_data_catboost/data_columns_info.xlsx')
cat_features = columns_info[columns_info['category'] == 1]['index'].tolist()

#Pool 만들기
train_pool = Pool(X_train, y_train, cat_features=cat_features)
train_pool.get_feature_names()  
X_train.iloc[:,train_pool.get_cat_feature_indices()].columns
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

#%%3. Basic한 Catboost 모델 만들기
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

#%%4. 변수 중요도 추출하기
import matplotlib.pyplot as plt
import numpy as np

feature_importances = model.get_feature_importance()
sorted_indices = np.argsort(feature_importances)[::-1]
feature_importances[sorted_indices]

# Plot 상위 N개의 변수만 표시
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), np.array(model.feature_names_)[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()