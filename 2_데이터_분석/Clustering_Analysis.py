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
        
    finally:
        # 연결 종료
        cursor.close()
        conn.close()

main()

#%%2. user_features 클러스터링
#데이터를 구성하는 모든 변수가 범주형 변수
#K-modes는 범주형 변수 간의 거리를 측정하여 클러스터를 형성하는 알고리즘
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#(1). TRAVEL_ID 드랍
user_features_target = user_features.drop(columns='TRAVEL_ID')

#(2). 클러스터링 수 찾기
#첫번째)Elbow Method
cost = []
for k in range(1, 10):
    kmode = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)
    kmode.fit_predict(user_features_target)
    cost.append(kmode.cost_)

# Elbow method 그래프
plt.plot(range(1, 10), cost, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.show()

#두번째) Silhouette Score
#더미 변수가 아닌 변수를 더미화한다.
user_features_target_dummies = pd.get_dummies(user_features_target[['GENDER', 'AGE_GRP', 'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_STYL_1','TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6',]], drop_first=True)
user_features_target_dummies.nunique()
user_features_target_dummies.head()
user_features_target_dummies.shape

user_features_target_dummies_combine = pd.concat([user_features_target_dummies,
                                                user_features_target.drop(['GENDER', 'AGE_GRP', 'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_STYL_1','TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6'], axis=1)], axis=1)
user_features_target_dummies_combine.head()
user_features_target_dummies_combine.shape
user_features_target_dummies_combine.nunique()

# StandardScaler를 사용하여 변수 스케일링
scaler = StandardScaler()
user_features_target_scaled = pd.DataFrame(scaler.fit_transform(user_features_target_dummies_combine), columns=user_features_target_dummies_combine.columns)
user_features_target_scaled

# 여러 클러스터 수에 대해 Silhouette Score 계산
silhouette_scores = []
k_values = range(2, 8)

for k in k_values:
    kmode = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
    clusters = kmode.fit_predict(user_features_target_scaled)
    silhouette_avg = silhouette_score(user_features_target_scaled, clusters)
    silhouette_scores.append(silhouette_avg)

# Silhouette Score를 그래프로 시각화
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.show()

#(3). k=4로 클러스터링 실시
# 한글 폰트 경로 설정
font_path = "font/H2PORL.TTF"

# 폰트 매니저를 통해 폰트 설정
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# K-Mode 모델 생성
k = 4  # 클러스터의 개수
km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)

# 모델 피팅
clusters = km.fit_predict(user_features_target)

# 클러스터링 결과를 데이터프레임에 추가
user_features_target['Cluster'] = clusters
user_features_target['Cluster'].unique()

# 클러스터별 분포 시각화
plt.figure(figsize=(12, 6))
for col in list(user_features_target.columns):
    plt.subplot(2, 3, list(user_features_target.columns).index(col)+1)
    sns.countplot(x=col, hue='Cluster', data=user_features_target)
    plt.title(f'{col} Distribution by Cluster')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=270,fontsize=8)

plt.tight_layout()
plt.show()

#%%3. item_features 클러스터링
#범주형 변수 뿐만 아니라 연속형 변수도 존재
#K-prototypes는 K-means와 K-modes의 결합으로, 범주형 변수와 수치형 변수 모두를 고려하여 클러스터링을 수행
from kmodes.kprototypes import KPrototypes

item_features.columns
item_features_target = item_features.copy()
item_features_target.drop(['new_spot_id', 'spot_id', 'VISIT_AREA_NM', 'rename', 'address_name',
                            'category_name', 'place_name', 'place_url', 'x', 'y', 'category_name_1',
                            'category_name_2', 'category_name_3', 'category_name_4','category_name_5'], axis=1, inplace=True)

item_features_target.nunique()
item_features_target.columns

numeric_columns = ['RESIDENCE_TIME_MIN', 'DGSTFN', 'REVISIT_INTENTION','RCMDTN_INTENTION']
item_features_target_categorical = item_features_target.drop(numeric_columns, axis=1)
item_feautures_target_numeric = item_features_target[numeric_columns]

item_features_target_categorical.nunique()
item_feautures_target_numeric.nunique()

#(1). 적절한 k값 찾기 : Silhouette Score
scaler = StandardScaler()
item_feautures_target_numeric_scaled = pd.DataFrame(scaler.fit_transform(item_feautures_target_numeric), columns=item_feautures_target_numeric.columns)

item_feautures_target_combined = pd.concat([item_features_target_categorical, item_feautures_target_numeric_scaled], axis=1)

# 여러 클러스터 수에 대해 Silhouette Score 계산
silhouette_scores = []
k_values = range(2, 8)  # 예시로 2부터 8까지의 클러스터 수를 확인

for k in k_values:
    kproto = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=0)
    clusters = kproto.fit_predict(item_feautures_target_combined, categorical=[0, 1])
    silhouette_avg = silhouette_score(item_feautures_target_combined, clusters)
    silhouette_scores.append(silhouette_avg)

# Silhouette Score를 그래프로 시각화
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.show()

#(2) k=2로 클러스터링
k = 2
kproto = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=1)
clusters = kproto.fit_predict(item_feautures_target_combined, categorical=[0, 1])




















