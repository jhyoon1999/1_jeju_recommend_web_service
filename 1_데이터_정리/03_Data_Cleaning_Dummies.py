import pandas as pd
import numpy as np
import mysql.connector

#%% 1. DB로부터 user_features, item_features, ratings 데이터를 추출

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
        user_table_name = 'user_features'
        user_features = fetch_table_data(cursor, user_table_name)
        print("\nUser Data:")
        print(user_features.head())
        print(user_features.nunique())
        print(user_features.shape)

        #item_features
        item_table_name = 'item_features'
        item_features = fetch_table_data(cursor, item_table_name)
        print("\nItem Data:")
        print(item_features.head())
        print(item_features.nunique())
        print(item_features.shape)
        
        #ratings
        ratings_table_name = 'ratings'
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

#%%2. 다중 선택 변수 더미화하기
#다중 선택 변수를 더미화한다.
#아래는 특정 예시이다.

#(1). 예시1 : 여행 미션 변수
mission = user_features[['TRAVEL_ID', 'TRAVEL_MISSION_1', 'TRAVEL_MISSION_2', 'TRAVEL_MISSION_3',
                        'TRAVEL_MISSION_4', 'TRAVEL_MISSION_5', 'TRAVEL_MISSION_6',
                        'TRAVEL_MISSION_7', 'TRAVEL_MISSION_8', 'TRAVEL_MISSION_9',
                        'TRAVEL_MISSION_10', 'TRAVEL_MISSION_11', 'TRAVEL_MISSION_12',
                        'TRAVEL_MISSION_13', 'TRAVEL_MISSION_14', 'TRAVEL_MISSION_15',
                        'TRAVEL_MISSION_16', 'TRAVEL_MISSION_17', 'TRAVEL_MISSION_18',
                        'TRAVEL_MISSION_19', 'TRAVEL_MISSION_20', 'TRAVEL_MISSION_21']]

mission_melted = mission.melt(id_vars=['TRAVEL_ID'],
                            value_vars=['TRAVEL_MISSION_1', 'TRAVEL_MISSION_2', 'TRAVEL_MISSION_3',
                                        'TRAVEL_MISSION_4', 'TRAVEL_MISSION_5', 'TRAVEL_MISSION_6',
                                        'TRAVEL_MISSION_7', 'TRAVEL_MISSION_8', 'TRAVEL_MISSION_9',
                                        'TRAVEL_MISSION_10', 'TRAVEL_MISSION_11', 'TRAVEL_MISSION_12',
                                        'TRAVEL_MISSION_13', 'TRAVEL_MISSION_14', 'TRAVEL_MISSION_15',
                                        'TRAVEL_MISSION_16', 'TRAVEL_MISSION_17', 'TRAVEL_MISSION_18',
                                        'TRAVEL_MISSION_19', 'TRAVEL_MISSION_20', 'TRAVEL_MISSION_21'],
                            value_name='TRAVEL_MISSION')
mission_melted.drop(['variable'], axis = 1, inplace=True)
mission_melted.columns = ['TRAVEL_ID', 'TRAVEL_MISSION']

#missing은 드랍한다.
mission_melted = mission_melted[mission_melted['TRAVEL_MISSION'] != 'missing']

#원핫 인코딩을 실시한다.
mission_dummies = pd.get_dummies(mission_melted, columns=['TRAVEL_MISSION']).groupby('TRAVEL_ID').sum().reset_index()

#user_features에 합친다.
user_features.drop(['TRAVEL_MISSION_1', 'TRAVEL_MISSION_2', 'TRAVEL_MISSION_3',
                    'TRAVEL_MISSION_4', 'TRAVEL_MISSION_5', 'TRAVEL_MISSION_6',
                    'TRAVEL_MISSION_7', 'TRAVEL_MISSION_8', 'TRAVEL_MISSION_9',
                    'TRAVEL_MISSION_10', 'TRAVEL_MISSION_11', 'TRAVEL_MISSION_12',
                    'TRAVEL_MISSION_13', 'TRAVEL_MISSION_14', 'TRAVEL_MISSION_15',
                    'TRAVEL_MISSION_16', 'TRAVEL_MISSION_17', 'TRAVEL_MISSION_18',
                    'TRAVEL_MISSION_19', 'TRAVEL_MISSION_20', 'TRAVEL_MISSION_21'],
                axis=1, inplace=True)

user_features = pd.merge(user_features, mission_dummies, on='TRAVEL_ID', how='left')

#(2). 예시2 : 여행 동기 변수
motive = user_features[['TRAVEL_ID', 'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3']]

motive_melted = motive.melt(id_vars=['TRAVEL_ID'],
                            value_vars=['TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3'],
                            value_name='TRAVEL_MOTIVE')
motive_melted.drop(['variable'], axis = 1, inplace=True)

motive_melted = motive_melted[motive_melted['TRAVEL_MOTIVE'] != 'missing']

motive_dummies = pd.get_dummies(motive_melted, columns=['TRAVEL_MOTIVE']).groupby('TRAVEL_ID').sum().reset_index()
user_features.drop(['TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3'],
                axis=1, inplace=True)
user_features = pd.merge(user_features, motive_dummies, on='TRAVEL_ID', how='left')

#%% 3. DB에 적재하기
db_config = {
    'host': '****',
    'user': '****',
    'password': '****',
    'database': '****',
    'port': 30575
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

user_features.to_sql('user_features_dummies', conn, if_exists='replace', index=False)
item_features.to_sql('item_features_dummies', conn, if_exists='replace', index=False)
ratings.to_sql('ratings_recleaning', conn, if_exists='replace', index=False)