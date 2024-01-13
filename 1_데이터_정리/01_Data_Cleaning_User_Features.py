import pandas as pd
import numpy as np
import mysql.connector

#%%1. DB로부터 Raw Data 불러오기

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
        # 여행 데이터 불러오기
        travel_table_name = 'travel'
        travel = fetch_table_data(cursor, travel_table_name)
        print("Travel Data:")
        print(travel.head())
        print(travel.nunique())
        print(travel.shape)

        # 여행객 Master 데이터 불러오기
        traveler_table_name = 'traveler_master'
        traveler = fetch_table_data(cursor, traveler_table_name)
        print("\nTraveler Data:")
        print(traveler.head())
        print(traveler.nunique())
        print(traveler.shape)

        # code 데이터 불러오기
        code_table_name = 'tc_codeb'
        code = fetch_table_data(cursor, code_table_name)
        print("\nCode Data:")
        print(code.head())
        print(code.nunique())
        print(code.shape)

    finally:
        # 연결 종료
        cursor.close()
        conn.close()

main()

#%%2. 데이터 정리

#(1).여행기간 변수 만들기 : 여행시작일과 여행종료일 사이의 기간을 변수로 만들기
from datetime import datetime

travel['TRAVEL_START_YMD'] = travel['TRAVEL_START_YMD'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
travel['TRAVEL_END_YMD'] = travel['TRAVEL_END_YMD'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
travel.info()

travel['TRAVEL_RANGE'] = travel['TRAVEL_END_YMD'] - travel['TRAVEL_START_YMD']
travel['TRAVEL_RANGE'].value_counts()

def change_datetime_to_str(x):
    x = int(x.days)
    return f'{x}박 {x+1}일' 

travel['TRAVEL_RANGE'] = travel['TRAVEL_RANGE'].apply(change_datetime_to_str)

#(2). 다중변수 분해하기 : 한 셀에 다중선택이 모두 기록된 경우(Ex.한 셀에 1;21;22;24;27;로 기록)
#Primary key인 TRAVEL_ID 변수를 이용해 정리
mission = travel[['TRAVEL_ID','TRAVEL_MISSION']]
mission.set_index('TRAVEL_ID', inplace=True)

mission = mission['TRAVEL_MISSION'].str.split(';')
mission = mission.explode()
mission = mission.str.strip()
mission = mission[mission != '']
mission = mission.reset_index()

#code 정보를 이용해 각 숫자의 의미 삽입하기
target_code = code[code['cd_a'] == 'MIS']
target_code = target_code[['cd_b','cd_nm']]

mission = pd.merge(mission, target_code, left_on= 'TRAVEL_MISSION',right_on='cd_b', how='left')
mission.drop(['cd_b', 'TRAVEL_MISSION'], axis=1, inplace=True)
mission.columns = ['TRAVEL_ID', 'TRAVEL_MISSION']

#분해하기
mission_copy = mission.copy()
mission_copy['count'] = mission_copy.groupby('TRAVEL_ID').cumcount() + 1

mission_copy_pivot = mission_copy.pivot_table(index='TRAVEL_ID', columns='count', values='TRAVEL_MISSION', aggfunc='first')
mission_copy_pivot.columns = [f'TRAVEL_MISSION_{i}' for i in mission_copy_pivot.columns]
mission_copy_pivot.reset_index(inplace=True)

#NA를 "missing"으로 바꾸고, 원데이터에 붙이기
mission_copy_pivot.fillna('missing', inplace=True)

travel.drop('TRAVEL_MISSION', axis=1, inplace=True)
travel = pd.merge(travel, mission_copy_pivot, on='TRAVEL_ID', how='left')


#(3). 다중변수 분해하기 : 여러 셀로 나누어 다중선택이 기록된 경우(Ex.여행동기 : 'TRAVEL_MOTIVE_1','TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3')
motive = traveler[['TRAVELER_ID','TRAVEL_MOTIVE_1','TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3']]
motive.set_index('TRAVELER_ID', inplace=True)
motive.columns = ['TRAVEL_MOTIVE'] * 3
motive = pd.concat([motive.iloc[:,0],
                    motive.iloc[:,1],
                    motive.iloc[:,2]])
motive = motive.reset_index()

#NaN값은 드랍
motive.dropna(inplace=True)
motive = motive.astype({"TRAVEL_MOTIVE": str})

#code 정보로 각 숫자에 의미 삽입하기
target_code = code[code['cd_a'] == 'TMT']
target_code = target_code[['cd_b','cd_nm']]

motive = pd.merge(motive, target_code, left_on='TRAVEL_MOTIVE',right_on='cd_b', how='left')
motive.drop(['TRAVEL_MOTIVE', 'cd_b'], axis=1, inplace=True)
motive.columns = ['TRAVELER_ID', 'TRAVEL_MOTIVE']

#분해하기
motive_copy = motive.copy()
motive_copy['count'] = motive_copy.groupby('TRAVELER_ID').cumcount() + 1

motive_copy_pivot = motive_copy.pivot_table(index='TRAVELER_ID', columns='count', values='TRAVEL_MOTIVE', aggfunc='first')
motive_copy_pivot.columns = [f'TRAVEL_MOTIVE_{i}' for i in motive_copy_pivot.columns]

motive_copy_pivot.reset_index(inplace=True)

#NA를 "missing"으로 바꾸고, 원데이터에 붙이기
motive_copy_pivot.fillna('missing', inplace=True)
traveler.drop(['TRAVEL_MOTIVE_1','TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3'], axis=1, inplace=True)
traveler = pd.merge(traveler, motive_copy_pivot, on='TRAVELER_ID', how='left')


#%% 3. User_Features 데이터 만들기
#travel과 traveler에 모두 존재하면서, Primary Key 역할을 수행할 수 있는 'TRAVELER_ID' 변수를 기준으로 Inner_Join한다.
user_features = pd.merge(travel,traveler, on='TRAVELER_ID', how='inner')

#해당 user_features 데이터를 DB에 적재한다.
db_config = {
    'host': '****',
    'user': '****',
    'password': '****',
    'database': '****',
    'port': 30575
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()
user_features.to_sql('user_features', conn, if_exists='replace', index=False)

cursor.close()
conn.close()



















































































































