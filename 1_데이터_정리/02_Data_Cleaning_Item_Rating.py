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
        # 방문지 데이터 불러오기
        spot_table_name = 'visit_info'
        spot = fetch_table_data(cursor, spot_table_name)
        print("\nSpot Data:")
        print(spot.head())
        print(spot.nunique())
        print(spot.shape)

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

#(1). 방문지 유형 변수 기반 관광지만 남기기
target_vis_code = [1,2,3,4,5,6,7,8,13]
spot = spot[spot['VISIT_AREA_TYPE_CD'].isin(target_vis_code)]

#(2). 방문지명을 기준으로 spot_id 생성
new_id_index = {original: 'spot_' + str(idx) for idx, original in enumerate(spot['VISIT_AREA_NM'].unique())}
spot['spot_id'] = spot['VISIT_AREA_NM'].map(new_id_index)



#%%3. 관광지 식별 : 카카오지도를 통해 식별이 가능한 관광지만을 남긴다.
#카카오 지도로 검색되지 않은 관광지는 수동으로 탐색 후 수정 혹은 탈락시킨다.
#해당 과정을 총 3회 반복하였다.

#(1). 관광지명 기반 카카오지도 검색
import requests
spot_geo_info = spot[['spot_id', 'VISIT_AREA_NM']]

# 상수 정의
API_KEY = '****'  # 실제 키로 교체해야 함
URL = 'https://dapi.kakao.com/v2/local/search/keyword.json'

def get_place_info(target_name):
    header = {'Authorization': 'KakaoAK ' + API_KEY}
    params = {'query': target_name, 'page': 1}
    
    try:
        places = requests.get(URL, params=params, headers=header).json()['documents']
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

    if not places:
        print(f"No results for: {target_name}")
        return None

    for place in places:
        if '제주특별자치도' in place['address_name']:
            return {
                'address_name': place['address_name'],
                'category_name': place['category_name'],
                'place_name': place['place_name'],
                'place_url': place['place_url'],
                'road_address_name': place['road_address_name'],
                'x': place['x'],
                'y': place['y']
            }

    # 만약 검색된 주소들에 '제주특별자치도'가 없는 경우, 첫 번째 결과를 반환
    return {
        'address_name': places[0]['address_name'],
        'category_name': places[0]['category_name'],
        'place_name': places[0]['place_name'],
        'place_url': places[0]['place_url'],
        'road_address_name': places[0]['road_address_name'],
        'x': places[0]['x'],
        'y': places[0]['y']
    }

# spot 데이터프레임에서 unique한 spot_id들을 추출
unique_spot_ids = spot['spot_id'].unique()

# 데이터프레임을 구성하기 위한 리스트 초기화
spot_id_list = []
area_name_record_list = []
address_name_list = []
category_name_list = []
place_name_list = []
place_url_list = []
road_address_name_list = []
x_list = []
y_list = []

for target_spot in unique_spot_ids:
    # spot_id 및 관광지 명 추출
    spot_id_list.append(target_spot)
    target_name = spot_geo_info[spot_geo_info['spot_id'] == target_spot]['VISIT_AREA_NM'].unique().item()
    area_name_record_list.append(target_name)

    # 카카오지도 API 호출을 통한 정보 추출
    place_info = get_place_info(target_name)

    if place_info:
        address_name_list.append(place_info['address_name'])
        category_name_list.append(place_info['category_name'])
        place_name_list.append(place_info ['place_name'])
        place_url_list.append(place_info['place_url'])
        road_address_name_list.append(place_info['road_address_name'])
        x_list.append(place_info['x'])
        y_list.append(place_info['y'])
    else:
        # API 호출 실패 시 NaN 값 추가
        address_name_list.append(np.NaN)
        category_name_list.append(np.NaN)
        place_name_list.append(np.NaN)
        place_url_list.append(np.NaN)
        road_address_name_list.append(np.NaN)
        x_list.append(np.NaN)
        y_list.append(np.NaN)

# 데이터프레임 생성
spot_geo = pd.DataFrame({
    'spot_id': spot_id_list,
    'area_name_record': area_name_record_list,
    'address_name': address_name_list,
    'category_name': category_name_list,
    'place_name': place_name_list,
    'place_url': place_url_list,
    'road_address_name': road_address_name_list,
    'x': x_list,
    'y': y_list
})

spot_geo.to_excel('spot_geo.xlsx')
#제주도가 아닌 관광지는 탈락
#검색되지 않은 관광지는 일일이 확인하여 관광지명을 수정하거나 탈락
#관광지명이 수정된 관광지는 재차 카카오지도 검색하는 과정을 거침 -> new_spot_id 변수 생성됨

#%%4. 데이터 분할 : item_features와 ratings 데이터로 나누기
ratings = spot[['new_spot_id', 'TRAVEL_ID', 'DGSTFN']]
item_features = spot.drop(['DGSTFN'], axis=1)

#item_features와 ratings 데이터를 DB에 적재한다.
db_config = {
    'host': '****',
    'user': '****',
    'password': '****',
    'database': '****',
    'port': 30575
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()
item_features.to_sql('item_features', conn, if_exists='replace', index=False)
ratings.to_sql('ratings', conn, if_exists='replace', index=False)

cursor.close()
conn.close()




































































































