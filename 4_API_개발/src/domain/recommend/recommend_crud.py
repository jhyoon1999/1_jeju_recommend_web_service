from src.domain.recommend.recommend_schema import New_User_Data, Filter
from src.domain.recommend.recommend_schema import Recommend_Data_List
from src.domain.recommend.recommend_schema import Recommend_Form_List
from src.database.models import ItemFeatures

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
import catboost

def top_spots(group):
    # nlargest를 사용하여 상위 5개(또는 그 이하)의 new_spot_id를 선택하고 딕셔너리로 반환
    return group.nlargest(5, 'prediction_score').set_index('new_spot_id')['prediction_score'].to_dict()

def make_data(db : Session, filter_item : Filter, new_user_data : New_User_Data) :
    new_user_data_dict = new_user_data.dict()
    
    filter_dict = filter_item.dict()
    filter_list = [key for key,value in filter_dict.items() if value == 1]
    item_call = db.query(ItemFeatures).filter(ItemFeatures.category.in_(filter_list)).all()
    item_data = [item.__dict__ for item in item_call]
    item_data = [{k: v for k, v in item.items() if k != "_sa_instance_state"} for item in item_data]
    
    item_data = pd.DataFrame(item_data)
    new_user_data = pd.DataFrame(new_user_data_dict, index=[0])
    new_user_data = pd.concat([new_user_data]*len(item_data), ignore_index=True)

    recommend_data = pd.concat([item_data, new_user_data], axis=1)
    recommend_data_dict = recommend_data.to_dict(orient='records')

    recommend_data_dict_pydantic = Recommend_Data_List(recommend_data_list=recommend_data_dict)
    recommend_data_input = recommend_data_dict_pydantic.dict()
    return recommend_data_input

def get_recommend(recommend_data_input: Recommend_Data_List) :
    predict_data = recommend_data_input['recommend_data_list']
    predict_data = pd.DataFrame(predict_data)
    predict_data.drop_duplicates(inplace=True)
    
    model = catboost.CatBoostRegressor()
    model.load_model('final_model.cbm')
    
    predict_data_spot_id = list(predict_data['new_spot_id'])
    predict_data_category = list(predict_data['category'])
    predict_data_input = predict_data
    predict_data_input.drop(['new_spot_id', 'category'], axis = 1, inplace=True)
    predict_data_input.rename(columns={"TRAVEL_MISSION_Well_ness_여행":'TRAVEL_MISSION_Well-ness_여행'}, inplace=True)

    predictions = model.predict(predict_data_input)
    predictions_result = pd.DataFrame({'new_spot_id' : predict_data_spot_id,
                                "category" : predict_data_category,
                                "prediction_score" : predictions})
    predictions_result['prediction_score'] = np.clip(predictions_result['prediction_score'], 1, 5)
    predictions_top_5_dict = predictions_result.groupby('category').apply(top_spots).to_dict()
    return predictions_top_5_dict

def returning_recommended_item(db : Session, predictions_top_5_dict : dict) :
    recommended_item = predictions_top_5_dict
    for category, items in recommended_item.items() :
        items_new_spot_id = list(items.keys())
        items_item_info = db.query(ItemFeatures).filter(ItemFeatures.new_spot_id.in_(items_new_spot_id)).all()
        items_item_info = [item.__dict__ for item in items_item_info]
        items_item_info = [{k: v for k, v in item.items() if k != "_sa_instance_state"} for item in items_item_info]
        items_item_info = pd.DataFrame(items_item_info)
        items_item_info.drop_duplicates(subset=['new_spot_id'], inplace = True)

        #추천점수 붙이기
        score = pd.DataFrame(list(items.items()), columns=['new_spot_id', 'recommend_score'])
        score['recommend_score'] = ((score['recommend_score'] - 1) / (5 - 1)) * (100 - 0) + 0
        items_item_info = pd.merge(items_item_info, score, how = 'left', on = 'new_spot_id')

        items_dict = items_item_info.to_dict('records')
        items_dict_pydantic = Recommend_Form_List(recommend_form_list = items_dict)
        items_dict_return = items_dict_pydantic.dict()['recommend_form_list']

        recommended_item[category] = items_dict_return
    
    return recommended_item






















