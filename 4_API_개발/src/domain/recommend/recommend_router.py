from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.domain.recommend import recommend_crud, recommend_schema

router = APIRouter(
    prefix = "/recommend",
    tags = ["recommend"]
)

@router.post('/recommendation')
async def get_recommend_item(db : Session = Depends(get_db),
                            new_user_data : recommend_schema.New_User_Data = None,
                            filter_item : recommend_schema.Filter = None):
    recommend_data_input = recommend_crud.make_data(db = db, new_user_data= new_user_data,
                                                        filter_item=filter_item)
    predictions_top_5_dict = recommend_crud.get_recommend(recommend_data_input = recommend_data_input)
    recommended_item = recommend_crud.returning_recommended_item(db = db,
                                                                predictions_top_5_dict=predictions_top_5_dict)
    return recommended_item
