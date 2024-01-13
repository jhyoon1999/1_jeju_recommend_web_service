from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List

class BinaryValue(IntEnum):
    ZERO = 0
    ONE = 1

class New_User_Data(BaseModel):
    GENDER: str = Field(..., pattern="^(남|여)$", description="The gender must be one of '남' or '여'")
    AGE_GRP: str = Field(..., pattern="^(30대|40대|50대|20대|60대)$", description="The age must be one of '20, 30, 40, 50, 60대'")
    TRAVEL_STATUS_ACCOMPANY: str = Field(..., pattern="^(3대 동반 여행\(친척 포함\)|자녀 동반 여행|2인 여행\(가족 외\)|나홀로 여행|3인 이상 여행\(가족 외\)|부모 동반 여행|2인 가족 여행)$", description="The accompany is error'")
    TRAVEL_STYL_1 : str = Field(..., pattern="^(자연매우선호|중립|도시중간선호|자연약간선호|자연중간선호|도시약간선호|도시매우선호)$", description="The style_1 is error'")
    TRAVEL_STYL_3 : str = Field(..., pattern="^(새로운지역중간선호|새로운지역매우선호|새로운지역약간선호|중립|익순한지역약간선호|익순한지역매우선호|익순한지역중간선호)$", description="The style_3 is error'")
    TRAVEL_STYL_5 : str = Field(..., pattern="^(체험활동중간선호|중립|체험활동약간선호|휴양또는휴식중간선호|휴양또는휴식약간선호|체험활동매우선호|휴양또는휴식매우선호)$", description="The style_5 is error'")
    TRAVEL_STYL_6 : str = Field(..., pattern="^(잘알려지지않은방문지중간선호|알려진방문지매우선호|알려진방문지중간선호|중립|잘알려지지않은방문지약간선호|잘알려지지않은방문지매우선호|알려진방문지약간선호)$", description="The style_6 is error'")
    TRAVEL_MISSION_SNS_인생샷_여행: BinaryValue
    TRAVEL_MISSION_Well_ness_여행: BinaryValue	
    TRAVEL_MISSION_교육_체험_프로그램_참가: BinaryValue	
    TRAVEL_MISSION_드라마_촬영지_방문: BinaryValue	
    TRAVEL_MISSION_등반_여행: BinaryValue
    TRAVEL_MISSION_반려동물_동반_여행: BinaryValue	
    TRAVEL_MISSION_쇼핑	: BinaryValue
    TRAVEL_MISSION_시티투어	: BinaryValue
    TRAVEL_MISSION_신규_여행지_발굴	: BinaryValue
    TRAVEL_MISSION_야외_스포츠_레포츠_활동: BinaryValue
    TRAVEL_MISSION_역사_유적지_방문: BinaryValue	
    TRAVEL_MISSION_온천_스파: BinaryValue	
    TRAVEL_MISSION_유흥_오락_나이트라이프: BinaryValue
    TRAVEL_MISSION_인플루언서_따라하기_여행: BinaryValue
    TRAVEL_MISSION_종교_성지_순례: BinaryValue
    TRAVEL_MISSION_지역_문화예술_공연_전시시설_관람: BinaryValue
    TRAVEL_MISSION_지역_축제_이벤트_참가: BinaryValue
    TRAVEL_MISSION_친환경_여행_플로깅_여행: BinaryValue
    TRAVEL_MISSION_캠핑: BinaryValue
    TRAVEL_MISSION_테마파크_놀이시설_동_식물원_방문: BinaryValue
    TRAVEL_MISSION_호캉스_여행: BinaryValue
    TRAVEL_MOTIVE_SNS_사진_등록_등_과시: BinaryValue
    TRAVEL_MOTIVE_기타: BinaryValue
    TRAVEL_MOTIVE_새로운_경험_추구: BinaryValue
    TRAVEL_MOTIVE_쉴_수_있는_기회_육체_피로_해결_및_정신적인_휴식: BinaryValue
    TRAVEL_MOTIVE_여행_동반자와의_친밀감_및_유대감_증진: BinaryValue
    TRAVEL_MOTIVE_역사_탐방_문화적_경험_등_교육적_동기: BinaryValue
    TRAVEL_MOTIVE_운동_건강_증진_및_충전: BinaryValue
    TRAVEL_MOTIVE_일상적인_환경_및_역할에서의_탈출_지루함_탈피: BinaryValue
    TRAVEL_MOTIVE_진정한_자아_찾기_또는_자신을_되돌아볼_기회_찾기: BinaryValue
    TRAVEL_MOTIVE_특별한_목적_칠순여행_신혼여행_수학여행_인센티브여행: BinaryValue
    
    class Config :
        json_schema_extra = {
            'example' : {
                        "GENDER": "남",
                        "AGE_GRP": "30대",
                        "TRAVEL_STATUS_ACCOMPANY": "자녀 동반 여행",
                        "TRAVEL_STYL_1": "중립",
                        "TRAVEL_STYL_3": "중립",
                        "TRAVEL_STYL_5": "중립",
                        "TRAVEL_STYL_6": "중립",
                        "TRAVEL_MISSION_SNS_인생샷_여행": 1,
                        "TRAVEL_MISSION_Well_ness_여행": 0,
                        "TRAVEL_MISSION_교육_체험_프로그램_참가": 0,
                        "TRAVEL_MISSION_드라마_촬영지_방문": 0,
                        "TRAVEL_MISSION_등반_여행": 0,
                        "TRAVEL_MISSION_반려동물_동반_여행": 1,
                        "TRAVEL_MISSION_쇼핑": 0,
                        "TRAVEL_MISSION_시티투어": 0,
                        "TRAVEL_MISSION_신규_여행지_발굴": 0,
                        "TRAVEL_MISSION_야외_스포츠_레포츠_활동": 0,
                        "TRAVEL_MISSION_역사_유적지_방문": 1,
                        "TRAVEL_MISSION_온천_스파": 1,
                        "TRAVEL_MISSION_유흥_오락_나이트라이프": 0,
                        "TRAVEL_MISSION_인플루언서_따라하기_여행": 0,
                        "TRAVEL_MISSION_종교_성지_순례": 0,
                        "TRAVEL_MISSION_지역_문화예술_공연_전시시설_관람": 1,
                        "TRAVEL_MISSION_지역_축제_이벤트_참가": 0,
                        "TRAVEL_MISSION_친환경_여행_플로깅_여행": 0,
                        "TRAVEL_MISSION_캠핑": 0,
                        "TRAVEL_MISSION_테마파크_놀이시설_동_식물원_방문": 0,
                        "TRAVEL_MISSION_호캉스_여행": 0,
                        "TRAVEL_MOTIVE_SNS_사진_등록_등_과시": 0,
                        "TRAVEL_MOTIVE_기타": 0,
                        "TRAVEL_MOTIVE_새로운_경험_추구": 1,
                        "TRAVEL_MOTIVE_쉴_수_있는_기회_육체_피로_해결_및_정신적인_휴식": 1,
                        "TRAVEL_MOTIVE_여행_동반자와의_친밀감_및_유대감_증진": 0,
                        "TRAVEL_MOTIVE_역사_탐방_문화적_경험_등_교육적_동기": 0,
                        "TRAVEL_MOTIVE_운동_건강_증진_및_충전": 1,
                        "TRAVEL_MOTIVE_일상적인_환경_및_역할에서의_탈출_지루함_탈피": 0,
                        "TRAVEL_MOTIVE_진정한_자아_찾기_또는_자신을_되돌아볼_기회_찾기": 1,
                        "TRAVEL_MOTIVE_특별한_목적_칠순여행_신혼여행_수학여행_인센티브여행": 1
                    }
        }

class Filter(BaseModel):
    travel : BinaryValue
    sports : BinaryValue
    cafe : BinaryValue
    service : BinaryValue
    art : BinaryValue
    market : BinaryValue
    transport : BinaryValue
    
    class Config :
        json_schema_extra = {
            'example' : {
                "travel" : 1,
                "sports" : 1,
                "cafe" : 1,
                "service" : 1,
                "art" : 1,
                "market" : 1,
                "transport" : 1
            }
        }


class Recommend_Data(BaseModel) :
    new_spot_id : str
    category : str
    GENDER : str
    AGE_GRP : str
    TRAVEL_STATUS_ACCOMPANY : str
    TRAVEL_STYL_1 : str
    TRAVEL_STYL_3 : str
    TRAVEL_STYL_5 : str
    TRAVEL_STYL_6 : str
    TRAVEL_MISSION_SNS_인생샷_여행: BinaryValue
    TRAVEL_MISSION_Well_ness_여행: BinaryValue
    TRAVEL_MISSION_교육_체험_프로그램_참가: BinaryValue
    TRAVEL_MISSION_드라마_촬영지_방문: BinaryValue
    TRAVEL_MISSION_등반_여행: BinaryValue
    TRAVEL_MISSION_반려동물_동반_여행: BinaryValue
    TRAVEL_MISSION_쇼핑: BinaryValue
    TRAVEL_MISSION_시티투어: BinaryValue
    TRAVEL_MISSION_신규_여행지_발굴: BinaryValue
    TRAVEL_MISSION_야외_스포츠_레포츠_활동: BinaryValue
    TRAVEL_MISSION_역사_유적지_방문: BinaryValue
    TRAVEL_MISSION_온천_스파: BinaryValue
    TRAVEL_MISSION_유흥_오락_나이트라이프: BinaryValue
    TRAVEL_MISSION_인플루언서_따라하기_여행: BinaryValue
    TRAVEL_MISSION_종교_성지_순례: BinaryValue
    TRAVEL_MISSION_지역_문화예술_공연_전시시설_관람: BinaryValue
    TRAVEL_MISSION_지역_축제_이벤트_참가: BinaryValue
    TRAVEL_MISSION_친환경_여행_플로깅_여행: BinaryValue
    TRAVEL_MISSION_캠핑: BinaryValue
    TRAVEL_MISSION_테마파크_놀이시설_동_식물원_방문: BinaryValue
    TRAVEL_MISSION_호캉스_여행: BinaryValue
    TRAVEL_MOTIVE_SNS_사진_등록_등_과시: BinaryValue
    TRAVEL_MOTIVE_기타: BinaryValue
    TRAVEL_MOTIVE_새로운_경험_추구: BinaryValue
    TRAVEL_MOTIVE_쉴_수_있는_기회_육체_피로_해결_및_정신적인_휴식: BinaryValue
    TRAVEL_MOTIVE_여행_동반자와의_친밀감_및_유대감_증진: BinaryValue
    TRAVEL_MOTIVE_역사_탐방_문화적_경험_등_교육적_동기: BinaryValue
    TRAVEL_MOTIVE_운동_건강_증진_및_충전: BinaryValue
    TRAVEL_MOTIVE_일상적인_환경_및_역할에서의_탈출_지루함_탈피: BinaryValue
    TRAVEL_MOTIVE_진정한_자아_찾기_또는_자신을_되돌아볼_기회_찾기: BinaryValue
    TRAVEL_MOTIVE_특별한_목적_칠순여행_신혼여행_수학여행_인센티브여행: BinaryValue
    RESIDENCE_TIME_MIN : float
    VISIT_CHC_REASON_가기_편해서_교통이_좋아서: BinaryValue
    VISIT_CHC_REASON_가성비가_좋아서: BinaryValue
    VISIT_CHC_REASON_과거_경험이_좋아서: BinaryValue
    VISIT_CHC_REASON_교육성이_좋아서: BinaryValue
    VISIT_CHC_REASON_기타: BinaryValue
    VISIT_CHC_REASON_미디어_TV_정보_프로그램_등_평가가_좋아서: BinaryValue
    VISIT_CHC_REASON_온라인_SNS_블로그_등_평가가_좋아서: BinaryValue
    VISIT_CHC_REASON_지나가다_우연히: BinaryValue
    VISIT_CHC_REASON_지명도_명소_핫플레이스: BinaryValue
    VISIT_CHC_REASON_지인의_추천이_있어서: BinaryValue
    VISIT_CHC_REASON_편의시설_서비스가_좋아서: BinaryValue
    DGSTFN: float
    REVISIT_INTENTION: float
    RCMDTN_INTENTION: float
    VISIT_AREA_TYPE_CD_레저_스포츠_관련_시설_스키_카트_수상레저: BinaryValue
    VISIT_AREA_TYPE_CD_문화_시설_공연장_영화관_전시관_등: BinaryValue
    VISIT_AREA_TYPE_CD_산책로_둘레길_등: BinaryValue
    VISIT_AREA_TYPE_CD_상업지구_거리_시장_쇼핑시설: BinaryValue
    VISIT_AREA_TYPE_CD_역사_유적_종교_시설_문화재_박물관_촬영지_절_등: BinaryValue
    VISIT_AREA_TYPE_CD_자연관광지: BinaryValue
    VISIT_AREA_TYPE_CD_지역_축제_행사: BinaryValue
    VISIT_AREA_TYPE_CD_체험_활동_관광지: BinaryValue
    VISIT_AREA_TYPE_CD_테마시설_놀이공원_워터파크: BinaryValue

class Recommend_Data_List(BaseModel):
    recommend_data_list : List[Recommend_Data] = []

class Recommend_Form(BaseModel) :
    new_spot_id : str
    category_name : str
    place_name : str
    address_name : str
    place_url : str
    x : str
    y : str
    RESIDENCE_TIME_MIN : str
    VISIT_CHC_REASON : str
    VISIT_AREA_TYPE_CD : str
    DGSTFN : str
    REVISIT_INTENTION : str
    RCMDTN_INTENTION : str
    recommend_score : float

class Recommend_Form_List(BaseModel) :
    recommend_form_list : List[Recommend_Form] = []