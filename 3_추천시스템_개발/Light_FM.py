import pandas as pd
import numpy as np
import os
import pickle

print(os.getcwd())
os.chdir('c:/Users/gardp/OneDrive/바탕 화면/업무/여행지_추천_웹서비스_업무/추천시스템')

#%% Load ratings, user_features, item_features

#1. ratings
ratings_data = pd.read_csv('06_추천시스템개발3/train_data/ratings.csv')
ratings_data.head()
ratings_data.info()
ratings_data.nunique()

#2. user_features
with open('06_추천시스템개발3/train_data/user_features.pkl', 'rb') as f:
    user_features = pickle.load(f)
print(type(user_features))
len(user_features)

with open('06_추천시스템개발3/train_data/user_features_name.pkl', 'rb') as f:
    user_features_name = pickle.load(f)
print(type(user_features_name))
len(user_features_name)

#3. item_features
with open('06_추천시스템개발3/train_data/item_features.pkl', 'rb') as f:
    item_features = pickle.load(f)
print(type(item_features))
len(item_features)

with open('06_추천시스템개발3/train_data/item_features_name.pkl', 'rb') as f:
    item_features_name = pickle.load(f)
print(type(item_features_name))
len(item_features_name)

#%%Make Dataset
from lightfm.data import Dataset

dataset = Dataset()

#1. fit
dataset.fit(users = ratings_data['TRAVEL_ID'],
            items = ratings_data['spot_id'],
            user_features = user_features_name,
            item_features = item_features_name
            )

#2. interactions
interactions, weights = dataset.build_interactions((x[1], x[0]) for x in ratings_data.values)
interactions.todense()
interactions.shape

weights.todense()
weights.shape

#3. item_features
item_features_input = dataset.build_item_features(item_features)
item_features_input.todense()
item_features_input.shape

#4. user_features
user_features_input = dataset.build_user_features(user_features)
user_features_input.todense()
user_features_input.shape

#%% Split Data
from lightfm.cross_validation import random_train_test_split
train_interactions, test_interactions = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(42))
train_weights, test_weights = random_train_test_split(weights, test_percentage=0.2, random_state=np.random.RandomState(42))

#%% Modeling
from lightfm import LightFM

#Grid search 결과를 불러와서 최적의 하이퍼파라미터 입력
tuning_result = pd.read_excel('05_추천시스템개발2/hyper_tuning/implicit_user_5.xlsx')
tuning_result = tuning_result.sort_values('AUC',ascending=False).head()

model = LightFM(no_components=tuning_result.iloc[0]['no_components'],
                    learning_rate=tuning_result.iloc[0]['learning_rate'],
                    loss=tuning_result.iloc[0]['loss'],
                    item_alpha=tuning_result.iloc[0]['item_alpha'],
                    user_alpha=tuning_result.iloc[0]['user_alpha'],
                    learning_schedule=tuning_result.iloc[0]['learning_schedule']
)

model.fit(interactions = train_interactions,
        user_features = user_features_input,
        item_features = item_features_input,
        epochs = 10,
        num_threads = 4)

#%% Model Evaluation
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

auc_score(model, test_interactions,
        train_interactions = train_interactions,
        item_features=item_features_input,
        user_features=user_features_input,
        check_intersections = False).mean()

precision_at_k(model, test_interactions,
            k=10, train_interactions = train_interactions,
            item_features=item_features_input,
            user_features=user_features_input,
            check_intersections = False).mean()

recall_at_k(model, test_interactions,
            k=10, train_interactions = train_interactions,
            item_features=item_features_input,
            user_features=user_features_input,
            check_intersections = False).mean()

#%% Final Model
final_model = LightFM(no_components=tuning_result.iloc[0]['no_components'],
                    learning_rate=tuning_result.iloc[0]['learning_rate'],
                    loss=tuning_result.iloc[0]['loss'],
                    item_alpha=tuning_result.iloc[0]['item_alpha'],
                    user_alpha=tuning_result.iloc[0]['user_alpha'],
                    learning_schedule=tuning_result.iloc[0]['learning_schedule']
)
final_model.fit(interactions = interactions,
        user_features = user_features_input,
        item_features = item_features_input,
        epochs = 10,
        num_threads = 4)

parameter_info = {'no_component' : tuning_result.iloc[0]['no_components'],
                'learning_rate' : tuning_result.iloc[0]['learning_rate'],
                'loss' : tuning_result.iloc[0]['loss'],
                'item_alpha' : tuning_result.iloc[0]['item_alpha'],
                'user_alpha' : tuning_result.iloc[0]['user_alpha'],
                'learning_schedule' : tuning_result.iloc[0]['learning_schedule']}
parameter_info

with open('06_추천시스템개발3/train_data/parameter_info.pkl', 'wb') as f:
    pickle.dump(parameter_info, f)
