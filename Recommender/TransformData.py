import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import datetime

impressions = pd.read_table(
    "behaviors.tsv",
    header = None,
    names= ['impression_id', 'user_id','time','history','impressions']
    )

impressions= impressions.dropna(subset=["history"])

news = pd.read_table("news.tsv",
              header=None,
              names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])
              
## Impressions
impressions['impressions'] = impressions['impressions'].str.replace(r'\w+-1( |)','').str.replace(r'-0','')
impressions['impressions'] = impressions['impressions'].str.split(' ').tolist()
impressions['impressions'] = impressions['impressions'].map(lambda x: x[0])

impressions_features = impressions['impressions'].tolist()
impressions_features = np.reshape(impressions_features, (-1, 1)).tolist()

## History id to title
dictionary = pd.Series(news.title.values, index=news.id).to_dict()

for items in tqdm(impressions_features):
    items.append(dictionary.get(items[0]))

## User History
impressions['history'] = impressions['history'].str.split(' ').tolist()
impressions['history'] = impressions['history'].map(lambda x: x[:10])
users_history = np.zeros([len(impressions["history"]), 10], dtype="<U100")

for i,j in enumerate(impressions['history']):
    users_history[i][0:len(j)] = j

## User Category from history
dictionary = pd.Series(news.category.values, index=news.id).to_dict()
users_history_category = []
for items in tqdm(users_history):
    item_category_list = []
    for news_id in items:
        if news_id:
            item_category_list.append(dictionary.get(news_id))
        else:
            item_category_list.append("unknown")
    users_history_category.append(item_category_list)

for items in tqdm(impressions_features):
    items.append(dictionary.get(items[0]))

## User SubCategory from history
dictionary = pd.Series(news.subcategory.values, index=news.id).to_dict()
users_history_subcategory = []
for items in tqdm(users_history):
    item_category_list = []
    for news_id in items:
        if news_id:
            item_category_list.append(dictionary.get(news_id))
        else:
            item_category_list.append("unknown")
    users_history_subcategory.append(item_category_list)

for items in tqdm(impressions_features):
    items.append(dictionary.get(items[0]))

## User news titles from history
dictionary = pd.Series(news.title.values, index=news.id).to_dict()
users_history_titles = []
for items in tqdm(users_history):
    item_category_list = []
    for news_id in items:
        if news_id:
            item_category_list.append([dictionary.get(news_id)])
        else:
            item_category_list.append(["unknown"])
    users_history_titles.append(item_category_list)
        

## Add to dataframe
transformedData = pd.DataFrame()

transformedData["user_id"]= impressions.pop("user_id")
transformedData["timestamp"]= impressions.pop("time")
transformedData["history"]= users_history.tolist()
transformedData["Category"] = users_history_category
transformedData["subcategory"] = users_history_subcategory
transformedData["title"] = users_history_titles
transformedData["next_item"]= impressions_features

transformedData.to_csv("finalDataB.csv", index=False)

print("Done")
