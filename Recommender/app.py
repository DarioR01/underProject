import pandas as pd
import numpy as np
import tensorflow as tf
import re
import tensorflow_recommenders as tfrs

history_length = 10
learning_rate = 0.1
epochs = 100
embedding_dimension = 64

##User
users = pd.read_table(
    "behaviors.tsv",
    header = None,
    names= ['impression_id', 'user_id','time','history','impressions']
    )

users= users.dropna(subset=["history"])

users['history'] = users['history'].str.split(' ').tolist()
users['history'] = users['history'].map(lambda x: x[:history_length])
users_data = np.zeros([len(users["history"]), history_length], dtype="<U100")
for i,j in enumerate(users['history']):
    users_data[i][0:len(j)] = j

users['impressions'] = users['impressions'].str.replace(r'\w+-0( |)','').str.replace(r'-1','')
users['impressions'] = users['impressions'].str.split(' ').tolist()
users['impressions'] = users['impressions'].map(lambda x: x[0])

user_tensor = tf.constant(users[['impressions']].values, dtype=tf.string)
user_news_tensor =  tf.convert_to_tensor(users_data, dtype=tf.string)

dataset = tf.data.Dataset.from_tensor_slices((user_news_tensor, user_tensor))

##News
news_data = pd.read_table("news.tsv",
              header=None,
              names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])

news_tensor = tf.constant(news_data["id"], dtype=tf.string)
news = tf.data.Dataset.from_tensor_slices((news_tensor))
news_ids = news.batch(1000)
unique_news = np.unique(np.concatenate(list(news_ids)))

## Creating Model

query_model = tf.keras.Sequential()
query_model.add(tf.keras.layers.StringLookup(vocabulary=unique_news, mask_token=None))
query_model.add(tf.keras.layers.Embedding(len(unique_news)+1, embedding_dimension))
query_model.add(tf.keras.layers.GRU(embedding_dimension))

candidate_model = tf.keras.Sequential()
candidate_model.add(tf.keras.layers.StringLookup(vocabulary=unique_news, mask_token=None))
candidate_model.add(tf.keras.layers.Embedding(len(unique_news) +1, embedding_dimension))

metrics = tfrs.metrics.FactorizedTopK(candidates=news.batch(1024).map(candidate_model))

task = tfrs.tasks.Retrieval(
    metrics=metrics
)

class Model(tfrs.Model):

    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features[0]
        watch_next_label = features[1]

        query_embedding = self._query_model(watch_history)    
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)


## Compile and Train Model
model = Model(query_model, candidate_model)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

cached_train = dataset.shuffle(10_000).batch(10000).cache()
cached_test = dataset.batch(5000).cache()

model.fit(cached_train, epochs=epochs)

model.evaluate(cached_test, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model._query_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((news.batch(100), news.batch(100).map(model._candidate_model)))
)

# Get recommendations.
_, titles = index(tf.constant([["N57072","N60496","N44258","N62771","N49728","N18845","N39737","N45080","",""]]))
print(f"Then Give this: {titles[0, :3]}")