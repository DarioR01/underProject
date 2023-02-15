import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from ast import literal_eval

#training  constants
embedding_dimension=32
learning_rate=0.1
epochs=100

#data import
impressions = pd.read_csv(
    "TransformedData.csv",
    header = None,
    names= ['user_id', 'timestamp','history','category','subcategory',"next_item"]
    ) 

news_data = pd.read_table("news.tsv",
              header=None,
              names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])

history_eval = impressions["history"].map(lambda x: literal_eval(x)).tolist()
category_eval = impressions["category"].map(lambda x: literal_eval(x)).tolist()
subcategory_eval = impressions["subcategory"].map(lambda x: literal_eval(x)).tolist()

#Create Data Tensors && dataset
history_tensor = tf.convert_to_tensor(history_eval, dtype=tf.string)
category_tensor = tf.convert_to_tensor(category_eval, dtype=tf.string)
subcategory_tensor = tf.convert_to_tensor(subcategory_eval, dtype=tf.string)
next_news_eval = tf.convert_to_tensor(impressions["next_item"].values, dtype=tf.string)
dataset = tf.data.Dataset.from_tensor_slices((next_news_eval, history_eval, category_eval, subcategory_eval))

#Vocabularies
news_id_vocabulary = tf.convert_to_tensor(list(news_data["id"].unique()), dtype=tf.string)
news_category_vocabulary = tf.convert_to_tensor(list(news_data["category"].unique()), dtype=tf.string)
news_subcategory_vocabulary = tf.convert_to_tensor(list(news_data["subcategory"].unique()), dtype=tf.string)

news_id_vocabulary_dataset = tf.data.Dataset.from_tensor_slices((news_id_vocabulary))

class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        #Create History Model
        self.history_model = tf.keras.Sequential()
        self.history_model.add(tf.keras.layers.StringLookup(vocabulary=news_id_vocabulary, mask_token=None))
        self.history_model.add(tf.keras.layers.Embedding(len(news_id_vocabulary)+1, embedding_dimension))
        self.history_model.add(tf.keras.layers.GRU(embedding_dimension))

        #Create Category Model
        self.category_model = tf.keras.Sequential()
        self.category_model.add(tf.keras.layers.StringLookup(vocabulary=news_category_vocabulary, mask_token=None))
        self.category_model.add(tf.keras.layers.Embedding(len(news_category_vocabulary)+1, embedding_dimension))
        self.category_model.add(tf.keras.layers.GRU(embedding_dimension))

        #Create SubCategory Model
        self.subcategory_model = tf.keras.Sequential()
        self.subcategory_model.add(tf.keras.layers.StringLookup(vocabulary=news_subcategory_vocabulary, mask_token=None))
        self.subcategory_model.add(tf.keras.layers.Embedding(len(news_subcategory_vocabulary)+1, embedding_dimension))
        self.subcategory_model.add(tf.keras.layers.GRU(embedding_dimension))

    def call(self, features):
        return tf.concat([
            self.history_model(features[0]), 
            self.category_model(features[1]), 
            self.subcategory_model(features[2])
        ], axis=1)
    

class Model(tfrs.Model):

    def __init__(self):
        super().__init__()

        self.query_model = tf.keras.Sequential([
            UserModel(),
            tf.keras.layers.Dense(32)
        ])

        #Candidate_model
        self.candidate_model = tf.keras.Sequential()
        self.candidate_model.add(tf.keras.layers.StringLookup(vocabulary=news_id_vocabulary, mask_token=None))
        self.candidate_model.add(tf.keras.layers.Embedding(len(news_id_vocabulary) +1, embedding_dimension))

        self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=news_id_vocabulary_dataset.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        candidate_embedding = self.candidate_model(features[0])

        query_embedding = self.query_model(features[1:])

        return self.task(query_embedding, candidate_embedding, compute_metrics=not training)



## Compile and Train Model
model = Model()

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

cached_train = dataset.shuffle(10_000).batch(10000).cache()
cached_test = dataset.batch(5000).cache()

model.fit(cached_train, epochs=epochs)

model.evaluate(cached_test, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((news_id_vocabulary_dataset.batch(100), news_id_vocabulary_dataset.batch(100).map(model.candidate_model)))
)

# Get recommendations.
_, titles = index(tf.constant([
    ["N57072","N60496","N44258","N62771","N49728","N18845","N39737","N45080","",""],
    ['tv', 'sports', 'tv', 'news', 'sports', 'lifestyle', 'movies', 'news', 'news', 'unknown'],
    ['tvnews', 'baseball_mlb', 'tvnews', 'newscrime', 'football_ncaa', 'lifestylebuzz', 'movienews', 'newspolitics', 'newspolitics', 'unknown']
]))
print(f"Then Give this: {titles[0, :3]}")

