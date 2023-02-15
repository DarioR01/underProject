import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from ast import literal_eval

#data import
impressions = pd.read_csv(
    "TransformedData.csv",
    header = None,
    names= ['user_id','timestamp','history','category','subcategory','title','next_item']
    ) 

impressions = impressions.drop(columns=['user_id','timestamp','title'])

news_data = pd.read_table("news.tsv",
              header=None,
              names=[
                  'next_id', 'next_category', 'next_subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])

news_data = news_data.drop(columns=['title', 'abstract','url', 'title_entities','abstract_entities'])
news_data = news_data.drop_duplicates('next_id')

history = impressions["history"].map(lambda x: literal_eval(x)).tolist()
category = impressions["category"].map(lambda x: literal_eval(x)).tolist()
subcategory = impressions["subcategory"].map(lambda x: literal_eval(x)).tolist()
next_id = impressions["next_item"].map(lambda x: literal_eval(x)[0])
next_category = impressions["next_item"].map(lambda x: literal_eval(x)[1])
next_subcategory = impressions["next_item"].map(lambda x: literal_eval(x)[2])

history = tf.ragged.constant(history, dtype=tf.string)
category = tf.ragged.constant(category, dtype=tf.string)
subcategory = tf.ragged.constant(subcategory, dtype=tf.string)
next_id = tf.constant(next_id, dtype=tf.string)
next_category = tf.constant(next_category, dtype=tf.string)
next_subcategory = tf.constant(next_subcategory, dtype=tf.string)

news_dict = {name: np.array(value) for name, value in news_data.items()}
impressions_dict = {
    "history" : history,
    "category" : category,
    "subcategory" : subcategory,
    "next_id" : next_id,
    "next_category" : next_category,
    "next_subcategory" : next_subcategory,
}

news_ds = tf.data.Dataset.from_tensor_slices(news_dict)
impressions_ds = tf.data.Dataset.from_tensor_slices(impressions_dict)

#Vocabularies
news_id_vocabulary = np.unique(np.concatenate(list(news_ds.batch(1_000).map(lambda x: x["next_id"]))))
news_category_vocabulary = np.unique(np.concatenate(list(news_ds.batch(1_000).map(lambda x: x["next_category"]))))
news_subcategory_vocabulary = np.unique(np.concatenate(list(news_ds.batch(1_000).map(lambda x: x["next_subcategory"]))))

news_ds = news_ds.map(lambda x: {
    "next_id": x['next_id'],
    "next_category": x['next_category'],
    "next_subcategory": x['next_subcategory'],
})

impressions_ds = impressions_ds.map(lambda x: {
    "history" : x["history"],
    "category" : x["category"],
    "subcategory" : x["subcategory"],
    "next_id" : x["next_id"],
    "next_category" : x["next_category"],
    "next_subcategory" : x["next_subcategory"],
})

embedding_dimension=64
learning_rate=0.1
epochs=3

class UserModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        
        #Create History Model
        self.history_model = tf.keras.Sequential()
        self.history_model._name = "user_history"
        self.history_model.add(tf.keras.layers.StringLookup(vocabulary=news_id_vocabulary, mask_token=None))
        self.history_model.add(tf.keras.layers.Embedding(len(news_id_vocabulary)+1, embedding_dimension))
        self.history_model.add(tf.keras.layers.GRU(embedding_dimension))

        #Create Category Model
        self.category_model = tf.keras.Sequential()
        self.category_model._name = "user_category"
        self.category_model.add(tf.keras.layers.StringLookup(vocabulary=news_category_vocabulary, mask_token=None))
        self.category_model.add(tf.keras.layers.Embedding(len(news_category_vocabulary)+1, embedding_dimension))
        self.category_model.add(tf.keras.layers.GRU(embedding_dimension))

        #Create SubCategory Model
        self.subcategory_model = tf.keras.Sequential()
        self.subcategory_model._name = "user_subcategory"
        self.subcategory_model.add(tf.keras.layers.StringLookup(vocabulary=news_subcategory_vocabulary, mask_token=None))
        self.subcategory_model.add(tf.keras.layers.Embedding(len(news_subcategory_vocabulary)+1, embedding_dimension))
        self.subcategory_model.add(tf.keras.layers.GRU(embedding_dimension))

    def call(self, features):
        return tf.concat([
            self.history_model(features["history"]),
            self.category_model(features["category"]),
            self.subcategory_model(features["subcategory"]),
        ], axis = 1)
    
class NewsModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # ID_model
        self.NewsId_model = tf.keras.Sequential()
        self.NewsId_model._name = "news_id"
        self.NewsId_model.add(tf.keras.layers.StringLookup(vocabulary=news_id_vocabulary, mask_token=None))
        self.NewsId_model.add(tf.keras.layers.Embedding(len(news_id_vocabulary) +1, embedding_dimension))
        
        # category model
        self.news_category_model = tf.keras.Sequential()
        self.news_category_model._name = "news_category"
        self.news_category_model.add(tf.keras.layers.StringLookup(vocabulary=news_category_vocabulary, mask_token=None))
        self.news_category_model.add(tf.keras.layers.Embedding(len(news_category_vocabulary) +1, embedding_dimension))
        
        # subcategory model
        self.news_subcategory_model = tf.keras.Sequential()
        self.news_subcategory_model._name = "news_subcategory"
        self.news_subcategory_model.add(tf.keras.layers.StringLookup(vocabulary=news_subcategory_vocabulary, mask_token=None))
        self.news_subcategory_model.add(tf.keras.layers.Embedding(len(news_subcategory_vocabulary) +1, embedding_dimension))

    def call(self, features):
        return tf.concat([
            self.NewsId_model(features["next_id"]),
            self.news_category_model(features["next_category"]),
            self.news_subcategory_model(features["next_subcategory"]),
        ], axis = 1)
    
class Model(tfrs.Model):
    def __init__(self):
        super().__init__()

        self.query_model = tf.keras.Sequential([
            UserModel(),

        ])
        
        self.query_model._name = "query"
        
        self.candidate_model = tf.keras.Sequential([
            NewsModel(),
        ])
        
        self.candidate_model._name = "candidate"
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates= news_ds.batch(1024).map(self.candidate_model),
                ),
            name = "retrival_task"
        )

    def compute_loss(self, features, training=False):
        candidate_embedding = self.candidate_model({
            "next_id": features["next_id"],
            "next_category":features["next_category"],
            "next_subcategory": features["next_subcategory"],
        })
        query_embedding = self.query_model({
            "history": features["history"],
            "category":features["category"],
            "subcategory": features["subcategory"],
        })
        return self.task(query_embedding, candidate_embedding, compute_metrics=not training)

model = Model()

## Train Model
#training  constants

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

train_ds = impressions_ds.take(130_000)
test_ds = impressions_ds.skip(130_000).take(10_000)
validation_ds = impressions_ds.skip(130_000).skip(10_000)

cached_train = train_ds.shuffle(10_000).batch(10000).cache()
cached_test = test_ds.batch(1024).cache()

model.fit(cached_train, epochs=epochs)


model.evaluate(cached_test)

identi = tf.data.Dataset.from_tensor_slices(news_data["next_id"])
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(tf.data.Dataset.zip(identi.batch(100), news_ds.batch(100).map(model.candidate_model)))

# Get recommendations.
_, titles = index(tf.constant([
    ["N57072","N60496","N44258","N62771","N49728","N18845","N39737","N45080","",""],
    ['tv', 'sports', 'tv', 'news', 'sports', 'lifestyle', 'movies', 'news', 'news', 'unknown'],
    ['tvnews', 'baseball_mlb', 'tvnews', 'newscrime', 'football_ncaa', 'lifestylebuzz', 'movienews', 'newspolitics', 'newspolitics', 'unknown']
]))