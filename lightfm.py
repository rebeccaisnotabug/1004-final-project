import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

from time import time

import json
from itertools import islice

import numpy as np
import pandas as pd
from collections import Counter

from scipy.sparse import csr_matrix

def main(spark):
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    validation_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    test_path = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'
    
    train = spark.read.parquet(train_path)
    train.createOrReplaceTempView('train')
    
    validation = spark.read.parquet(validation_path)
    validation.createOrReplaceTempView('validation')
    
    test = spark.read.parquet(test_path)
    test.createOrReplaceTempView('test')
    
    #Downsample
    downsample_path = 'hdfs:/user/ss14359/train10.parquet'
    df = spark.read.parquet(downsample_path)
    df.createOrReplaceTempView('df')
    
    # StringIndexer
    user_index = StringIndexer(inputCol="user_id", outputCol="indexed_user_id", handleInvalid = 'skip')
    track_index = StringIndexer(inputCol="track_id", outputCol="indexed_track_id", handleInvalid='skip')
    
    user_string_indexer = user_index.fit(df)
    track_string_indexer = track_index.fit(df)
   
    df = user_string_indexer.transform(df)
    df = track_string_indexer.transform(df)

    validation = user_string_indexer.transform(validation)
    validation = track_string_indexer.transform(validation)

    test = user_string_indexer.transform(test)
    test = track_string_indexer.transform(test)
    
    for rank in [1, 5, 10]:
        map_score, time = train_and_test(df, validation, rank, 0.01, 10, top=500, model_type='warp')

def transform_interaction(df, test_percent):
    
    interaction = pd.pivot_table(df, index='indexed_user_id', columns='indexed_track_id', values='rating')
    interaction = interaction.fillna(0)
    
    all_csr = csr_matrix(interaction.values)
    
    (train_matrix, test_matrix) = random_train_test_split(all_csr, test_percentage=test_percent)
    
    return (train_matrix, test_matrix)


def lightfm_train(train, rank, regParam, maxIter, model_type='warp'):
    
    if model_type == 'bpr':
        model = LightFM(loss='bpr',
                no_components=rank,
                user_alpha=regParam)
        
    else:    
        model = LightFM(loss='warp',
                no_components=rank,
                user_alpha=regParam)

    model = model.fit(train, epochs=maxIter,verbose=False)
    
    return model

def train_and_test(train, test, rank, regParam, maxIter, top=500, model_type='warp'):
    
    start = time()
    
    model = lightfm_train(train, rank, regParam, maxIter, model_type='warp')
    p_at_k = precision_at_k(model, test, k=top).mean()
    
    end = time()
    time = round(end - start, 5)
    
    print('Model with maxIter = {}, reg = {}, rank = {} complete'.format(maxIter,regParam,rank))
    print('Precision at K:', p_at_k)
    print('Time used:', t)
    
    return p_at_k, time

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('lightfm').getOrCreate()
    
    # Call our main routine
    main(spark)

