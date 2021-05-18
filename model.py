#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Use getpass to obtain user netID
import getpass
import pandas as pd
import numpy as np


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
from pyspark.sql.functions  import collect_list
import itertools

from time import time

def main(spark):
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    validation_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    #test_path = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'
    
    #df = spark.read.parquet(train_path)
    #df.createOrReplaceTempView('df')
    #df = df.repartition(12000)
    
    validation = spark.read.parquet(validation_path)
    validation.createOrReplaceTempView('validation')
    
    #test = spark.read.parquet(test_path)
    #test.createOrReplaceTempView('test')
    
    #Downsample
    downsample_path = 'hdfs:/user/ss14359/train25.parquet'
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
    
    #test = user_string_indexer.transform(test)
    #test = track_string_indexer.transform(test)
    
    
    # ALS
    rank = [40]
    reg_params = [1]
    alpha = [10]
    param_choice = itertools.product(rank, reg_params, alpha)

    # distinct users from validation
    user_validation = validation.select('indexed_user_id').distinct()
    #user_test = test.select('indexed_user_id').distinct()

    # true item
    #true_item = test.select('indexed_user_id', 'indexed_track_id')\
    #true_item = validation.select('indexed_user_id', 'indexed_track_id')\
    true_item = validation.select('indexed_user_id', 'indexed_track_id')\
                    .groupBy('indexed_user_id')\
                    .agg(collect_list('indexed_track_id').alias('track_id_val'))
                
    
    # hyperparameter tuning
    for i in param_choice:
        #start = time()
        als = ALS(rank = i[0], 
                  maxIter = 20, 
                  regParam = i[1], 
                  alpha = i[2],
                  userCol = 'indexed_user_id',
                  itemCol = 'indexed_track_id',
                  ratingCol = 'count', 
                  implicitPrefs = True, 
                  coldStartStrategy='drop', 
                  nonnegative=False)
        
        
        model = als.fit(df)
        print('Finish training for {} combination'.format(i))
        
        # Evaluate the model by computing the MAP on the validation data
        predictions = model.recommendForUserSubset(user_validation, 500)
        #predictions = model.recommendForUserSubset(user_test, 500)
        predictions.createOrReplaceTempView('predictions')
        pred_item = predictions.select('indexed_user_id','recommendations.indexed_track_id')

        # convert to rdd for evaluation
        pred_item_rdd = pred_item.join(F.broadcast(true_item), 'indexed_user_id', 'inner') \
                            .rdd \
                            .map(lambda row: (row[1], row[2]))

        # evaluation
        metrics = RankingMetrics(pred_item_rdd)
        #map_score = metrics.meanAveragePrecision
        precision = metrics.precisionAt(500)
        ndcg = metrics.ndcgAt(500)
        #print('map score is: ', map_score)
        print('precision is: ', precision)
        print('ndcg is: ',ndcg)
        #time_taken = time() - start
        #print('Time taken: ' + str(time_taken))


    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    # Call our main routine
    main(spark)
