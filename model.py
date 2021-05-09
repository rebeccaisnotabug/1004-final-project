#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
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
    
    # downsample
    df = train.sample(False, 0.01, 42)
    df.createOrReplaceTempView('df')
    
    # StringIndexer
    user_index = StringIndexer(inputCol="user_id", outputCol="indexed_user_id", handleInvalid = 'skip')
    track_index = StringIndexer(inputCol="track_id", outputCol="indexed_track_id", handleInvalid='skip')

    train = user_index.fit_transform(train)
    train = track_index.fit_transform(train)

    validation = user_index.fit_transform(validation)
    validation = track_index.fit_transform(validation)

    test = user_index.fit_transform(test)
    test = track_index.fit_transform(test)


    # ALS
    rank = [5, 10, 20, 30, 50]
    reg_params = [0.001, 0.01, 0.1]
    alpha = [1, 5, 10, 20]

    # distinct users from validation
    user_validation = validation.select('indexed_user_id').distinct()

    # true item
    true_item = validation.select('indexed_user_id', 'indexed_track_id')\
                    .groupBy('indexed_user_id')\
                    .agg(collect_list('indexed_track_id').alias('track_id_val'))
                
    
    # without tuning
    als = ALS(rank = 20,\
              maxIter = 10,\ 
              regParam = 0.01,\ 
              userCol = 'indexed_user_id',\
              itemCol = 'indexed_track_id',\ 
              ratingCol = 'count',\
              coldStartStrategy='drop',\ 
              nonnegative=True)
    
    model = als.fit(train)
    
    # Evaluate the model by computing the MAP on the validation data
    predictions = model.recommendForUserSubset(user_validation,500)
    predictions.createOrReplaceTempView('predictions')
    pred_item= predictions.select('indexed_user_id','recommendations.indexed_track_id')
    
    # convert to rdd for evaluation
    pred_item_rdd = pred_item.join(F.broadcast(true_label), 'indexed_user_id', 'inner') \
                        .rdd \
                        .map(lambda row: (row[1], row[2]))

    # evaluation
    metrics = RankingMetrics(pred_item_rdd)
    map_score = metrics.meanAveragePrecision
    print('map score is: ', map_score)
    
    
 
    # hyperparameter tuning
    """
    for r in rank:
        for reg in reg_params:
            for a in alpha:
                als = ALS(rank = r, maxIter = 10, regParam = reg, userCol = 'indexed_user_id',\
                                itemCol = 'indexed_track_id', ratingCol = 'count', coldStartStrategy='drop', nonnegative=True)
                model = als.fit(train)

                # Evaluate the model by computing the MAP on the validation data
                predictions = model.recommendForUserSubset(user_validation,500)
                predictions.createOrReplaceTempView('predictions')
                pred_item= predictions.select('indexed_user_id','recommendations.indexed_track_id')

                # convert to rdd for evaluation
                pred_item_rdd = pred_item.join(F.broadcast(true_label), 'indexed_user_id', 'inner') \
                        .rdd \
                        .map(lambda row: (row[1], row[2]))

                # evaluation
                metrics = RankingMetrics(pred_item_rdd)
                map_score = metrics.meanAveragePrecision
                print('map score is: ', map_score)

    """

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()
    
    # Call our main routine
    main(spark)


