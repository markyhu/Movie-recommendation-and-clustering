import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.clustering import KMeans
import pandas as pd

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Assignment") \
    .config("spark.local.dir","/fastdata/acq21pp") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.
my_seed = 9904

# Load the data
ratings = spark.read.load('Data/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
tags = spark.read.load('Data/tags.csv', format='csv', inferSchema='true', header='true').cache()
# Divide data into splits and prepare classes
splits = ratings.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], my_seed)
als = ALS(userCol="userId", itemCol="movieId", seed=my_seed, coldStartStrategy="drop")
my_als = ALS(userCol="userId", itemCol="movieId", seed=my_seed, coldStartStrategy="drop", maxIter=1)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
fig, ax = plt.subplots()
kmeans = KMeans(k=10, seed = my_seed)
output_tags = []
hot_rmses = []
cool_rmses = []

for index in range(len(splits)):
    # Define current training and testing splits
    current_testing_split = splits[index].cache()
    current_training_splits = splits[0:index]
    current_training_splits.extend(splits[index+1:])
    current_training_splits = reduce(DataFrame.unionAll, current_training_splits).cache()

    # Extract cool and hot users
    ten_percent = current_training_splits.groupBy('userID').count().count() // 10 # using // to make sure ten_percent is an int
    hot_users = current_training_splits.groupBy('userID').count().orderBy('count', ascending=False).limit(ten_percent)
    cool_users = current_training_splits.groupBy('userID').count().orderBy('count').limit(ten_percent)
    hot_users.createOrReplaceTempView("hotUsers")
    cool_users.createOrReplaceTempView("coolUsers")
    current_testing_split.createOrReplaceTempView("testingSplit")
    hot_users = spark.sql("SELECT * FROM testingSplit WHERE EXISTS(SELECT userId FROM hotUsers WHERE testingSplit.userId == hotUsers.userID)")
    cool_users = spark.sql("SELECT * FROM testingSplit WHERE EXISTS(SELECT userId FROM coolUsers WHERE testingSplit.userId == coolUsers.userID)")

    # Train the model
    model = als.fit(current_training_splits)
    my_model = my_als.fit(current_training_splits)

    # Test the model
    hot_predictions = model.transform(hot_users)
    my_hot_predictions = my_model.transform(hot_users)
    hot_rmse = evaluator.evaluate(hot_predictions)
    my_hot_rmse = evaluator.evaluate(my_hot_predictions)

    cool_predictions = model.transform(cool_users)
    my_cool_predictions = my_model.transform(cool_users)
    cool_rmse = evaluator.evaluate(cool_predictions)
    my_cool_rmse = evaluator.evaluate(my_cool_predictions)

    # Plot the results
    ax.plot(index, hot_rmse, 'ro')
    ax.plot(index, my_hot_rmse, 'rx')
    ax.plot(index, cool_rmse, 'bo')
    ax.plot(index, my_cool_rmse, 'bx')

    # Store the results
    hot_rmses.append([hot_rmse, my_hot_rmse])
    cool_rmses.append([cool_rmse, my_cool_rmse])

    # Perform K-means clustering
    kmodel = kmeans.fit(model.itemFactors)
    transformed = kmodel.transform(model.itemFactors)

    # Extract 2 largest clusters
    cluster_sizes = kmodel.summary.clusterSizes
    largest_cluster_index, second_largest_cluster_index = sorted(range(len(cluster_sizes)), key=lambda i: cluster_sizes[i], reverse=True)[:2]
    largest_cluster = transformed.where(transformed.prediction == largest_cluster_index).toPandas()['id'].tolist()
    second_largest_cluster = transformed.filter(transformed.prediction == second_largest_cluster_index).toPandas()['id'].tolist()

    # Connect with tags table
    largest_cluster = tags[tags.movieId.isin(largest_cluster)]
    second_largest_cluster = tags[tags.movieId.isin(second_largest_cluster)]

    # Extract top and bottom tags for both clusters
    top_tag1 = largest_cluster.groupBy('tag').count().sort('count', ascending=False).first()['tag']
    top_tag2 = second_largest_cluster.groupBy('tag').count().sort('count', ascending=False).first()['tag']
    bot_tag1 = largest_cluster.groupBy('tag').count().sort('count').first()['tag']
    bot_tag2 = second_largest_cluster.groupBy('tag').count().sort('count').first()['tag']

    # Store the results
    output_tags.append([top_tag1, bot_tag1])
    output_tags.append([top_tag2, bot_tag2])

print(hot_rmses)
print(cool_rmses)
print(output_tags)
plt.savefig("Output/RMSE.png")
spark.stop()
