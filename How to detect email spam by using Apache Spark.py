# By Symon Kimitei
# NAÏVE BAYES CLASSIFIER
# USING APACHE SPARK TO DETECT EMAIL SPAM
#==============================================================================
#  Import all the necessary libraries
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.ml.feature import  Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import  IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Converted the file into a .CSV file and Uploaded the data file to the server.
X_Rdd = spark.read.load("file:///home/usrmkang/skimitei/assign4/smsspamcollection.csv", format="csv", sep=",", inferSchema="true", header="false")

# Show the first five messages and labels
X_Rdd.show(5, truncate = True)
# Rename column names as "label" and "message"
X_Rdd = X_Rdd.withColumnRenamed('_c0', 'label').withColumnRenamed('_c1', 'message')

# Display the first five rows
X_Rdd.show(5, truncate = True)

# Creates a new temporary view using a SparkDataFrame in the Spark Session
X_Rdd.createOrReplaceTempView('temp')

# Convert the class label to binary with ham as 1.0 and spam as 0
X_Rdd = spark.sql('select case label when "ham" then 1.0  else 0 end as label, message from temp')

# Display the first five rows
X_Rdd.show(5, truncate = True)

#  Next Let us tokenize the messages. To tokenize is to break the message into
# individual words
tokenizer = Tokenizer(inputCol="message", outputCol="words")
Words_Rdd = tokenizer.transform(X_Rdd)

#  Display the first ten rows of the DataFrame with 
#  three columns: label, message and the tokenized words
Words_Rdd.show(10)

# Apply CountVectorizer which  CountVectorizer converts the word tokens 
# into vectors of token counts. 
token_counts = CountVectorizer (inputCol="words", outputCol="new_features")
model =token_counts.fit(Words_Rdd)
featurized_Rdd = model.transform(Words_Rdd)
featurized_Rdd.show(10)

# Apply Term Frequency–Inverse Document Frequency (TF-IDF)
idf = IDF(inputCol="new_features", outputCol="features")
idfModel = idf.fit(featurized_Rdd)
rescaled_Rdd = idfModel.transform(featurized_Rdd)
rescaled_Rdd.select("label", "features").show(10)

# Split the dataset into Training data  = 80% and Testing data = 20%
# Set seed for reproducibility zero
seed = 0  
train_df, test_df = rescaled_Rdd.randomSplit([0.8,0.2],seed)

# Number of records of each dataframe
train_df.count()
test_df.count()

# FIT the Naïve Bayes classifier
nb = NaiveBayes()
paramGrid_nb = ParamGridBuilder().addGrid(nb.smoothing, np.linspace(0.3, 10, 5)).build()
# Apply five-fold cross validation
cross_val_nb = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid_nb,evaluator=BinaryClassificationEvaluator(),numFolds= 5) 
cv_model_nb = cross_val_nb.fit(train_df)

# Display the average area under the ROC curve values 
# using the five smoothing values
cv_model_nb.avgMetrics

# Making Predictions
predictions_nb = cv_model_nb.transform(test_df)
predictions_nb.select('label', 'prediction').show(10)
# Calculate the number of correct and incorrect predictions
predictions_nb.groupBy('label','prediction').count().show()

# Calculate the area under the ROC curve
areaUnderROC_eval_nb = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label', metricName='areaUnderROC')
areaUnderROC_eval_nb.evaluate(predictions_nb)
