# %%
from pyspark.sql import SparkSession
spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()
    
sc = spark.sparkContext


print('Hello World!')

spark.stop()