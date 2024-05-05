import numpy as np
import pandas as pd
import time
from pyspark.sql import *
from pyspark.sql.types import *
from sklearn.neighbors import KDTree
from pyspark.sql.functions import array, sqrt, pow, col
from pyspark.sql import types as T
from pyspark.sql import functions as F

#spark = SparkSession.builder.getOrCreate()
spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()

sc = spark.sparkContext

IDENTIFIER = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


f = open(f'./exp_logs/{IDENTIFIER}.txt', 'w')

FILE_PATH = "hdfs://vm1:9000/user/azureuser/dataset/birch/birch3.txt"
print('FILE_PATH:', FILE_PATH, file=f)

a1 = pd.read_csv(FILE_PATH, delimiter=r'\s+', header=None, names=["X", "Y"])

# spark dataframe
df = spark.createDataFrame(a1)

start_time = time.time()

dfinput = df.select(array("X", "Y").alias("point")) \
    .rdd.map(lambda row: (row["point"])) \
    .zipWithIndex() \
    .toDF(["point", "id"]) \
    .select("id", "point") \
    .cache()

# repartition with 20
rdd_input = dfinput.rdd.repartition(20).cache()

schema = T.StructType([
    T.StructField("neighbour_id", T.LongType(), nullable = True),
    T.StructField("core_id", T.LongType(), nullable = True),
    T.StructField("neighbour_point", T.ArrayType(T.DoubleType(), containsNull = False), nullable = True),
    T.StructField("core_point", T.ArrayType(T.DoubleType(), containsNull = False), nullable = True)
])

dfpair_raw = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

print("after data preparation: ", time.time() - start_time, file=f)

#repartition the dataframe
partition_num = 10
df_repar = dfinput.repartition(partition_num)

#broadcast to different partitions
eps = 1000
min_samples = 20

for i in range(0,partition_num):
    # in i-th partition, partition in dataframe, num is 10, transfer df into rdd
    rdd_repar_i = df_repar.rdd.mapPartitionsWithIndex(
        lambda idx, iterator: iterator if (idx == i ) else iter([]) )
    repar_i = rdd_repar_i.collect()
    idxs_i = [x["id"] for x in repar_i]
    points_i = [x["point"] for x in repar_i]
    # create a tree for every point in each partition
    tree_i = KDTree(np.array(points_i), leaf_size = 500, metric = 'minkowski') # minkowski: standard Euclidean distance when p = 2 
    # note: leaf_size <= n_points <= 2 * leaf_size, except in the case that n_samples < leaf_size
    broad_i = sc.broadcast((idxs_i,points_i,tree_i))
    
    def fn(iterator):
        list_co_nei = [] #neighbour_id,core_id,neighbour_point,core_point
        idxs_i,points_i,tree_i = broad_i.value
        for row in iterator:
            core_id = row["id"]
            core_point = row["point"]
            # in KDTree: 1st idx is the idxs of neighbours within r; 2nd idx is diatance
            index = tree_i.query_radius(np.array([core_point]), r = eps)[0] 
            for j in index:
                list_co_nei.append([idxs_i[j],core_id,points_i[j],core_point])
        return iter(list_co_nei)

    # repartition in rdd with partition num 20 and merge the core points with thier neighbours in tree
    dfpair_raw_i = spark.createDataFrame(rdd_input.mapPartitions(fn)).toDF("neighbour_id","core_id","neighbour_point","core_point")
    dfpair_raw = dfpair_raw.union(dfpair_raw_i)

    dfpair = dfpair_raw.where(sqrt(pow(col("core_point")[0] - col("neighbour_point")[0], 2) + pow(col("core_point")[1] - col("neighbour_point")[1], 2)) < eps) \
                       .cache()
    
    dfcore = dfpair.groupBy("core_id").agg(
    F.first("core_point").alias("core_point"),
    F.count("neighbour_id").alias("neighbour_cnt"),
    F.collect_list("neighbour_id").alias("neighbour_ids")
    ).filter(F.col("neighbour_cnt") >= min_samples) \
     .cache()

print("after building KDtree and calculating core points: ", time.time() - start_time, file=f)
# core points of temporary clusters

dfpair_join = dfcore.select("core_id").join(dfpair, ["core_id"], "inner")
df_fids = dfcore.select(dfcore["core_id"].alias("neighbour_id"))
dfpair_core = df_fids.join(dfpair_join, ["neighbour_id"], "inner")
rdd_core = dfpair_core.groupBy("core_id").agg(
    F.min("neighbour_id").alias("min_core_id"),
    F.collect_set("neighbour_id").alias("core_id_set")
).rdd.map(lambda row: (row["min_core_id"], set(row["core_id_set"])))

rdd_core.cache()

print("before_dbscan, number of core points: ", rdd_core.count(), file=f)

def mergeSets(list_set):
    result = []
    while  len(list_set)>0 :
        cur_set = list_set.pop(0)
        intersect_idxs = [i for i in list(range(len(list_set)-1,-1,-1)) if cur_set&list_set[i]]
        while  intersect_idxs :
            for idx in intersect_idxs:
                cur_set = cur_set|list_set[idx]

            for idx in intersect_idxs:
                list_set.pop(idx)
                
            intersect_idxs = [i for i in list(range(len(list_set)-1,-1,-1)) if cur_set&list_set[i]]
        
        result = result+[cur_set]
    return result


def mergeRDD(rdd, partition_num):
    def fn(iterator):
        list_set = [x[1] for x in iterator]
        list_set_merged = mergeSets(list_set)
        merged_core = [(min(x),x) for x in list_set_merged] 
        return(iter(merged_core))
    rdd_merged = rdd.partitionBy(partition_num).mapPartitions(fn)
    return rdd_merged

#after partitioning rdd_core, merge them, and reduce num of partitions, finally in one partition
#if scale large, can merge into small num of partitions
#rdd: (min_core_id,core_id_set)
for par_num in (16,8,4,2,1):
    rdd_core = mergeRDD(rdd_core,par_num)
    
rdd_core.cache()

print("after dbscan, number of core points: ",rdd_core.count(), file=f)

print("after merging and dbscan: ", time.time() - start_time, file=f)

dfcluster_ids = spark.createDataFrame(
    rdd_core.flatMap(lambda t: [(t[0], core_id) for core_id in t[1]])).toDF("cluster_id","core_id")

dfclusters =  dfcore.join(dfcluster_ids, "core_id", "left")


# find the representative of each cluster and the # of points in each cluster
rdd_cluster = dfclusters.rdd.map(
    lambda row: (row["cluster_id"],(row["core_point"],row["neighbour_cnt"],set(row["neighbour_ids"])))
)

def reduce_fn(a,b):
    id_set = a[2]|b[2] # merge neighbour_ids with the same cluster_id
    result = (a[0],a[1],id_set) if a[1]>=b[1] else (b[0],b[1],id_set) # select larger neighbour_cnt
    return result

rdd_merge = rdd_cluster.reduceByKey(reduce_fn)

def map_fn(r):
    cluster_label = r[0]
    final_core_point = r[1][0]
    neighbour_points_cnt = r[1][1]
    id_set = list(r[1][2])
    cluster_points_cnt = len(id_set)
    return (cluster_label,final_core_point,neighbour_points_cnt,cluster_points_cnt,id_set)

df_merge = spark.createDataFrame(rdd_merge.map(map_fn)) \
     .toDF("cluster_label","final_core_point","neighbour_points_cnt","cluster_points_cnt","cluster_points_ids") \
     .cache()

# find cluster label for each point
rdd_points_cluster = df_merge.select("cluster_label", "cluster_points_ids").rdd.flatMap(
    lambda t: [(x, t["cluster_label"]) for x in t["cluster_points_ids"]]
)

df_points_cluster = spark.createDataFrame(rdd_points_cluster, ["id", "cluster_label"])
df_result_raw = dfinput.join(df_points_cluster, "id", "left")
# the cluetser_label of noise point is -1
df_result = df_result_raw.fillna(-1, subset=["cluster_label"])

df_result = df_result.selectExpr("id", "cluster_label", "point[0] as X", "point[1] as Y").cache()
df_result.show()

print("after finding the final cluster label: ", time.time() - start_time, file=f)

# plot the result of DBSCAN clustering
# pd_result = df_result.toPandas()
# pd_result.plot.scatter('X','Y', s = 10,
#     c = list(pd_result['cluster_label']),cmap = 'rainbow',colorbar = False,
#     alpha = 0.6,title = 'sparksql DBSCAN cluster result')


f.close()
spark.stop()