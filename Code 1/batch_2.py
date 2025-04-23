from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, explode, split, lower, round
from pyspark.sql.functions import slice, collect_list


# Tạo SparkSession
spark = SparkSession.builder \
    .appName("Batch Sentiment Analysis") \
    .master("local[2]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Đọc dữ liệu từ HDFS
input_path = "hdfs://localhost:9000/test/output/*.csv"
df = spark.read \
    .option("header", "true") \
    .csv(input_path)

# Kiểm tra dữ liệu
df.show(5, truncate=False)

# 1. Tỷ lệ các loại cảm xúc
total_count = df.count()
sentiment_ratios = df.groupBy("label") \
    .agg(count("*").alias("count")) \
    .withColumn("ratio", round(col("count") / total_count * 100, 2)) \
    .select("label", "count", "ratio")

print("Tỷ lệ các loại cảm xúc:")
sentiment_ratios.show(truncate=False)

# 2. Top 5 từ khóa phổ biến trong từng nhóm cảm xúc
# Tách các từ từ cột clean_text
words_df = df.select("label", explode(split(col("clean_text"), "\\s+")).alias("word")) \
    .filter(col("word") != "")  # Loại bỏ từ rỗng

# Đếm tần suất từ theo nhóm cảm xúc
word_counts = words_df.groupBy("label", "word") \
    .agg(count("*").alias("word_count")) \
    .orderBy("label", col("word_count").desc())

# Lấy top 5 từ cho mỗi nhóm cảm xúc
top_words = word_counts.groupBy("label") \
    .agg(
        collect_list("word").alias("word_list"),
        collect_list("word_count").alias("count_list")
    ) \
    .withColumn("top_words", slice(col("word_list"), 1, 5)) \
    .withColumn("top_counts", slice(col("count_list"), 1, 5)) \
    .select("label", "top_words", "top_counts")

print("Top 5 từ khóa phổ biến trong từng nhóm cảm xúc:")
top_words.show(truncate=False)

# Dừng SparkSession
spark.stop()
