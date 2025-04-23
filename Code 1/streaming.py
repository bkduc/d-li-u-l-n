from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf, count
from pyspark.sql.types import StringType
from pyspark.sql.functions import split, to_timestamp, trim
import time

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("Real-time Text Sentiment Analysis") \
    .master("local[2]") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Danh sách từ khóa cảm xúc và Broadcast
positive_words = ["love", "great", "awesome", "excellent", "fantastic", "amazing", "brilliant"]
negative_words = ["bad", "worst", "hate", "boring", "terrible", "awful", "poor"]

positive_bc = spark.sparkContext.broadcast(positive_words)
negative_bc = spark.sparkContext.broadcast(negative_words)

# Đọc dữ liệu từ socket
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load()

split_data = lines.withColumn("text", split(col("value"), "\\|").getItem(0)) \
                  .withColumn("timestamp_str", split(col("value"), "\\|").getItem(1)) \
                  .withColumn("timestamp", to_timestamp(trim(col("timestamp_str")), "yyyy-MM-dd HH:mm:ss"))

# Làm sạch văn bản
cleaned_data = split_data.withColumn("clean_text", lower(
    regexp_replace(col("text"), "[^a-zA-Z\\s]", "")
))

# Hàm gắn nhãn cảm xúc
def label_sentiment(text):
    if text is None:
        return "neutral"
    words = text.split()
    if any(word in words for word in positive_bc.value):
        return "positive"
    elif any(word in words for word in negative_bc.value):
        return "negative"
    return "neutral"

label_udf = udf(label_sentiment, StringType())

# Gắn nhãn cảm xúc
labeled_data = cleaned_data \
    .withColumn("label", label_udf("clean_text"))

# Thống kê theo processing time (1 phút)
windowed_counts = labeled_data.groupBy(
    col("label")
).agg(count("*").alias("count"))

# Xuất thống kê ra console mỗi 1 phút
query_windowed = windowed_counts.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="1 minute") \
    .start()

# Hàm xử lý từng batch để lưu mỗi dòng thành một file CSV trên HDFS
def save_to_hdfs(batch_df, batch_id):
    start_time = time.time()
    try:
        if batch_df.count() > 0:
            print(f"\n-------------------------------------------")
            print(f"Batch: {batch_id}")
            print(f"-------------------------------------------")
            batch_df.groupBy("label").count().show(truncate=False)

            batch_df = batch_df.coalesce(1)
            for row in batch_df.collect():
                single_row_df = spark.createDataFrame([row], batch_df.schema)
                timestamp = int(time.time() * 1000)
                file_path = f"hdfs://localhost:9000/test/output/row_{batch_id}_{timestamp}.csv"
                single_row_df.write \
                    .format("csv") \
                    .option("header", "true") \
                    .mode("overwrite") \
                    .save(file_path)
    except Exception as e:
        print(f"Error saving batch {batch_id} to HDFS: {e}")
    finally:
        elapsed_time = time.time() - start_time
        print(f"Batch {batch_id} processed in {elapsed_time:.2f} seconds")

# Lưu mỗi dòng vào HDFS dưới dạng file CSV riêng
hdfs_output_query = labeled_data.writeStream \
    .outputMode("append") \
    .foreachBatch(save_to_hdfs) \
    .trigger(processingTime="1 minute") \
    .start()

# Giữ tiến trình hoạt động
try:
    query_windowed.awaitTermination()
except KeyboardInterrupt:
    print("Stopping queries...")
    query_windowed.stop()
    hdfs_output_query.stop()
    spark.stop()
    print("Queries stopped")
