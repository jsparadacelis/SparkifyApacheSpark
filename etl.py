import configparser
from datetime import datetime
import os

from pyspark.sql import SparkSession, types
from pyspark.sql.functions import udf, col, to_timestamp, monotonically_increasing_id
from pyspark.sql.functions import (
    year,
    month,
    dayofmonth,
    hour,
    weekofyear,
    date_format,
    dayofweek
)
from pyspark.sql.types import StructType, StructField


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Extracs information from song json files to build a Dataframe to
    create song table and artist table.

    Args:
        spark (SparkSession): instance of spark session.
        input_data (str): base path to get the data.
        output_data (str): base path to write the parquet files.
    """
    # get filepath to song data file
    song_data = f"{input_data}song_data/*/*/*/*.json"

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(
        "song_id",
        "title",
        "artist_id",
        "year",
        "duration"
    ).distinct()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite")\
        .partitionBy("year", "artist_id")\
        .parquet(f"{output_data}/songs_table.parquet")

    # extract columns to create artists table
    artists_table = df.select(
        "artist_id",
        "artist_name",
        "artist_location",
        "artist_latitude",
        "artist_longitude"
    ).distinct()

    # write artists table to parquet files
    artists_table.write.mode("overwrite")\
        .parquet(f"{output_data}/artists_table.parquet")


def process_log_data(spark, input_data, output_data):
    """Extracs information from log_data json files to build a Dataframe to
    create tables for users, time and sonplays.

    Args:
        spark (SparkSession): instance of spark session.
        input_data (str): base path to get the data.
        output_data (str): base path to write the parquet files.
    """
    # get filepath to log data file
    log_data = f"{input_data}log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)

    # extract columns for users table
    users_table = df.select(
        "userId",
        "firstName",
        "lastName",
        "gender",
        "level"
    ).distinct()

    # write users table to parquet files
    users_table.write.parquet(f"{output_data}/users_table.parquet")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn("timestamp", get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
        col("datetime"),
        hour("datetime").alias("hour"),
        dayofmonth("datetime").alias("day"),
        weekofyear("datetime").alias("week"),
        month("datetime").alias("month"),
        year("datetime").alias("year"),
        dayofweek("datetime").alias("weekday")
    ).distinct()

    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite")\
        .partitionBy("year", "month")\
        .parquet(f"{output_data}/time_table.parquet")

    # read in song data to use for songplays table
    song_data = f"{input_data}song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data)

    # extract columns from joined
    # song and log datasets to create songplays table
    df = df.join(song_df, song_df.title == df.song)

    songplays_table = df.select(
        col("ts").alias("ts"),
        col("userId").alias("user_id"),
        col("level").alias("level"),
        col("song_id").alias("song_id"),
        col("artist_id").alias("artist_id"),
        col("ssessionId").alias("session_id"),
        col("location").alias("location"),
        col("userAgent").alias("user_agent"),
        col("year").alias("year"),
        month("datetime").alias("month")
    )

    songplays_table = songplays_table.selectExpr("ts as start_time")
    songplays_table.select(
        monotonically_increasing_id().alias('songplay_id')).collect()

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite")\
        .partitionBy('year', 'month')\
        .parquet(f"{output_data}/songplays.parquet")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://<your-s3-bucket>"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
