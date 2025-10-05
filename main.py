import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# Define DPD and MOB parameters here - FIXED
dpd = 30
mob = 6

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Define the 4 files to process
files_to_process = [
    "feature_clickstream.csv",
    "features_attributes.csv", 
    "features_financials.csv",
    "lms_loan_daily.csv"
]

# create bronze datalake directories for each file type
bronze_directories = {
    "clickstream": "datamart/bronze/clickstream/",
    "attributes": "datamart/bronze/attributes/",
    "financials": "datamart/bronze/financials/",
    "lms": "datamart/bronze/lms/"
}

for directory in bronze_directories.values():
    if not os.path.exists(directory):
        os.makedirs(directory)

# run bronze backfill for all files
for date_str in dates_str_lst:
    for file_type, directory in bronze_directories.items():
        utils.data_processing_bronze_table.process_bronze_table(
            date_str, directory, spark, file_type=file_type
        )


# create silver datalake directories for each file type
silver_directories = {
    "clickstream": "datamart/silver/clickstream/",
    "attributes": "datamart/silver/attributes/",
    "financials": "datamart/silver/financials/", 
    "lms": "datamart/silver/lms/"
}

for directory in silver_directories.values():
    if not os.path.exists(directory):
        os.makedirs(directory)

# run silver backfill for all files
for date_str in dates_str_lst:
    for file_type, silver_directory in silver_directories.items():
        bronze_directory = bronze_directories.get(file_type, bronze_directories["lms"])
        utils.data_processing_silver_table.process_silver_table(
            date_str, bronze_directory, silver_directory, spark, file_type=file_type
        )


# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill for labels and features 
for date_str in dates_str_lst:
    print(f"\n{'='*50}")
    print(f"Processing gold tables for {date_str}")
    print(f"{'='*50}")
    
    # Process labels
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str, 
        silver_directories,
        gold_label_store_directory, 
        spark, 
        dpd=dpd, 
        mob=mob
    )
    
    # Process features 
    utils.data_processing_gold_table.process_features_gold_table(
        date_str, 
        silver_directories, 
        gold_feature_store_directory, 
        spark, 
        dpd,  
        mob   
    )


# Verify results
print(f"\n{'='*50}")
print("FINAL VERIFICATION")
print(f"{'='*50}")

folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.parquet'))]
if files_list:
    df = spark.read.option("header", "true").parquet(*files_list)
    print("label_store row_count:", df.count())
    print("label_store unique customers:", df.select("customer_id").distinct().count())
    df.show()
else:
    print("No label store files found")

folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*.parquet'))]
if files_list:
    df = spark.read.option("header", "true").parquet(*files_list)
    print("feature_store row_count:", df.count())
    print("feature_store unique customers:", df.select("customer_id").distinct().count())
    df.show()
else:
    print("No feature store files found")
    