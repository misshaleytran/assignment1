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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_directory, spark, file_type="lms"):
    # Define file configuration
    file_config = {
        "lms": {
            "source_file": "data/lms_loan_daily.csv",
            "output_prefix": "bronze_lms"  
        },
        "clickstream": {
            "source_file": "data/feature_clickstream.csv", 
            "output_prefix": "bronze_clickstream"
        },
        "attributes": {
            "source_file": "data/features_attributes.csv",
            "output_prefix": "bronze_attributes"
        },
        "financials": {
            "source_file": "data/features_financials.csv",
            "output_prefix": "bronze_financials"
        }
    }
    
    # Validate file type
    if file_type not in file_config:
        print(f"Error: Unknown file_type '{file_type}'. Must be one of: {list(file_config.keys())}")
        return None
    
    # Get configuration for current file type
    config = file_config[file_type]
    csv_file_path = config["source_file"]
    output_prefix = config["output_prefix"]
    
    print(f"Processing bronze table for {file_type} on {snapshot_date_str}")
    print(f"Source file: {csv_file_path}")
    
    # Check if source file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: Source file not found: {csv_file_path}")
        return None
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = output_prefix + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df