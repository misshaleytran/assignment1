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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark, file_type="lms"):
    # Define file configuration
    file_config = {
        "lms": {
            "bronze_prefix": "bronze_lms", 
            "silver_prefix": "silver_lms", 
            "cleaning_function": clean_lms_data 
        },
        "clickstream": {
            "bronze_prefix": "bronze_clickstream", 
            "silver_prefix": "silver_clickstream",
            "cleaning_function": clean_clickstream_data
        },
        "attributes": {
            "bronze_prefix": "bronze_attributes",
            "silver_prefix": "silver_attributes", 
            "cleaning_function": clean_attributes_data
        },
        "financials": {
            "bronze_prefix": "bronze_financials",
            "silver_prefix": "silver_financials",
            "cleaning_function": clean_financials_data
        }
    }
    
    # Validate file type
    if file_type not in file_config:
        print(f"Error: Unknown file_type '{file_type}'. Must be one of: {list(file_config.keys())}")
        return None
    
    # Get configuration for current file type
    config = file_config[file_type]
    bronze_prefix = config["bronze_prefix"]
    silver_prefix = config["silver_prefix"]
    cleaning_function = config["cleaning_function"]
    
    print(f"Processing silver table for {file_type} on {snapshot_date_str}")
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = bronze_prefix + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    
    # Check if bronze file exists
    if not os.path.exists(filepath):
        print(f"Error: Bronze file not found: {filepath}")
        return None
        
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # Apply file-specific cleaning
    df_clean = cleaning_function(df, snapshot_date_str)
    
    # save silver table - IRL connect to database to write
    partition_name = silver_prefix + "_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df_clean.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df_clean


def normalize_dataframe(df, file_name):
    """
    Shared normalization logic for all file types
    """
    print(f"Normalizing data: {file_name}")
    initial_count = df.count()
    
    # 1. Normalize column names
    for col_name in df.columns:
        normalized_col = col_name.strip().lower().replace(' ', '_').replace('-', '_')
        if col_name != normalized_col:
            df = df.withColumnRenamed(col_name, normalized_col)
    
    print("Columns normalized")
    
    # 2. Normalize string cells - trim and lowercase
    string_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))
        df = df.withColumn(col_name, F.lower(F.col(col_name)))
    
    print(f"Normalized {len(string_cols)} string columns")
    
    # 3. Remove duplicates on customer_id + snapshot_date (if both exist)
    if all(col in df.columns for col in ['customer_id', 'snapshot_date']):
        df_clean = df.dropDuplicates(["customer_id", "snapshot_date"])
        duplicate_count = initial_count - df_clean.count()
        print(f"Removed {duplicate_count} duplicate rows")
    else:
        df_clean = df
        print("Skipped duplicate removal - required columns not found")
    
    return df_clean


def clean_lms_data(df, snapshot_date_str):
    """
    Clean and augment lms data
    """
    print(f"Cleaning lms data for {snapshot_date_str}")
    df_clean = normalize_dataframe(df, f"lms_{snapshot_date_str}")
    
    # Define type mapping for lms data
    column_type_map = {
        "loan_id": StringType(),
        "customer_id": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": DoubleType(),
        "due_amt": DoubleType(),
        "paid_amt": DoubleType(),
        "overdue_amt": DoubleType(),
        "balance": DoubleType(),
        "snapshot_date": DateType(),
    }

    # Apply type conversions
    for column, new_type in column_type_map.items():
        if column in df_clean.columns:
            df_clean = df_clean.withColumn(column, col(column).cast(new_type))
            print(f"Type converted {column} to {new_type}")

    # augment data: add month on book
    df_clean = df_clean.withColumn("mob", col("installment_num").cast(IntegerType()))
    print("Added mob (month on book) column")

    # augment data: add days past due
    df_clean = df_clean.withColumn(
        "installments_missed", 
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    
    df_clean = df_clean.withColumn(
        "first_missed_date", 
        F.when(
            col("installments_missed") > 0, 
            F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
        ).cast(DateType())
    )
    
    df_clean = df_clean.withColumn(
        "dpd", 
        F.when(
            col("overdue_amt") > 0.0, 
            F.datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0).cast(IntegerType())
    )
    
    print("Added installments_missed, first_missed_date, and dpd columns")
    
    return df_clean


def clean_clickstream_data(df, snapshot_date_str):
    """
    Clean clickstream data
    """
    print(f"Cleaning clickstream data for {snapshot_date_str}")
    df_clean = normalize_dataframe(df, f"clickstream_{snapshot_date_str}")
    
    # Convert all feature columns to numeric
    feature_cols = [col_name for col_name in df_clean.columns if col_name.startswith('fe_')]
    for col_name in feature_cols:
        df_clean = df_clean.withColumn(col_name, F.col(col_name).cast("double"))
    
    print(f"{len(feature_cols)} feature columns converted to numeric")
    
    return df_clean


def clean_attributes_data(df, snapshot_date_str):
    """
    Clean attributes data
    """
    print(f"Cleaning attributes data for {snapshot_date_str}")
    df_clean = normalize_dataframe(df, f"attributes_{snapshot_date_str}")
    
    # Clean specific columns
    if 'age' in df_clean.columns:
        print("Cleaning age column")
        
        # First remove non-numeric characters from age
        df_clean = df_clean.withColumn(
            "age_clean",
            F.regexp_replace(F.col("age"), r'[^\d]', '')
        )
        
        # Convert to integer and validate range
        df_clean = df_clean.withColumn(
            "age_temp",
            F.when(
                (F.col("age_clean") != '') & 
                (F.col("age_clean").cast("int") >= 1) & 
                (F.col("age_clean").cast("int") <= 120),
                F.col("age_clean").cast("int")
            ).otherwise(F.lit(30))  # Default to 30 if invalid
        )
        
        # Calculate median of valid ages for better default
        valid_ages = df_clean.filter(
            (F.col("age_clean") != '') & 
            (F.col("age_clean").cast("int") >= 1) & 
            (F.col("age_clean").cast("int") <= 120)
        ).select("age_temp").rdd.flatMap(lambda x: x).collect()
        
        if valid_ages:
            median_age = float(np.median(valid_ages))
            print(f"Median age from valid values: {median_age}")
            
            # Fill invalid ages with median
            df_clean = df_clean.withColumn(
                "age",
                F.when(
                    (F.col("age_clean") != '') & 
                    (F.col("age_clean").cast("int") >= 1) & 
                    (F.col("age_clean").cast("int") <= 120),
                    F.col("age_clean").cast("int")
                ).otherwise(median_age)
            )
        else:
            df_clean = df_clean.withColumn("age", F.col("age_temp"))
        
        df_clean = df_clean.drop("age_clean", "age_temp")
    
    if 'occupation' in df_clean.columns:
        print("Cleaning occupation column")
        df_clean = df_clean.withColumn(
            "occupation",
            # First, clean all non-null values
            F.when(
                F.col("occupation").isNotNull(),
                F.regexp_replace(
                    F.regexp_replace(
                        F.regexp_replace(
                            F.regexp_replace(F.col("occupation"), r'[^\w_]', ''),  # Remove non-word chars
                            r'_{2,}', '_'  # Collapse multiple underscores
                        ),
                        r'^_+|_+$', ''  # Trim leading/trailing underscores
                    ),
                    r'^$', 'unknown'  # Handle empty strings after cleaning
                )
            )
        ).withColumn(
            "occupation",
            # Final null handling
            F.coalesce(F.col("occupation"), F.lit("unknown"))
        )
    
    if 'name' in df_clean.columns:
        print("Cleaning name column")
        df_clean = df_clean.withColumn(
            "name", 
            F.regexp_replace(F.col("name"), r'\s+', ' ')
        )
    
    if 'ssn' in df_clean.columns:
        print("Cleaning SSN column")
        df_clean = df_clean.withColumn(
            "ssn",
            F.when(
                F.col("ssn").rlike(r'^\d{3}-\d{2}-\d{4}$'),
                F.col("ssn")
            ).otherwise("unknown")
        )
    
    return df_clean


def clean_financials_data(df, snapshot_date_str):
    """
    Clean financials data
    """
    print(f"Cleaning financials data for {snapshot_date_str}")
    df_clean = normalize_dataframe(df, f"financials_{snapshot_date_str}")
    
    # Clean specific numeric columns (double type)
    numeric_columns = [
        'annual_income', 'monthly_inhand_salary', 'interest_rate', 
        'changed_credit_limit', 'outstanding_debt', 'credit_utilization_ratio',
        'total_emi_per_month', 'amount_invested_monthly'
    ]
    
    print("Cleaning numeric columns (double)")
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            # Remove all non-numeric characters except decimal points and minus signs
            df_clean = df_clean.withColumn(
                col_name,
                F.when(
                    F.col(col_name).isNotNull() & (F.col(col_name) != ''),
                    F.regexp_replace(F.col(col_name), r'[^\d\.\-]', '')
                ).otherwise(F.col(col_name))
            )
            
            # Convert to double - only set to 0.0 when blank, null, or actually 0
            df_clean = df_clean.withColumn(
                col_name,
                F.when(
                    (F.col(col_name).isNull()) | (F.col(col_name) == '') | (F.col(col_name) == '0'),
                    F.lit(0.0)
                ).when(
                    F.col(col_name).rlike(r'^-?\d*\.?\d+$'),
                    F.col(col_name).cast("double")
                ).otherwise(
                    # For values that have numbers but invalid format, try to extract first valid number
                    F.coalesce(
                        F.regexp_extract(F.col(col_name), r'(-?\d*\.?\d+)', 0).cast("double"),
                        F.lit(0.0)
                    )
                )
            )
    
    # Clean Payment_Behaviour
    if 'payment_behaviour' in df_clean.columns:
        print("Cleaning payment_behaviour column")
        df_clean = df_clean.withColumn(
            'payment_behaviour',
            F.when(
                F.col('payment_behaviour').startswith('!') |
                F.col('payment_behaviour').startswith('-') |
                F.col('payment_behaviour').startswith('_'), 
                "unknown"
            ).when(
                F.col('payment_behaviour').isNull(), "unknown"
            ).otherwise(F.col('payment_behaviour'))
        )
    
    # Convert Credit_History_Age from "X Years and Y Months" to float
    if 'credit_history_age' in df_clean.columns:
        print("Converting credit_history_age to float")
        df_clean = df_clean.withColumn(
            'credit_history_age_float',
            # Extract years and months, convert to float years
            F.coalesce(
                F.regexp_extract(F.col('credit_history_age'), r'(\d+) Years', 1).cast('float'),
                F.lit(0.0)
            ) +
            F.coalesce(
                F.regexp_extract(F.col('credit_history_age'), r'(\d+) Months', 1).cast('float'),
                F.lit(0.0)
            ) / 12.0
        ).drop('credit_history_age').withColumnRenamed('credit_history_age_float', 'credit_history_age')
    
    # Clean Credit_Mix
    if 'credit_mix' in df_clean.columns:
        print("Cleaning credit_mix column")
        df_clean = df_clean.withColumn(
            'credit_mix',
            F.when(
                F.col('credit_mix').startswith('-') | 
                F.col('credit_mix').startswith('_') |
                F.col('credit_mix').startswith('!'), 
                "unknown"
            ).when(
                F.col('credit_mix').isNull(), "unknown"
            ).otherwise(F.col('credit_mix'))
        )
    
    # Convert integer columns - only set to 0 when blank, null, or actually 0
    integer_columns = [
        'num_credit_inquiries', 'num_of_delayed_payment', 'delay_from_due_date',
        'num_credit_card', 'num_bank_accounts'
    ]
    
    print("Converting integer columns")
    for col_name in integer_columns:
        if col_name in df_clean.columns:
            # First remove all non-digit characters
            df_clean = df_clean.withColumn(
                col_name,
                F.when(
                    F.col(col_name).isNotNull() & (F.col(col_name) != ''),
                    F.regexp_replace(F.col(col_name), r'[^\d]', '')
                ).otherwise(F.col(col_name))
            )
            
            # Convert to integer - only set to 0 when blank, null, or actually 0
            df_clean = df_clean.withColumn(
                col_name,
                F.when(
                    (F.col(col_name).isNull()) | (F.col(col_name) == '') | (F.col(col_name) == '0'),
                    F.lit(0)
                ).when(
                    (F.col(col_name) != '') & (F.col(col_name).rlike(r'^\d+$')),
                    F.col(col_name).cast("int")
                ).otherwise(
                    # For values that have numbers but invalid format, try to extract first valid integer
                    F.coalesce(
                        F.regexp_extract(F.col(col_name), r'(\d+)', 0).cast("int"),
                        F.lit(0)
                    )
                )
            )
    
    # Process Type_of_Loan and create num_of_loan
    if 'type_of_loan' in df_clean.columns:
        print("Processing type_of_loan column")
        
        # Remove existing num_of_loan if exists
        if 'num_of_loan' in df_clean.columns:
            df_clean = df_clean.drop('num_of_loan')
        
        # Clean and count loans
        df_clean = df_clean.withColumn(
            'type_of_loan_array',
            F.when(
                F.col('type_of_loan').isNotNull() & (F.col('type_of_loan') != ''),
                F.expr("""
                    array_distinct(
                        filter(
                            transform(
                                split(
                                    regexp_replace(
                                        regexp_replace(
                                            regexp_replace(type_of_loan, ' and ', ', '),
                                            ',+', ','
                                        ),
                                        '^,|,$', ''
                                    ),
                                    ','
                                ),
                                x -> trim(x)
                            ),
                            x -> x != ''
                        )
                    )
                """)
            ).otherwise(F.array())
        )
        
        # Create num_of_loan and final type_of_loan
        df_clean = df_clean.withColumn(
            'num_of_loan',
            F.size(F.col('type_of_loan_array')).cast("int")
        ).withColumn(
            'type_of_loan',
            F.when(
                F.size(F.col('type_of_loan_array')) > 0,
                F.concat_ws(', ', F.col('type_of_loan_array'))
            ).otherwise('')
        ).drop('type_of_loan_array')
    
    return df_clean