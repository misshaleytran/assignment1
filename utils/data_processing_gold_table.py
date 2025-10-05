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

from pyspark.sql.functions import col, countDistinct, row_number, ceil, datediff, months_between, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
from pyspark.sql.window import Window


def process_labels_gold_table(snapshot_date_str, silver_directories, gold_label_store_directory, spark, dpd, mob):
    """
    Process gold table for labels from lms data
    """
    silver_lms_directory = silver_directories.get("lms")  
    if not silver_lms_directory:
        print(f"Error: lms directory not found for {snapshot_date_str}")  
        return None

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver table
    partition_name = "silver_lms_" + snapshot_date_str.replace('-','_')+ '.parquet'
    filepath = silver_lms_directory + partition_name

    # Check if silver file exists
    if not os.path.exists(filepath):
        print(f"Error: Silver file not found: {filepath}")
        return None

    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)
    print(f"After filtering MOB={mob}, row count:", df.count())

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "customer_id", "label", "label_def", "snapshot_date")

    # save gold table
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_features_gold_table(snapshot_date_str, silver_directories, gold_feature_store_directory, spark, dpd, mob):
    """
    Process gold table for features from all data sources - ONLY CUSTOMERS IN ALL 4 SOURCES
    """
    print(f"Processing gold feature table for {snapshot_date_str}")

    # Process features from each data source
    features_dfs = []

    # 1. Clickstream features
    clickstream_features = process_clickstream_features(snapshot_date_str, silver_directories, spark)
    if clickstream_features is not None:
        features_dfs.append(clickstream_features)
        print(f"Clickstream features: {clickstream_features.count()} rows")

    # 2. Attributes features  
    attributes_features = process_attributes_features(snapshot_date_str, silver_directories, spark)
    if attributes_features is not None:
        features_dfs.append(attributes_features)
        print(f"Attributes features: {attributes_features.count()} rows")

    # 3. Financials features
    financials_features = process_financials_features(snapshot_date_str, silver_directories, spark)
    if financials_features is not None:
        features_dfs.append(financials_features)
        print(f"Financials features: {financials_features.count()} rows")

    # 4. LMS features
    lms_features = process_lms_features(snapshot_date_str, silver_directories, spark, mob)
    if lms_features is not None:
        features_dfs.append(lms_features)
        print(f"LMS features: {lms_features.count()} rows")

    if not features_dfs:
        print("Error: No features data found")
        return None

    print(f"Starting merge with {len(features_dfs)} feature sources")

    # Merge all features - ONLY KEEP CUSTOMERS PRESENT IN ALL SOURCES
    df_features = merge_features_inner_join(features_dfs, snapshot_date_str, spark)
    
    if df_features is None:
        print("Error: Failed to merge features")
        return None

    # FINAL VALIDATION
    total_rows = df_features.count()
    unique_customers = df_features.select("customer_id").distinct().count()
    print("FINAL VALIDATION:")
    print(f"   Total rows: {total_rows}")
    print(f"   Unique customers: {unique_customers}")
    
    if total_rows != unique_customers:
        print(f"CRITICAL: Row count ({total_rows}) doesn't match unique customers ({unique_customers})!")
        print("Dropping duplicates as safety measure...")
        df_features = df_features.dropDuplicates(["customer_id"])
        print(f"After deduplication: {df_features.count()} rows")
    else:
        print("SUCCESS: Row count matches unique customers!")

    # Save feature store
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_features.write.mode("overwrite").parquet(filepath)
    print(f'saved to: {filepath}')

    return df_features


def merge_features_inner_join(features_dfs, snapshot_date_str, spark):
    """
    Merge all feature DataFrames using INNER JOIN to only keep customers present in ALL sources
    """
    print("Merging all features with INNER JOIN (only customers in all sources)...")
    
    if not features_dfs:
        return None
    
    # Debug: Show what we're merging
    print(f"   Merging {len(features_dfs)} DataFrames:")
    for i, df in enumerate(features_dfs):
        unique_customers = df.select("customer_id").distinct().count()
        print(f"     DataFrame {i}: {df.count()} rows, {unique_customers} unique customers")
    
    # Start with the first DataFrame that has snapshot_date (clickstream)
    # Find clickstream DataFrame to start with (it has snapshot_date)
    clickstream_df = None
    other_dfs = []
    
    for df in features_dfs:
        if "snapshot_date" in df.columns:
            clickstream_df = df
        else:
            other_dfs.append(df)
    
    if clickstream_df is None:
        # If no clickstream, start with first DataFrame
        merged_df = features_dfs[0]
    else:
        merged_df = clickstream_df
        other_dfs = [df for df in features_dfs if df != clickstream_df]
    
    print(f"   Starting with DataFrame containing snapshot_date: {merged_df.count()} rows, {merged_df.select('customer_id').distinct().count()} unique customers")
    
    # Sequentially INNER JOIN all other DataFrames
    for i, df in enumerate(other_dfs, 1):
        print(f"   INNER JOIN with DataFrame {i}...")
        
        join_cols = ["customer_id"]
        common_cols = [col for col in join_cols if col in merged_df.columns and col in df.columns]
        
        if common_cols:
            # Remove duplicate columns from right side before join
            right_cols = [col for col in df.columns if col not in merged_df.columns or col in common_cols]
            df_subset = df.select(*right_cols)
            
            before_rows = merged_df.count()
            before_unique = merged_df.select("customer_id").distinct().count()
            right_rows = df_subset.count()
            right_unique = df_subset.select("customer_id").distinct().count()
            
            print(f"     Before join: {before_rows} rows, {before_unique} unique customers")
            print(f"     Right side: {right_rows} rows, {right_unique} unique customers")
            
            # Use INNER JOIN to only keep customers present in both DataFrames
            merged_df = merged_df.join(df_subset, on=common_cols, how="inner")
            
            after_rows = merged_df.count()
            after_unique = merged_df.select("customer_id").distinct().count()
            
            print(f"     After INNER JOIN {i}: {after_rows} rows, {after_unique} unique customers")
            
            if after_rows > after_unique:
                print(f"     WARNING: Duplicates detected! Rows: {after_rows}, Unique: {after_unique}")
            elif after_rows < before_rows:
                print(f"     INFO: Lost {before_rows - after_rows} customers not present in both sources")
                
        else:
            print(f"   Warning: No common columns for join with DataFrame {i}")
            return None
    
    final_rows = merged_df.count()
    final_unique = merged_df.select("customer_id").distinct().count()
    print(f"   Final merged: {final_rows} rows, {final_unique} unique customers, {len(merged_df.columns)} columns")
    
    # Verify we only have customers present in all sources
    if len(features_dfs) > 1 and final_rows == 0:
        print("   ERROR: No common customers found across all data sources!")
        return None
    
    return merged_df


def process_clickstream_features(snapshot_date_str, silver_directories, spark):
    """
    Process clickstream features with date filtering - EXACTLY as specified in requirements
    Group by customer_id AND snapshot_date to preserve temporal information
    """
    print("Processing clickstream features...")
    
    clickstream_df = load_silver_table("clickstream", snapshot_date_str, silver_directories, spark)
    if clickstream_df is None:
        return None
    
    # Filter by snapshot_date to get only relevant records
    clickstream_df = clickstream_df.filter(col("snapshot_date") == snapshot_date_str)
    print(f"   After date filtering: {clickstream_df.count()} rows")
    
    # Check if we have any data after filtering
    if clickstream_df.count() == 0:
        print("   Warning: No clickstream data found after date filtering")
        return None
    
    # Get all feature columns (fe_1 to fe_20)
    feature_cols = [col for col in clickstream_df.columns if col.startswith('fe_')]
    print(f"   Found {len(feature_cols)} clickstream features")
    
    if not feature_cols:
        print("   No clickstream feature columns found")
        return None
    
    # Calculate clickstream_total_mean as specified: average and sum of clickstream features per customer_id and snapshot_date
    # First calculate sum of all features
    sum_expr = None
    for i, feature_col in enumerate(feature_cols):
        if i == 0:
            sum_expr = col(feature_col)
        else:
            sum_expr = sum_expr + col(feature_col)
    
    # Calculate total mean: sum of all features divided by number of features (20)
    clickstream_with_total = clickstream_df.withColumn(
        "clickstream_total_mean", 
        sum_expr / len(feature_cols)
    )
    
    # Group by customer_id AND snapshot_date to preserve temporal information
    df_agg = clickstream_with_total.groupBy("customer_id", "snapshot_date").agg(
        F.avg("clickstream_total_mean").alias("clickstream_total_mean")
    )
    
    print(f"   Aggregated clickstream features for {df_agg.count()} customer-date combinations")
    return df_agg


def process_attributes_features(snapshot_date_str, silver_directories, spark):
    """
    Process attributes features for overwrite table - EXACTLY as specified in requirements
    """
    print("Processing attributes features...")
    
    attributes_df = load_silver_table("attributes", snapshot_date_str, silver_directories, spark)
    if attributes_df is None:
        return None
    
    # Since Attributes is an OVERWRITE table, get the latest record per customer
    window_spec = Window.partitionBy("customer_id").orderBy(col("snapshot_date").desc())
    attributes_df_ranked = attributes_df.withColumn("row_num", row_number().over(window_spec))
    attributes_df_latest = attributes_df_ranked.filter(col("row_num") == 1).drop("row_num")
    
    # Create age groups exactly as specified
    df_with_age_groups = attributes_df_latest.withColumn(
        "age_group",
        F.when((col("age") >= 14) & (col("age") <= 24), "14-24")
        .when((col("age") >= 25) & (col("age") <= 34), "25-34") 
        .when((col("age") >= 35) & (col("age") <= 44), "35-44")
        .when((col("age") >= 45) & (col("age") <= 56), "45-56")
        .otherwise("other")
    )
    
    # Keep occupation as-is for now (will apply OHE later in modeling)
    # Group by customer_id to ensure one record per customer
    df_features = df_with_age_groups.groupBy("customer_id").agg(
        F.first("age_group").alias("age_group"),
        F.first("occupation").alias("occupation")
    )
    
    print(f"   Processed attributes features for {df_features.count()} customers")
    return df_features


def process_financials_features(snapshot_date_str, silver_directories, spark):
    """
    Process financials features for overwrite table - EXACTLY as specified in requirements
    """
    print("Processing financials features...")
    
    financials_df = load_silver_table("financials", snapshot_date_str, silver_directories, spark)
    if financials_df is None:
        return None
    
    # Since Financials is an OVERWRITE table, get the latest record per customer
    window_spec = Window.partitionBy("customer_id").orderBy(col("snapshot_date").desc())
    financials_df_ranked = financials_df.withColumn("row_num", row_number().over(window_spec))
    financials_df_latest = financials_df_ranked.filter(col("row_num") == 1).drop("row_num")
    
    # Print available columns for debugging
    print(f"   Available columns in financials: {financials_df_latest.columns}")
    
    # Calculate debt_to_income exactly as specified: outstanding_debt / annual_income
    df_with_ratios = financials_df_latest.withColumn(
        "debt_to_income",
        F.when(
            (col("annual_income").isNotNull()) & (col("annual_income") > 0),
            col("outstanding_debt") / col("annual_income")
        ).otherwise(None)
    )
    
    # Calculate emi_to_salary exactly as specified: total_emi_per_month / monthly_inhand_salary
    df_with_ratios = df_with_ratios.withColumn(
        "emi_to_salary", 
        F.when(
            (col("monthly_inhand_salary").isNotNull()) & (col("monthly_inhand_salary") > 0),
            col("total_emi_per_month") / col("monthly_inhand_salary")
        ).otherwise(None)
    )
    
    # Calculate balance_to_debt exactly as specified: Monthly_Balance / Outstanding_Debt
    df_with_ratios = df_with_ratios.withColumn(
        "balance_to_debt",
        F.when(
            (col("outstanding_debt").isNotNull()) & (col("outstanding_debt") > 0),
            col("monthly_balance") / col("outstanding_debt")
        ).otherwise(None)
    )
    
    # Use existing credit_history_age column (already available in the data)
    # The column already exists, so we don't need to calculate it
    df_with_ratios = df_with_ratios.withColumnRenamed("credit_history_age", "credit_history_age")
    
    # Group by customer_id to ensure one record per customer
    df_features = df_with_ratios.groupBy("customer_id").agg(
        F.first("debt_to_income").alias("debt_to_income"),
        F.first("emi_to_salary").alias("emi_to_salary"),
        F.first("balance_to_debt").alias("balance_to_debt"),
        F.first("credit_history_age").alias("credit_history_age")
    )
    
    print(f"   Processed financials features for {df_features.count()} customers")
    return df_features


def process_lms_features(snapshot_date_str, silver_directories, spark, mob):
    """
    Process lms features with MOB filtering - EXACTLY as specified in requirements
    """
    print("Processing lms features...")
    
    lms_df = load_silver_table("lms", snapshot_date_str, silver_directories, spark)
    if lms_df is None:
        return None
    
    # Filter by MOB to match label processing
    lms_df = lms_df.filter(col("mob") == mob)
    print(f"   After MOB filtering: {lms_df.count()} rows")
    
    # Check if we have any data after MOB filtering
    if lms_df.count() == 0:
        print(f"   Warning: No LMS data found after MOB={mob} filtering")
        return None
    
    # Calculate installments_missed exactly as specified: ceil(overdue_amt / due_amt)
    lms_df = lms_df.withColumn(
        "installments_missed",
        F.when(
            (col("due_amt").isNotNull()) & (col("due_amt") > 0),
            ceil(col("overdue_amt") / col("due_amt"))
        ).otherwise(0)
    )
    
    # MOB is already calculated as specified: months_between(snapshot_date, loan_start_date)
    # We'll keep the existing mob field
    
    # Calculate first_missed_date using installment count logic
    # Assuming we have loan_start_date and installment frequency to calculate first_missed_date
    lms_df = lms_df.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.date_add(col("loan_start_date"), col("installment_num") * 30)  # Approximate 30 days per installment
        ).otherwise(None)
    )
    
    # Calculate DPD exactly as specified: datediff(snapshot_date, first_missed_date) when overdue
    lms_df = lms_df.withColumn(
        "dpd",
        F.when(
            col("first_missed_date").isNotNull(),
            datediff(col("snapshot_date"), col("first_missed_date"))
        ).otherwise(0)
    )
    
    # Group by customer_id to get one record per customer with all required features
    lms_features = lms_df.groupBy("customer_id").agg(
        F.avg("mob").alias("mob"),
        F.avg("installments_missed").alias("installments_missed"),
        F.first("first_missed_date").alias("first_missed_date"),
        F.avg("dpd").alias("dpd")
    )
    
    print(f"   Processed lms features for {lms_features.count()} customers")
    return lms_features


def load_silver_table(table_type, snapshot_date_str, silver_directories, spark):
    """
    Load a specific silver table for the given date
    """
    directory = silver_directories.get(table_type)
    if not directory:
        print(f"Directory not found for table type: {table_type}")
        return None
    
    # Map table types to file prefixes
    prefix_map = {
        "lms": "silver_lms",
        "clickstream": "silver_clickstream", 
        "attributes": "silver_attributes",
        "financials": "silver_financials"
    }
    
    prefix = prefix_map.get(table_type)
    if not prefix:
        print(f"Prefix not found for table type: {table_type}")
        return None
    
    partition_name = prefix + "_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = directory + partition_name
    
    if not os.path.exists(filepath):
        print(f"Silver file not found: {filepath}")
        return None
    
    try:
        df = spark.read.parquet(filepath)
        print(f"   Loaded {table_type} table with {df.count()} rows")
        return df
    except Exception as e:
        print(f"Error loading {table_type} table: {str(e)}")
        return None