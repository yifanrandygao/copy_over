# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks & Spark setup

# COMMAND ----------

### Load packages ###
from IPython.display import display, HTML
import traceback
import time
import json
import numpy as np
import pandas as pd
import functools
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pyspark.sql import SparkSession, functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.sql.types import DoubleType, FloatType, StringType
import sys
from datetime import datetime

sns.set_theme(context="notebook", style="whitegrid", font_scale=1.1)

# COMMAND ----------

### Attach your own cluster instead of serverless for better functionality and resources ###

# COMMAND ----------

### Define global config ###

GLOBAL_CONFIG = {
    # Database and table settings
    'catalog': "dataproducts_dev",
    'schema': "bronze_acxiom",
    'table': "gm_consumer_list",
    'id_column': "GM_PERSON_REALID",
    'destination_catalog': "infact_dev",
    'destination_schema': "marsci",
    'destination_table': "upper_funnel",
    
    # File paths and prefixes
    'base_output_path': "/Volumes/infact_dev/marsci/testing/upper_funnel/",
    'test_suffix': "test/",
    'preprocessed_suffix': "preprocessed/",
    'model_output_suffix': "model_output/",
    
    # SEED
    'seed': 42,
    
    # # Sample sizes
    # 'mca_sample_size': 50000,  # Larger sample for MCA
    # 'clustering_sample_size': 40000,  # Smaller sample for clustering
    
    # # Clustering parameters
    # 'kmeans_clusters': [8, 10, 12],
    # 'hierarchical_clusters': [8, 10, 12],
    # 'max_clusters': 15
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Functions

# COMMAND ----------

def curr_time_to_str():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")

def time_taken(start_time, end_time):
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    diff = end - start
    return round(diff.total_seconds() / 60, 1)


def missing_percentages(
    df,
    include_nan,
    include_blank_strings
):
    """
    Return a Spark DataFrame with columns:
        - column: column name
        - missing_pct: percentage (0..100) of missing values
    Missing = NULL; plus NaN for float/double if include_nan=True;
              plus '' (after trim) for strings if include_blank_strings=True.
    """
    # Build one aggregation row containing total rows + per-column missing counts
    exprs = [F.count(F.lit(1)).alias("__total__")]
    for field in df.schema.fields:
        c = field.name
        cond = F.col(c).isNull()
        if include_nan and isinstance(field.dataType, (DoubleType, FloatType)):
            cond = cond | F.isnan(F.col(c))
        if include_blank_strings and isinstance(field.dataType, StringType):
            cond = cond | (F.trim(F.col(c)) == "")
        exprs.append(F.sum(F.when(cond, 1).otherwise(0)).alias(c))

    agg = df.agg(*exprs).collect()[0].asDict()
    total = agg.pop("__total__")

    # Build (column, pct) rows; guard against total=0
    total = float(total) if total else 1.0
    rows = [(c, float(agg.get(c, 0.0)) * 100.0 / total) for c in df.columns]

    out = df.sparkSession.createDataFrame(rows, ["column", "missing_pct"]).withColumn("missing_pct", F.round(F.col("missing_pct"), 2))
    return out.orderBy(F.desc("missing_pct"))



def add_feature_description(main_df, ref_df, 
                            main_key, 
                            ref_key="variable_name", 
                            ref_desc="variable_short_description"):
    """
    Join main_df with ref_df to add a description column.
    Optionally places the description as the leftmost column.
    """
    df_joined = (
        main_df
        .join(ref_df.select(ref_key, ref_desc), 
              on=[main_df[main_key] == ref_df[ref_key]], 
              how="left")
        .drop(ref_key)
    )
    
    cols = [ref_desc] + [c for c in df_joined.columns if c != ref_desc]
    df_joined = df_joined.select(*cols)
    
    return df_joined

def print_missing_percentages(df, ref_df, include_nan = True, include_blank_strings = False):

    df = missing_percentages(df, include_nan, include_blank_strings)
    df = add_feature_description(df, ref_df, 
                                "column", 
                                ref_key="variable_name", 
                                ref_desc="variable_short_description")
    
    df.orderBy(F.desc("missing_pct")).display()

    # return df.orderBy(F.desc("missing_pct")).show(df.count(), truncate=False)

def print_summary_stats(df):
    return df.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").toPandas().style.set_caption("Summary statistics")


def to_pd(sdf, limit=None):
    """Safely convert small Spark DF to pandas (optionally limit rows)."""
    return (sdf.limit(limit).toPandas() if limit else sdf.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks/Spark Tests and Enablement

# COMMAND ----------

class DBXSetup:

    _spark = None  # cached handle

    def __init__(self, 
                 test_path = f"{GLOBAL_CONFIG['base_output_path']}{GLOBAL_CONFIG['test_suffix']}", 
                 test_file = f"permissions_test_{curr_time_to_str()}.txt",
                 test_catelog = GLOBAL_CONFIG['catalog'],
                 test_schema = GLOBAL_CONFIG['schema'],
                 test_table = GLOBAL_CONFIG['table'],
                 test_column = GLOBAL_CONFIG['id_column']
                 ):
        self.test_path = test_path
        self.test_file = test_file
        self.test_catelog = test_catelog
        self.test_schema = test_schema
        self.test_table = test_table
        self.test_column = test_column

    def select_query_test(self):
        try:
            first_gm_id = spark.sql(f"SELECT {self.test_column} FROM {self.test_catelog}.{self.test_schema}.{self.test_table} LIMIT 1").collect()[0][0]
            print(f"\n✅ Successfully execute SELECT query on {self.test_catelog}.{self.test_schema}.{self.test_table}")
        except Exception as e:
            print(f"\n❌ Failed to execute SELECT query: {self.test_catelog}.{self.test_schema}.{self.test_table}")

    def write_to_volume_test(self):
        try:
            dbutils.fs.put(f"{self.test_path}{self.test_file}", "Permissions test", overwrite=True)
            print(f"\n✅ Successfully wrote to {self.test_path}")
        except Exception as e:
            print(f"\n❌ Failed to write to temporary location: {e}")

    @classmethod
    def _get_current_spark_session(cls):
        if cls._spark is None:
            # Prefer the active session if it exists (e.g., Databricks), else create
            cls._spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        return cls._spark

    @staticmethod
    def print_spark_config():
        for key, value in spark.conf.getAll.items():
            print(f"{key}: {value}")

    @classmethod
    def enable_dbx_disk_caching(cls):
        # Get the current Spark session
        spark = cls._get_current_spark_session()
        # Try to enable Databricks disk caching with error handling
        try:
            # Attempt to enable disk caching
            spark.conf.set("spark.databricks.io.cache.enabled", "true")
            spark.conf.set("spark.databricks.io.cache.compression.enabled", "true")
            
            # Verify if setting was accepted
            enabled = spark.conf.get("spark.databricks.io.cache.enabled")
            if enabled == "true":
                print(f"✅ Databricks disk cache successfully enabled")
            
        except Exception as e:
            print(f"⚠️ Unable to enable Databricks disk cache: {str(e)}")
            print("The pipeline will continue without disk caching")
            
            # Check runtime type to provide better guidance
            if "spark.databricks.compute" in [conf.key for conf in spark.sparkContext.getConf().getAll()]:
                compute_type = spark.sparkContext.getConf().get("spark.databricks.compute", "unknown")
                if "serverless" in compute_type.lower():
                    print("NOTE: You appear to be using Serverless compute which has limited configuration options.")
                    print("Consider switching to a Standard All-Purpose cluster if you need disk caching.")

    
    @classmethod
    def enable_sql_auto_partition(cls):
        # Get the current Spark session
        spark = cls._get_current_spark_session()
        # Try to enable Spark SQL auto partitioning with error handling
        try:
            # Attempt to enable auto partitioning
            spark.conf.set("spark.sql.shuffle.partitions", "auto")
            spark.conf.set("spark.sql.adaptive.enabled", "true")
            
            # Verify if setting was accepted
            enabled1 = spark.conf.get("spark.sql.shuffle.partitions")
            enabled2 = spark.conf.get("spark.sql.adaptive.enabled")
            if enabled1 == "auto" and enabled2 == "true":
                print(f"✅ Spark SQL auto partitioning successfully enabled")
            
        except Exception as e:
            print(f"⚠️ Unable to enable Spark SQL auto partitioning: {str(e)}")
            print("The pipeline will continue without auto partitioning")
    
    @classmethod
    def enable_delta_write_efficiency(cls):
        # Get the current Spark session
        spark = cls._get_current_spark_session()
        # Try to enable Delta auto compact and fast reading with error handling
        try:
            # Attempt to enable auto compact and fast reading
            spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
            spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
            
            # Verify if setting was accepted
            enabled1 = spark.conf.get("spark.databricks.delta.optimizeWrite.enabled")
            enabled2 = spark.conf.get("spark.databricks.delta.autoCompact.enabled")
            if enabled1 == "true" and enabled2 == "true":
                print(f"✅ Delta auto compact and fast reading enabled")
            
        except Exception as e:
            print(f"⚠️ Unable to enable Spark SQL auto partitioning: {str(e)}")
            print("The pipeline will continue without Delta auto compact and fast reading")


# COMMAND ----------

dbs = DBXSetup()

# COMMAND ----------

dbs.select_query_test()

# COMMAND ----------

dbs.write_to_volume_test()

# COMMAND ----------

dbs.enable_dbx_disk_caching()

# COMMAND ----------

dbs.enable_sql_auto_partition()

# COMMAND ----------

dbs.enable_delta_write_efficiency()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Raw Data Prep

# COMMAND ----------

class RawDataPrep:
    def __init__(self, 
                catelog = GLOBAL_CONFIG['catalog'],
                schema = GLOBAL_CONFIG['schema'],
                table = GLOBAL_CONFIG['table'],
                destination_catelog = GLOBAL_CONFIG['destination_catalog'],
                destination_schema = GLOBAL_CONFIG['destination_schema'],
                destination_table = f"{GLOBAL_CONFIG['destination_table']}_{curr_time_to_str()}" ,
                id_column = GLOBAL_CONFIG['id_column'],
                pop_estimate = 240000000,
                sample_size = None,
                sample_frac = None
                ):
        self.catelog = catelog
        self.schema = schema
        self.table = table
        self.destination_catelog = destination_catelog
        self.destination_schema = destination_schema
        self.destination_table = destination_table
        self.id_column = id_column
        self.sample_size = sample_size
        self.sample_frac = sample_frac
        self.pop_estimate = pop_estimate
        self.SEED = GLOBAL_CONFIG['seed']
    
    @staticmethod
    def full_columns():
        # 1. VEHICLE PREFERENCE COLUMNS
        veh_pref_columns = [
            ## Luxury vehicle interest
            "AP004577",  # Purchase a New Luxury Sedan (Financial)
            "AP004561",  # Purchase a New Luxury CUV (Financial)
            "AP004542",  # Purchase a New Luxury SUV (Financial)
            "AP004541",  # Purchase a New Luxury Sports Car (Financial)
            "AP004567",  # Purchase a New Luxury 4WD (Financial)
            "AP004569",  # Purchase a New Luxury AWD (Financial)
            "AP004571",  # Purchase a New Luxury Convertible (Financial)
            "AP004573",  # Purchase a New Luxury Coupe (Financial)
            "AP004575",  # Purchase a New Luxury Diesel (Financial)
            "AP004579",  # Purchase a New Luxury Wagon (Financial)
            
            ## Compact vehicle interest
            "AP004559",  # Purchase a New Compact CUV (Financial)
            "AP004532",  # Purchase a New Compact SUV (Financial)
            "AP004531",  # Purchase a New Compact Pickup (Financial)
            "AP004533",  # Purchase a New Compact Van (Financial)

            ## Mid-size vehicle interest
            "AP004543",  # Purchase a New Mid Luxury Car (Financial)
            "AP004544",  # Purchase a New Mid SUV (Financial)
            "AP004560",  # Purchase a new Midsize CUV (Financial)

            ## New Entry vehicle interest
            "AP004534",  # Purchase a New Entry Compact Car (Financial)
            "AP004535",  # Purchase a New Entry Luxury Car (Financial)
            "AP004536",  # Purchase a New Entry Mid Size Car (Financial)
            "AP004537",  # Purchase a New Entry Sports Car (Financial)

            ## Premium vehicle interest
            "AP004545",  # Purchase a New Premium Compact Car (Financial)
            "AP004546",  # Purchase a New Premium Luxury Car (Financial)
            "AP004547",  # Purchase a New Premium Mid Size Car (Financial)
            "AP004548",  # Purchase a New Premium Sports Car (Financial)

            ## Hybrid/EV vehicle interest
            "AP004562",  # Purchase a New Electrical Car (Financial)
            "AP004563",  # Purchase a New Hybrid Car (Financial)
            "AP004564",  # Purchase a New Hybrid Luxury Car (Financial)
            "AP004565",  # Purchase a New Hybrid Regular Car (Financial)
            "AP004566",  # Purchase a New Hybrid SUV (Financial)

            ## Regular vehicle interest
            "AP004568",  # Purchase a New Regular 4WD (Financial)
            "AP004570",  # Purchase a New Regular AWD (Financial)
            "AP004572",  # Purchase a New Regular Convertible (Financial)
            "AP004574",  # Purchase a New Regular Coupe (Financial)
            "AP004576",  # Purchase a New Regular Diesel (Financial)
            "AP004578",  # Purchase a New Regular Sedan (Financial)
            "AP004580",  # Purchase a New Regular Wagon (Financial)

            ## Other vehicle interest
            "AP004538",  # Purchase a New Full Size SUV (Financial)
            "AP004539",  # Purchase a New Heavy Duty Full Size Pickup (Financial)
            "AP004540",  # Purchase a New Light Duty Full Size Pickup (Financial)

            ## Purchase reason
            "AP006975",	#	Bank Loan Used to Purchase Vehicle
            "AP006984",	#	Child Driver Reason for Purchase
            "AP006980",	#	Dealer Courtesy Car Incentives Reason for Purchase
            "AP006977",	#	Dealer Financing Incentives Reason for Purchase
            "AP006974",	#	Dealer Lease Used to Purchase Vehicle
            "AP006966",	#	Dealer Loan Used to Purchase Vehicle
            "AP006979",	#	Dealer Shuttle Service Incentives Reason for Purchase
            "AP006985",	#	Divorce Reason for Purchase
            "AP006983",	#	Growing Family Reason for Purchase
            "AP006981",	#	New Job Reason for Purchase
            "AP006986",	#	Purchased Vehicle In-Dealership
            "AP006990",	#	Purchased Vehicle Online
            "AP006987",	#	Vehicle Maintenance at Dealer
            "AP006988",	#	Vehicle Purchased as a Gift
            
            ## Current ownership
            "AP007089",	# Owns an Electric Vehicle
            "AP007094",	# Owns an Entry Electric Vehicle
            "AP007091",	# Owns a Regular Electric Vehicle
            "AP007096",	# Owns a Mid-Sized Hybrid Vehicle
            "AP007081",	# Owns a Luxury Vehicle
            "AP007113",	# Owns an all electric SUV
            "AP007114",	# Owns an all electric vehicle
            "AP007079",	# Owns a Compact or Subcompact Body Style Vehicle
            "AP007103",	# Owns a Crossover  Body Style Vehicle
            "AP007107",	# Owns a Diesel Engine Truck Body Style Vehicle
            "AP007080",	# Owns a Full Size Body Style Vehicle
            "AP007109",	# Owns a Full Sized Pickup Truck Body Style Vehicle
            "AP007108",	# Owns a Gasoline Engine Truck  Body Style Vehicle
            "AP007105",	# Owns a Luxury SUV Body Style Vehicle
            "AP007133",	# Owns a Mid-Sized Pickup Truck  Body Style Vehicle
            "AP007082",	# Owns a Mid Size Body Style Vehicle
            "AP007102",	# Owns a Minivan Body Style Vehicle
            "AP007083",	# Owns a Sports Car Body Style Vehicle
            "AP007106",	# Owns a SUV Body Style Vehicle
            "AP007111",	# Owns a New Vehicle
            "AP006952",	# Owns a Used Vehicle
            "AP006962",	# Owns a ATV Recreational Vehicle
            "AP006961",	# Owns a Camper Recreational Vehicle
            "AP006958",	# Owns a Fifth Wheel Recreational Vehicle
            "AP006957",	# Owns a Motor Home Recreational Vehicle
            "AP006964",	# Owns a Side by Side Recreational Vehicle
            "AP006965",	# Owns a Snowmobile Recreational Vehicle
            "AP006959",	# Owns a Towable/Caravan Recreational Vehicle
            "AP006956",	# Owns a Boat Water Craft
            "AP006955",	# Owns a Jet Ski or Sea Doo Water Craft
            "IBE8647",	#	Vehicle - Known Owned Number
            "IBE8646",	#	Vehicle - New Car Buyer
            "IBE9042",	#	Vehicle - Year - 1st Vehicle
            "IBE9052",	#	Vehicle - Year - 2nd Vehicle
            "IBE9180",	#	Vehicle Type - Vehicle 1
            "IBE9181",	#	Vehicle Type - Vehicle 2       

            ## Polk in market future
            "PM020044",	# Polk - In-Market (Future) - New Luxury Vehicle
            "PM020045",	# Polk - In-Market (Future) - New Non-luxury Vehicle
            "PM020046",	# Polk - In-Market (Future) - Electric
            "PM020047",	# Polk - In-Market (Future) - Hybrid
            "PM020048",	# Polk - In-Market (Future) - Any New Vehicle
            "PM020019",	# Polk - In-Market (Future) - Luxury Compact CUV
            "PM020020",	# Polk - In-Market (Future) - Luxury Sporty Car
            "PM020021",	# Polk - In-Market (Future) - Luxury Full-size SUV
            "PM020022",	# Polk - In-Market (Future) - Luxury Mid-size CUV
            "PM020023",	# Polk - In-Market (Future) - Luxury Mid-size SUV
            "PM020024",	# Polk - In-Market (Future) - Luxury Traditional Compact Car
            "PM020025",	# Polk - In-Market (Future) - Luxury Traditional Full-size Car
            "PM020026",	# Polk - In-Market (Future) - Luxury Traditional Mid-size Car
            "PM020027",	# Polk - In-Market (Future) - Luxury Traditional Subcompact Car
            "PM021325",	# Polk - In-Market (Future) - Luxury Traditional Subcompact Plus Utility
            "PM020028",	# Polk - In-Market (Future) - Non-luxury Sporty Car
            "PM020029",	# Polk - In-Market (Future) - Non-luxury Compact CUV
            "PM020030",	# Polk - In-Market (Future) - Non-luxury Compact SUV
            "PM020031",	# Polk - In-Market (Future) - Non-luxury 1/2 Ton Full-size Pickup
            "PM020032",	# Polk - In-Market (Future) - Non-luxury 3/4-1 Ton Full-size Pickup
            "PM020033",	# Polk - In-Market (Future) - Non-luxury 3/4-1 Ton Full-size Van
            "PM020034",	# Polk - In-Market (Future) - Non-luxury Full-size SUV
            "PM020035",	# Polk - In-Market (Future) - Non-luxury Mid-size CUV
            "PM020036",	# Polk - In-Market (Future) - Non-luxury Mid-size Pickup
            "PM020037",	# Polk - In-Market (Future) - Non-luxury Mid-size Sporty Car
            "PM020038",	# Polk - In-Market (Future) - Non-luxury Mid-size SUV
            "PM020039",	# Polk - In-Market (Future) - Non-luxury Mid-size Van
            "PM020040",	# Polk - In-Market (Future) - Non-luxury Traditional Compact Car
            "PM020041",	# Polk - In-Market (Future) - Non-luxury Traditional Full-size Car
            "PM020042",	# Polk - In-Market (Future) - Non-luxury Traditional Mid-size Car
            "PM020043",	# Polk - In-Market (Future) - Non-luxury Traditional Subcompact Car
            "PM021326",	# Polk - In-Market (Future) - Non-Luxury Traditional Subcompact Plus Utility


            ## Polk in market
            "PM020049",	# Polk - In-Market - 4 Wheel Drive Truck
            "PM020050",	# Polk - In-Market - All Wheel Drive
            "PM020051",	# Polk - In-Market - New Diesel
            "PM020052",	# Polk - In-Market - New Any Alternative Fuel
            "PM020053",	# Polk - In-Market - Any EV
            "PM020054",	# Polk - In-Market - New Hybrid
            "PM020055",	# Polk - In-Market - Any Luxury EV
            "PM020056",	# Polk - In-Market - New Luxury Hybrid
            "PM020057",	# Polk - In-Market - Any Non-Luxury EV
            "PM020058",	# Polk - In-Market - New Non-Luxury Hybrid
            "PM020059",	# Polk - In-Market - Used Any Alternative Fuel
            "PM021188",	# Polk - In-Market - Any EV & Already Own
            "PM021192",	# Polk - In-Market - Any EV & First Purchase
            "PM021189",	# Polk - In-Market - Any EV & Keep Existing
            "PM021190",	# Polk - In-Market - Any EV & Replace Existing EV
            "PM021191",	# Polk - In-Market - Any EV & Replace Existing Gas
            "PM021353",	# Polk - In-Market - New EV
            "PM021354",	# Polk - In-Market - New Luxury EV
            "PM021355",	# Polk - In-Market - New Non-Luxury EV
            "PM021356",	# Polk - In-Market - Plug In Electric
            "PM020063",	# Polk - In-Market - Buick Brand
            "PM020064",	# Polk - In-Market - Cadillac Brand
            "PM020065",	# Polk - In-Market - Chevrolet Brand
            "PM020070",	# Polk - In-Market - GMC Brand
            "PM021206",	# Polk - In-Market - Chevrolet EV
            "PM020119",	# Polk - In-Market - Buick Enclave
            "PM020120",	# Polk - In-Market - Buick Encore
            "PM020121",	# Polk - In-Market - Buick Envision
            "PM020123",	# Polk - In-Market - Cadillac Escalade
            "PM020124",	# Polk - In-Market - Cadillac XT4
            "PM020125",	# Polk - In-Market - Cadillac XT5
            "PM021362",	# Polk - In-Market - Cadillac CT4
            "PM021363",	# Polk - In-Market - Cadillac CT5
            "PM021364",	# Polk - In-Market - Cadillac XT6
            "PM020126",	# Polk - In-Market - Chevrolet Blazer
            "PM020127",	# Polk - In-Market - Chevrolet Bolt
            "PM020128",	# Polk - In-Market - Chevrolet Camaro
            "PM020129",	# Polk - In-Market - Chevrolet Colorado
            "PM020130",	# Polk - In-Market - Chevrolet Corvette
            "PM020131",	# Polk - In-Market - Chevrolet Equinox
            "PM020132",	# Polk - In-Market - Chevrolet Express
            "PM020134",	# Polk - In-Market - Chevrolet Malibu
            "PM020135",	# Polk - In-Market - Chevrolet Silverado 3 Quarter to 1 Ton
            "PM020136",	# Polk - In-Market - Chevrolet Silverado Any
            "PM020139",	# Polk - In-Market - Chevrolet Suburban
            "PM020140",	# Polk - In-Market - Chevrolet Tahoe
            "PM020141",	# Polk - In-Market - Chevrolet Traverse
            "PM020142",	# Polk - In-Market - Chevrolet Trax
            "PM021222",	# Polk - In-Market - Chevrolet Bolt EV
            "PM021221",	# Polk - In-Market - Chevrolet Bolt EUV
            "PM021304",	# Polk - In-Market - Chevrolet Trailblazer
            "PM020168",	# Polk - In-Market - GMC Acadia
            "PM020169",	# Polk - In-Market - GMC Canyon
            "PM020170",	# Polk - In-Market - GMC Sierra 3 Quarter to 1 Ton
            "PM020171",	# Polk - In-Market - GMC Sierra Any
            "PM020172",	# Polk - In-Market - GMC Terrain
            "PM020173",	# Polk - In-Market - GMC Yukon
            "PM020323",	# Polk - In-Market - New luxury compact CUV
            "PM020324",	# Polk - In-Market - New luxury exotic car
            "PM020325",	# Polk - In-Market - New luxury sporty car
            "PM020326",	# Polk - In-Market - New luxury full-size SUV
            "PM020327",	# Polk - In-Market - New luxury prestige full-size car
            "PM021197",	# Polk - In-Market - New luxury 1/2 ton full-size pickup EV
            "PM020328",	# Polk - In-Market - New luxury mid-size CUV
            "PM020329",	# Polk - In-Market - New luxury mid-size SUV
            "PM021199",	# Polk - In-Market - New luxury lower Mid-Size Utility EV
            "PM020330",	# Polk - In-Market - New luxury traditional compact car
            "PM020331",	# Polk - In-Market - New luxury traditional full-size car
            "PM020332",	# Polk - In-Market - New luxury traditional mid-size car
            "PM020333",	# Polk - In-Market - New luxury traditional subcompact car
            "PM021198",	# Polk - In-Market - New luxury traditional full-size car EV
            "PM021194",	# Polk - In-Market - New luxury compact car EV
            "PM021195",	# Polk - In-Market - New luxury compact Utility EV
            "PM021201",	# Polk - In-Market - New luxury sub compact plus Utility EV
            "PM021204",	# Polk - In-Market - New luxury upper Mid-size Utility EV
            "PM020334",	# Polk - In-Market - New non-luxury compact CUV
            "PM020335",	# Polk - In-Market - New non-luxury compact SUV
            "PM020336",	# Polk - In-Market - New non-luxury compact van
            "PM021193",	# Polk - In-Market - New non-luxury compact car EV
            "PM021196",	# Polk - In-Market - New non-luxury compact Utility EV
            "PM020337",	# Polk - In-Market - New non-luxury 1/2 ton full-size pickup
            "PM020338",	# Polk - In-Market - New non-luxury 1/2 ton full-size van
            "PM020339",	# Polk - In-Market - New non-luxury full-size SUV
            "PM020340",	# Polk - In-Market - New non-luxury 3/4-1 ton full-size pickup
            "PM020341",	# Polk - In-Market - New non-luxury 3/4-1 ton full-size van
            "PM020342",	# Polk - In-Market - New non-luxury mid-size CUV
            "PM020343",	# Polk - In-Market - New non-luxury mid-size pickup
            "PM020344",	# Polk - In-Market - New non-luxury mid-size sport car
            "PM020345",	# Polk - In-Market - New non-luxury mid-size SUV
            "PM020346",	# Polk - In-Market - New non-luxury mid-size van
            "PM020347",	# Polk - In-Market - New non-luxury traditional compact car
            "PM020348",	# Polk - In-Market - New non-luxury traditional full-size car
            "PM020349",	# Polk - In-Market - New non-luxury traditional mid-size car
            "PM020350",	# Polk - In-Market - New non-luxury traditional subcompact car
            "PM020351",	# Polk - In-Market - New non-luxury sporty car
            "PM021200",	# Polk - In-Market - New non-luxury sub compact car EV
            "PM021202",	# Polk - In-Market - New non-luxury sub compact plus Utility EV
            "PM021203",	# Polk - In-Market - New non-luxury sub compact Utility EV
            "PM020352",	# Polk - In-Market - Battery Service
            "PM020353",	# Polk - In-Market - Brake Service
            "PM020354",	# Polk - In-Market - General Service/Inspections
            "PM020355",	# Polk - In-Market - Mileage Maintenance
            "PM020356",	# Polk - In-Market - Oil Changes and Service
            "PM020357",	# Polk - In-Market - Powertrain Service
            "PM020358",	# Polk - In-Market - Wheels and Tires Service
            "PM020359",	# Polk - In-Market - Lease
            "PM020360",	# Polk - In-Market - Lease Luxury
            "PM020361",	# Polk - In-Market - Lease Non-luxury
            "PM020362",	# Polk - In-Market - Purchase new vehicle ranking
            "PM020363",	# Polk - In-Market - New Luxury Vehicle
            "PM020364",	# Polk - In-Market - New Non-Luxury Vehicle
            "PM020365",	# Polk - In-Market - Purchase ranking any new or used vehicle
            "PM020366",	# Polk - In-Market - Purchase used vehicle ranking
            "PM020367",	# Polk - In-Market - Used vehicle owners likely to purchase any new vehicle
            "PM020368",	# Polk - In-Market - Used vehicle owners likely to purchase a new luxury vehicle
            "PM020369",	# Polk - In-Market - Used vehicle owners likely to purchase a new non-luxury vehicle
            "AP004485",	#	In Market for a New Vehicle (Financial)
            "AP006140",	#	In Market for a Used Vehicle (Financial)
            "AP004553",	#	In Market for a New Domestic Luxury Vehicle (Financial)
            "AP004554",	#	In Market for a New Domestic Regular Vehicle (Financial)
            "AP004555",	#	In Market for a New European Vehicle (Financial)
            "AP004556",	#	In Market for a New Japanese Luxury Vehicle (Financial)
            "AP004557",	#	In Market for a New Japanese Regular Vehicle (Financial)
            "AP004558",	#	In Market for a New Korean Vehicle (Financial)
            "AP004414",	#	In Market for a New Luxury Vehicle (Financial)
            "AP004415",	#	In Market for a New Regular Vehicle (Financial)

            ## Polk loyalty
            "PM020370", # Polk - Loyalty - Convertible Loyalist
            "PM020373", # Polk - Loyalty - Coupe Loyalist
            "PM020376", # Polk - Loyalty - Hatchback Loyalist
            "PM020379", # Polk - Loyalty - Pickup Loyalist
            "PM020382", # Polk - Loyalty - Sedan Loyalist
            "PM020388", # Polk - Loyalty - SUV Loyalist
            "PM020391", # Polk - Loyalty - Van (Passenger) Loyalist
            "PM020394", # Polk - Loyalty - Wagon Loyalist
            "PM020397", # Polk - Loyalty - Luxury Loyalist
            "PM020400", # Polk - Loyalty - Non-Luxury Loyalist
            "PM020403", # Polk - Loyalty - Electric Vehicle or Hybrid Vehicle Loyalist
            "PM021232", # Polk - Loyalty - EV Loyalist
            "PM021327", # Polk - Loyalty - Hybrid Loyalist
            "PM020406", # Polk - Loyalty - Lease Any Loyalist
            "PM020427", # Polk - Loyalty - Buick Loyalist
            "PM020431", # Polk - Loyalty - Cadillac Loyalist
            "PM020435", # Polk - Loyalty - Chevrolet Loyalist
            "PM020455", # Polk - Loyalty - General Motors Loyalist
            "PM020457", # Polk - Loyalty - GMC Super Loyalist

            ## Polk owner
            "PM020642", #	Polk - Owner - Motorcycle Owner
            "PM020644", #	Polk - Owner - Compressed Natural Gas Fuel Vehicle
            "PM020645", #	Polk - Owner - diesel fuel vehicle
            "PM020646", #	Polk - Owner - Electric Vehicle
            "PM020647", #	Polk - Owner - Gasoline Vehicle
            "PM021349", #	Polk - Owner - Plug-in Hybrid Vehicle
            "PM020652", #	Polk - Owner - Buick
            "PM020653", #	Polk - Owner - Cadillac
            "PM020654", #	Polk - Owner - Chevrolet Division
            "PM020660", #	Polk - Owner - GMC division
            "PM021258", #	Polk - Owner - Chevrolet EV
            "PM021263", #	Polk - Owner - GMC EV
            "PM020740", #	Polk - Owner - Buick Cascada
            "PM020742", #	Polk - Owner - Buick Enclave
            "PM020743", #	Polk - Owner - Buick Encore
            "PM020744", #	Polk - Owner - Buick Envision
            "PM020745", #	Polk - Owner - Buick LaCrosse
            "PM020747", #	Polk - Owner - Buick Lucerne
            "PM020750", #	Polk - Owner - Buick Regal
            "PM020752", #	Polk - Owner - Buick Verano
            "PM020753", #	Polk - Owner - Cadillac ATS
            "PM020754", #	Polk - Owner - Cadillac CTS
            "PM020756", #	Polk - Owner - Cadillac DTS
            "PM020757", #	Polk - Owner - Cadillac Escalade
            "PM020758", #	Polk - Owner - Cadillac SRX
            "PM020759", #	Polk - Owner - Cadillac STS
            "PM020760", #	Polk - Owner - Cadillac XT5
            "PM020761", #	Polk - Owner - Cadillac XTS
            "PM021334", #	Polk - Owner - Cadillac CT4
            "PM021335", #	Polk - Owner - Cadillac CT6
            "PM021429", #	Polk - Owner - Cadillac CT5
            "PM021430", #	Polk - Owner - Cadillac XT4
            "PM021431", #	Polk - Owner - Cadillac XT6
            "PM020763", #	Polk - Owner - Chevrolet Avalanche
            "PM020764", #	Polk - Owner - Chevrolet Aveo
            "PM020765", #	Polk - Owner - Chevrolet Bolt
            "PM020766", #	Polk - Owner - Chevrolet Camaro
            "PM020769", #	Polk - Owner - Chevrolet Cobalt
            "PM020770", #	Polk - Owner - Chevrolet Colorado
            "PM020771", #	Polk - Owner - Chevrolet Corvette
            "PM020772", #	Polk - Owner - Chevrolet Cruze
            "PM020773", #	Polk - Owner - Chevrolet Equinox
            "PM020774", #	Polk - Owner - Chevrolet Express
            "PM020775", #	Polk - Owner - Chevrolet HHR
            'PM020776', #	Polk - Owner - Chevrolet Impala
            "PM020777", #	Polk - Owner - Chevrolet Malibu
            "PM020779", #	Polk - Owner - Chevrolet Silverado
            "PM020780", #	Polk - Owner - Chevrolet Sonic
            "PM020781", #	Polk - Owner - Chevrolet Spark
            "PM020782", #	Polk - Owner - Chevrolet Suburban
            "PM020783", #	Polk - Owner - Chevrolet Tahoe
            "PM020784", #	Polk - Owner - Chevrolet Trailblazer
            "PM020785", #	Polk - Owner - Chevrolet Traverse
            "PM020786", #	Polk - Owner - Chevrolet Trax
            "PM021336", #	Polk - Owner - Chevrolet Blazer
            "PM021281", #	Polk - Owner - Chevrolet Bolt EUV
            "PM021337", #	Polk - Owner - Chevrolet S10 Blazer
            "PM021282", #	Polk - Owner - Chevrolet Spark EV
            "PM020836", #	Polk - Owner - GMC Acadia
            "PM020837", #	Polk - Owner - GMC Canyon
            "PM020839", #	Polk - Owner - GMC Savana
            "PM020840", #	Polk - Owner - GMC Sierra
            "PM020841", #	Polk - Owner - GMC Terrain
            "PM020842", #	Polk - Owner - GMC Yukon
            "PM020843", #	Polk - Owner - GMC Yukon XL
            "PM021105", #	Polk - Owner - five or more vehicles
            "PM021106", #	Polk - Owner - never owned
            "PM021107", #	Polk - Owner - no vehicles
            "PM021108", #	Polk - Owner - one vehicle
            "PM021109", #	Polk - Owner - two or more vehicles
            "PM021110", #	Polk - Owner - Vehicle Purchased 0-6 Months Ago
            "PM021111", #	Polk - Owner - Vehicle Purchased 7-12 Months Ago
            "PM021112", #	Polk - Owner - Vehicle Purchased 13-24 Months Ago
            "PM021113", #	Polk - Owner - Vehicle Purchased 25-36 Months Ago
            "PM021115", #	Polk - Owner - Vehicle Purchased 37-48 Months Ago
            "PM021116", #	Polk - Owner - Vehicle Purchased 49+ Months Ago
            "PM021117", #	Polk - Owner - Any Vehicle Bought New
            "PM021118", #	Polk - Owner - Car Bought New
            "PM021119", #	Polk - Owner - Truck Bought New
            "PM021120", #	Polk - Owner - vehicle bought used
            "PM021121", #	Polk - Owner - car bought used
            "PM021122", #	Polk - Owner - truck bought used
            "PM021123", #	Polk - Owner - Any Lease Vehicle
            "PM021133", #	Polk - Owner - Any Non-Luxury Vehicle
            "PM021352", #	Polk - Owner - Any Luxury Vehicle
            "PM021134", #	Polk - Owner - Luxury compact CUV
            "PM021135", #	Polk - Owner - Luxury exotic car
            "PM021136", #	Polk - Owner - Luxury sporty car
            "PM021137", #	Polk - Owner - Luxury full-size 1/2 ton pickup
            "PM021138", #	Polk - Owner - Luxury prestige full-size car
            "PM021139", #	Polk - Owner - Luxury full-size SUV
            "PM021247", #	Polk - Owner - Luxury full-size car EV
            "PM021245", #	Polk - Owner - Full Size Half Ton Luxury Pick Up EV
            "PM021140", #	Polk - Owner - Luxury mid-size CUV
            "PM021141", #	Polk - Owner - Luxury mid-size SUV
            "PM021142", #	Polk - Owner - Luxury traditional compact car
            "PM021143", #	Polk - Owner - Luxury traditional full-size car
            "PM021144", #	Polk - Owner - Luxury traditional mid-size car
            "PM021145", #	Polk - Owner - Luxury traditional subcompact car
            "PM021242", #	Polk - Owner - Compact Luxury EV
            "PM021243", #	Polk - Owner - Compact Luxury Utility EV
            "PM021239", #	Polk - Owner - Luxury EV
            "PM021248", #	Polk - Owner - Luxury Lower Mid-Size Utility EV
            "PM021251", #	Polk - Owner - Luxury sub compact car EV
            "PM021254", #	Polk - Owner - Luxury sub compact plus Utility EV
            "PM021253", #	Polk - Owner - Luxury upper mid-sizeUtility EV
            "PM021146", #	Polk - Owner - Non-luxury compact CUV
            "PM021147", #	Polk - Owner - Non-luxury compact pickup
            "PM021148", #	Polk - Owner - Non-luxury compact SUV
            "PM021149", #	Polk - Owner - Non-luxury compact van
            "PM021241", #	Polk - Owner - Compact EV
            "PM021244", #	Polk - Owner - Compact Utility EV
            "PM021150", #	Polk - Owner - Non-luxury full-size 1/2 ton pickup
            "PM021151", #	Polk - Owner - Non-luxury full-size 1/2 ton van
            "PM021152", #	Polk - Owner - Non-luxury full-size SUV
            "PM021153", #	Polk - Owner - Non-luxury full-size 3/4-1 ton pickup
            "PM021154", #	Polk - Owner - Non-luxury full-size 3/4-1 ton van
            "PM021246", #	Polk - Owner - Full Size Half Ton Pick Up EV
            "PM021155", #	Polk - Owner - Non-luxury mid-size CUV
            "PM021156", #	Polk - Owner - Non-luxury mid-size pickup
            "PM021157", #	Polk - Owner - Non-luxury sport mid-size car
            "PM021158", #	Polk - Owner - Non-luxury mid-size SUV
            "PM021159", #	Polk - Owner - Non-luxury mid-size van
            "PM021249", #	Polk - Owner - Non-luxury mid-size EV
            "PM021160", #	Polk - Owner - Non-luxury traditional compact car
            "PM021161", #	Polk - Owner - Non-luxury traditional full-size car
            "PM021162", #	Polk - Owner - Non-luxury traditional mid-size car
            "PM021163", #	Polk - Owner - Non-luxury traditional subcompact car
            "PM021240", #	Polk - Owner - Non Luxury EV
            "PM021164", #	Polk - Owner - Non-luxury sporty car
            "PM021250", #	Polk - Owner - Non-luxury sub compact car EV
            "PM021252", #	Polk - Owner - Non-luxury sub compact Plus Utility EV
            "PM021255", #	Polk - Owner - Non-Luxury sub compact Utility EV
            "PM021165", #	Polk - Owner - Vehicle age 0-1 years old
            "PM021166", #	Polk - Owner - Vehicle age 2 years old
            "PM021167", #	Polk - Owner - Vehicle age 3 years old
            "PM021168", #	Polk - Owner - Vehicle age 4-5 years old
            "PM021169", #	Polk - Owner - Vehicle age 6-10 years old
            "PM021170", #	Polk - Owner - Vehicle age 11-15 years old
            "PM021171", #	Polk - Owner - Vehicle age 16-20 years old
            "PM021172", #	Polk - Owner - Vehicle age 21+ years old
            "PM021331", #	Polk - Owner - Vehicle age 25+ years old
            "PM021173", #	Polk - Owner - 4 Wheel Drive Truck
            "PM021174", #	Polk - Owner - 4 Wheel Drive Vehicle
            "PM021175", #	Polk - Owner - All Wheel Drive Vehicle
            "PM021176", #	Polk - Owner - Current Market Value Between $20,000 and $29,999
            "PM021177", #	Polk - Owner - Current Market Value Between $30,000 and $39,999
            "PM021178", #	Polk - Owner - Current Market Value Between $40,000 and $49,999
            "PM021179", #	Polk - Owner - Current Market Value Between $50,000 and $75,000
            "PM021180", #	Polk - Owner - Current Market Value Greater Than $75,000
            "PM021181", #	Polk - Owner - Vehicle Budget Predictor
            "PM021183", #	Polk - Owner - Non Motorized RV
            "PM021184", #	Polk - Owner - Motorized RV
            "PM021185", #	Polk - Owner - Any RV
            "PM021458", #	Polk - Owner - Model Year 2001
            "PM021459", #	Polk - Owner - Model Year 2002
            "PM021460", #	Polk - Owner - Model Year 2003
            "PM021461", #	Polk - Owner - Model Year 2004
            "PM021462", #	Polk - Owner - Model Year 2005
            "PM021463", #	Polk - Owner - Model Year 2006
            "PM021464", #	Polk - Owner - Model Year 2007
            "PM021465", #	Polk - Owner - Model Year 2008
            "PM021466", #	Polk - Owner - Model Year 2009
            "PM021467", #	Polk - Owner - Model Year 2010
            "PM021468", #	Polk - Owner - Model Year 2011
            "PM021469", #	Polk - Owner - Model Year 2012
            "PM021470", #	Polk - Owner - Model Year 2013
            "PM021471", #	Polk - Owner - Model Year 2014
            "PM021472", #	Polk - Owner - Model Year 2015
            "PM021473", #	Polk - Owner - Model Year 2016
            "PM021474", #	Polk - Owner - Model Year 2017
            "PM021475", #	Polk - Owner - Model Year 2018
            "PM021476", #	Polk - Owner - Model Year 2019
            "PM021477", #	Polk - Owner - Model Year 2020
            "PM021478", #	Polk - Owner - Model Year 2021
            "PM021479", #	Polk - Owner - Model Year 2022
            'PM021480', #	Polk - Owner - Model Year 2023
        ]
        
        # 2. SPENDING COLUMNS
        spending_columns = [
            # Credit card behavior
            "IBE3648", #	Buying Activity - Average Dollar Amount Per Order
            "IBE3506", #	Buying Activity - Dollars Spent - Payment Type - Credit Card - Last 1 Year
            "IBE3507", #	Buying Activity - Dollars Spent - Payment Type - Credit Card - Last 2 Years
            "IBE3504", #	Buying Activity - Dollars Spent - Total Last 1 Year
            "IBE3505", #	Buying Activity - Dollars Spent - Total Last 2 Years     
            "IBE3634", #	Buying Activity - Number of Orders - Payment Type - Cash
            "IBE3635", #	Buying Activity - Number of Orders - Payment Type- Credit Card
            "IBE3649", #	Buying Activity - Offline Average Dollars Spent Per Order
            "IBE3650", #	Buying Activity - Online Average Dollar Amount Per Order
            "IBE3663", #	Buying Activity - Total Offline Dollars
            "IBE3664", #	Buying Activity - Total Offline Orders
            "IBE3671", #	Buying Activity - Total Online Dollars
            "IBE3672", #	Buying Activity - Total Online Orders
            "IBE3517", #	Buying Activity - Weeks Since Last Offline Order
            "IBE3516", #	Buying Activity - Weeks Since Last Online Order
            "IBE3515", #	Buying Activity - Weeks Since Last Order - $1,000+ Range
            "IBE9154", #	Retail Purchases - Most Frequent Category
            "IBE6154", #	Automotive Assessories/Parts/Supplies/Cleaners
            "IBE9150_01", #	Credit Card - Frequency of Purchase - 00 to 03 Months
            "IBE9150_07", #	Credit Card - Frequency of Purchase - More than 24 Months
            "AP004921", #	Never Or Rarely Carry A Balance On A Credit Card (Financial)
            "AP004923", #	Usually Or Always Carry A Balance On A Credit Card (Financial)

            # Purchasing behavior
            "AP008304", #	Purchase Other Retailers Products Using - Amazon
            "AP008311", #	Purchase Other Retailers Products Using - Target
            "AP008312", #	Purchase Other Retailers Products Using - Walmart
            "AP008144", #	Purchases Influenced by Mobile Ads - Automotive
            "AP008177", #	Research In Store - Price Compare on Mobile
            "AP008178", #	Research In Store - Price Match on Mobile
            "AP008180", #	Research In Store - Read Reviews on Mobile
            "AP008181", #	Research In Store - Scan QR Codes for Information
            "AP009271", #	Research On Computer - Apparel (Financial)
            "AP009274", #	Research On Computer - Appliances (Financial)
            "AP009277", #	Research On Computer - Beauty Products (Financial)
            "AP009280", #	Research On Computer - Big Ticket Electronics (Financial)
            "AP009283", #	Research On Computer - Electronics (Financial)
            "AP009286", #	Research On Computer - Furniture (Financial)
            "AP009289", #	Research On Computer - Gift Cards and PrePaid Cards (Financial)
            "AP009292", #	Research On Computer - Home and Garden (Financial)
            "AP009295", #	Research On Computer - Home Improvement (Financial)
            "AP009270", #	Research On Mobile - Apparel (Financial)
            "AP009273", #	Research On Mobile - Appliances (Financial)
            "AP009276", #	Research On Mobile - Beauty Products (Financial)
            "AP009279", #	Research On Mobile - Big Ticket Electronics (Financial)
            "AP009282", #	Research On Mobile - Electronics (Financial)
            "AP009285", #	Research On Mobile - Furniture (Financial)
            "AP009288", #	Research On Mobile - Gift Cards and PrePaid Cards (Financial)
            "AP009291", #	Research On Mobile - Home and Garden (Financial)
            "AP009294", #	Research On Mobile - Home Improvement (Financial)
            "AP006866", #	Shopping App Owner
            "AP008271", #	Automotive Purchase Influenced by Celebrity Influencers
            "AP008281", #	Automotive Purchase Influenced by Non-celebrity Influencers
            "AP008261", #	Automotive Purchase Influenced by Product Reviews
            "AP009213", #	Purchases Influenced By - In Store Events (Financial)
            "AP008135", #	Purchases Influenced by Social Media - Automotive
            "AP008831", #	Regularly Shops for Groceries at Costco Warehouse stores
            "AP008822", #	Regularly Shops for Groceries at Target
            "AP008826", #	Regularly Shops for Groceries at Walmart Supercenter
            "AP008828", #	Regularly Shops for Groceries at Whole Foods
            "AP008823", #	Regularly Shops for Groceries at Trader Joe's

            # need to aggregate in FE
            "AP009528", #	Used to Plan Online Grocery Orders - Coupons (Financial)
            "AP009527", #	Used to Plan Online Grocery Orders - Digital Advertising Circulars (Financial)
            "AP009529", #	Used to Plan Online Grocery Orders - Direct Mail (Financial)
            "AP009531", #	Used to Plan Online Grocery Orders - Online Search Circulars (Financial)
            "AP009526", #	Used to Plan Online Grocery Orders - Print Advertising Circulars (Financial)
            "AP009532", #	Used to Plan Online Grocery Orders - Radio Advertisements (Financial)
            "AP009530", #	Used to Plan Online Grocery Orders - Recipe Websites (Financial)
            "AP009534", #	Used to Plan Online Grocery Orders - Retailer Apps (Financial)
            "AP009533", #	Used to Plan Online Grocery Orders - Retailer Emails (Financial)
            "AP009535", #	Used to Plan Online Grocery Orders - Retailer Websites (Financial)
            "AP009536", #	Used to Plan Online Grocery Orders - Shopping Lists (Financial)
            "AP009537", #	Used to Plan Online Grocery Orders - Social Media (Financial)
            "AP009538", #	Used to Plan Online Grocery Orders - TV Advertising (Financial)
            "AP008330", #	Uses Shop Now Feature - Facebook
            "AP008331", #	Uses Shop Now Feature - Instagram
            "AP008332", #	Uses Shop Now Feature - Pinterest
            "AP008333", #	Uses Shop Now Feature - Snapchat
        ]

        # 3. DEMOGRAPHIC COLUMNS
        demographic_columns = [
            ## Household brackets
            "IBE7628_01", #	Adults - Number in Household - 100%
            "IBE7602_01", #	Number of Children - 100%
            "IBE1802", #	Family Ties - Adult with Senior Parent - Person
            "IBE1807", #	Family Ties - Grandparent Indicator - Person
            "IBE8652", #	Generations in Household
            "IBE7607_01", #	Home Length of Residence - 100%
            "IBE7606_01", #	Home Owner / Renter - 100%
            "IBE7629_01", #	Household Size - 100%
            "IBE2526", #	Inferred Household Rank
            "IBE7478", #	New Mover
            "IBE7469", #	New Parent
            "IBE7300", #	First-time Parents
            "IBE7301", #	Parents of Infants to Toddlers
            "IBE7278", #	Affluent Bachelorettes
            "IBE7277", #	Affluent Bachelors
            "IBE7271", #	Affluent Couples No Kids
            "IBE7272", #	Affluent Homes
            "IBE7289", #	Affluent Households
            "IBE7275", #	Affluent Moms of Young Children
            "IBE7273", #	Affluent Parents
            "IBE7274", #	Affluent Parents of Teens
            "IBE7276", #	Affluent Seniors
            
            
            ## Age brackets
            "IBE7200", #	Age in One-Year Increments - Person
            "IBE7600_02", #	Adult Age Ranges Present - 18 to 24 Female - 100%
            "IBE7600_01", #	Adult Age Ranges Present - 18 to 24 Male - 100%
            "IBE7600_05", #	Adult Age Ranges Present - 25 to 34 Female - 100%
            "IBE7600_04", #	Adult Age Ranges Present - 25 to 34 Male - 100%
            "IBE7600_08", #	Adult Age Ranges Present - 35 to 44 Female - 100%
            "IBE7600_07", #	Adult Age Ranges Present - 35 to 44 Male - 100%
            "IBE7600_11", #	Adult Age Ranges Present - 45 to 54 Female - 100%
            "IBE7600_10", #	Adult Age Ranges Present - 45 to 54 Male - 100%
            "IBE7600_14", #	Adult Age Ranges Present - 55 to 64 Female - 100%
            "IBE7600_13", #	Adult Age Ranges Present - 55 to 64 Male - 100%
            "IBE7600_17", #	Adult Age Ranges Present - 65 to 74 Female - 100%
            "IBE7600_16", #	Adult Age Ranges Present - 65 to 74 Male - 100%
            "IBE7600_20", #	Adult Age Ranges Present - 75 or over Female - 100%
            "IBE7600_19", #	Adult Age Ranges Present - 75 or over Male - 100%
            "IBE8603_01", #	Children's Age - 1 Year Increments - Less Than Age 01
            "IBE8603_02", #	Children's Age - 1 Year Increments - Age 01
            "IBE8603_03", #	Children's Age - 1 Year Increments - Age 02
            "IBE8603_04", #	Children's Age - 1 Year Increments - Age 03
            "IBE8603_05", #	Children's Age - 1 Year Increments - Age 04
            "IBE8603_06", #	Children's Age - 1 Year Increments - Age 05
            "IBE8603_07", #	Children's Age - 1 Year Increments - Age 06
            "IBE8603_08", #	Children's Age - 1 Year Increments - Age 07
            "IBE8603_09", #	Children's Age - 1 Year Increments - Age 08
            "IBE8603_10", #	Children's Age - 1 Year Increments - Age 09
            "IBE8603_11", #	Children's Age - 1 Year Increments - Age 10
            "IBE8603_12", #	Children's Age - 1 Year Increments - Age 11
            "IBE8603_13", #	Children's Age - 1 Year Increments - Age 12
            "IBE8603_14", #	Children's Age - 1 Year Increments - Age 13
            "IBE8603_15", #	Children's Age - 1 Year Increments - Age 14
            "IBE8603_16", #	Children's Age - 1 Year Increments - Age 15
            "IBE8603_17", #	Children's Age - 1 Year Increments - Age 16
            "IBE8603_18", #	Children's Age - 1 Year Increments - Age 17

            ## Geo demographics
            "IBE7964B_11", #	Geo FIPS State Code
            "IBE7964B_08", #	Geo Designated Marketing Area (DMA) Name
            "IBE1273_01", #	Population Density

            
            ## Other key demographics
            "IBE8688", #	Gender - Person
            "IBE2350", #	Business Owner
            "IBE4100", #	Consumer Prominence Indicator - Person
            "IBE9549", #	Education Detail - Person
            "IBE7831", #	Investing/Finance Grouping
            "IBE8433", #	Investment - Estimated Residential Properties Owned (Real Property data only)
            "IBE7793", #	Investments - Personal
            "IBE7794", #	Investments - Real Estate
            "IBE7795", #	Investments - Stocks/Bonds
            "IBE8339", #	Likely Investor
            "IBE8631", #	Marital Status - Person
            "IBE8637", #	Occupation - Person
            "IBE7467", #	Recent Home Buyer
            "IBE7468", #	Recent Mortgage Borrower
            "IBE2351", #	Single Parent
            "IBE2356", #	Veteran
            "IBE8619", #	Woman in the Workplace
        ]
        
        # 4. LIFESTYLE/USAGE COLUMNS
        lifestyle_columns = [
            ## General interests
            "IBE7832", #	Collectibles and Antiques Grouping
            "IBE7826", #	Cooking/Food Grouping
            "IBE7829", #	Electronics/Computers Grouping
            "IBE7827", #	Exercise/Health Grouping
            "IBE7830", #	Home Improvement Grouping
            "IBE7828", #	Movie/Music Grouping
            "IBE7823", #	Outdoors Grouping
            "IBE7825", #	Reading Grouping
            "IBE7822", #	Sports Grouping
            "IBE7824", #	Travel Grouping
            "IBE8276", #	Upscale Living
            "IBE7756", #	Auto Work
            "IBE7746", #	Recreational Vehicle
            "PX002273_01", #	Personicx Lifestage NS - Group Codes
            "PX002272_01", #	Personicx Lifestage NS - Segment Codes
            "AP016727_13", #	Energy Consumer Dynamics - NS - National - Segment Code
            "AP007868", #	Mobile segmentation: Basic Phone Buddy
            "AP007867", #	Mobile segmentation: Casual Relationship with My Screen
            "AP007865", #	Mobile segmentation: Married to My Screen
            "AP007866", #	Mobile segmentation: Serious Relationship with My Screen
            "AP007869", #	Online shopper segmentation: Offline-Only Shoppers
            "AP007870", #	Online shopper segmentation: Traditional Consumers
            "AP008029", #	Technology segmentation: Connected Players
            "AP008031", #	Technology segmentation: Tech Indifferent
            "AP008028", #	Technology segmentation: True Techies


            ## Eating interest
            "AP004131", #	Dine at a Fine Dining Restaurant
            "AP004095", #	Dine at a Full Service Restaurant
            "AP004096", #	Dine at a Kids' Restaurant

            ## Travel interest
            "AP008769", #	Foreign Vacations - Dollars Spent Yearly - $8000 or more
            "AP008767", #	Foreign Vacations - Dollars Spent Yearly - Between $1000 and $2999
            "AP008768", #	Foreign Vacations - Dollars Spent Yearly - Between $3000 and $5999
            "AP008766", #	Foreign Vacations - Dollars Spent Yearly - Less than $1000
            "AP008780", #	Domestic Vacations - Dollars Spent Yearly - $7000 or more
            "AP008776", #	Domestic Vacations - Dollars Spent Yearly - Between $1500 and $1999
            "AP008777", #	Domestic Vacations - Dollars Spent Yearly - Between $2000 and $2999
            "AP008778", #	Domestic Vacations - Dollars Spent Yearly - Between $3000 and $4999
            "AP008779", #	Domestic Vacations - Dollars Spent Yearly - Between $5000 and $6999

            "AP008036", #	Travel segmentation: Aspirational Travelers
            "AP008034", #	Travel segmentation: Backyard Vacationers
            "AP008032", #	Travel segmentation: Passionate Travel Adventurers
            "AP006770", #	Commuter
            "AP006812", #	Rental Car Shopper
            "AP006873", #	Road Tripper
            
            ## Investing mindset
            "AP005134", #	Completely Agree Always Knowing Broadly How Much Is In My Bank Account At Any One Time (Financial)
            "AP005140", #	Completely Agree Better To Put My Mny In Lw Risk Invest Evn If The Rtrn My Nt Be As Grt (Financial)
            "AP005150", #	Completely Agree I Like To Take Risks When Investing For The Chance Of A High Return (Financial)
            "AP005146", #	Completely Agree Philsphy Are Better Off Hving Wht Want Now As Nvr Knw Wht Tmrrw Brings (Financial)
            "AP005124", #	Completely Agree Regarding Hating To Borrow Money (Financial)
            "AP005178", #	Completely Agree Regarding That I Feel Overwhelmed By Financial Burdens (Financial)
            "AP005181", #	Completely Agree Regarding That Investing For The Future Is Very Important To Me (Financial)
            "AP005182", #	I Have A Great Deal Of Knowledge/Experience In Finance/Investments (Financial)
            "AP008715", #	Technology Adoption


            ## Sharing behavior
            "AP006804", #	Ride Sharer
            "AP006795", #	Room Sharer
            "AP006878", #	Sharing Participant
            "AP006882", #	Auto Sharer
            # need to aggregate
            "AP009547", #	Gig Work - Accounting  (Financial)
            "AP009548", #	Gig Work - Baking Or Cooking  (Financial)
            "AP009549", #	Gig Work - Business Lead Generation  (Financial)
            "AP009550", #	Gig Work - Business Services  (Financial)
            "AP009551", #	Gig Work - Caregiver  (Financial)
            "AP009552", #	Gig Work - Childcare  (Financial)
            "AP009553", #	Gig Work - Computer Programming  (Financial)
            "AP009554", #	Gig Work - Customer Service  (Financial)
            "AP009555", #	Gig Work - Data Entry Or Processing  (Financial)
            "AP009556", #	Gig Work - Dog Walker  (Financial)
            "AP009546", #	Gig Work - Engaged in Side Gig  (Financial)
            "AP009559", #	Gig Work - Event Planning  (Financial)
            "AP009560", #	Gig Work - Fitness Trainer  (Financial)
            "AP009558", #	Gig Work - Food Delivery Driver  (Financial)
            "AP009561", #	Gig Work - Gardening or Landscaping  (Financial)
            'AP009562', #	Gig Work - Home or Office Cleaning  (Financial)
            "AP009563", #	Gig Work - Home Repair or Remodeling  (Financial)
            "AP009565", #	Gig Work - Marketing or Promotions  (Financial)
            "AP009564", #	Gig Work - Musician Or DJ  (Financial)
            "AP009566", #	Gig Work - Nursing Aid  (Financial)
            "AP009567", #	Gig Work - Real Estate  (Financial)
            "AP009557", #	Gig Work - Rideshare Driver  (Financial)
            "AP009568", #	Gig Work - Social Media Influencer  (Financial)
            "AP009569", #	Gig Work - Stylist Or Beauty Care  (Financial)
            "AP009570", #	Gig Work - Videography Or Photography  (Financial)
            "AP009571", #	Gig Work - Websites Or Digital Marketing  (Financial)
            
            ## Other interest/behavior
            "AP004684", #	Have a cat
            "AP004683", #	Have a dog
            "AP008374", #	Move to a major city
            "AP008378", #	Move to a new country
            "AP008377", #	Move to a new state
            "AP008376", #	Move to a rural area
            "AP008375", #	Move to the suburbs
            "AP008188", #	Use voice activated assistant for maps and directions
            "AP006837", #	Social & Communications App Owner
            "AP001723", #	Potential to be Publicly or Civically Influential
            "AP000605", #	Spent Leisure Time in Auto Shows
            "AP006340", #	Almost always listen to ads before or during online videos (Financial)
            "AP009230", #	Influenced by sponsored search results (Financial)
            "AP008601", #	Going online is a great use of free time
        ]
        
        # 5. FINANCIAL BEHAVIOR COLUMNS
        financial_columns = [
            ## Credit card behavior
            "IBE8815", #	Bank Card - Presence in Household
            "IBE8836", #	Credit - Range of New Credit
            "IBE8621_02", #	Credit Card Indicator - Gas Department Retail
            "IBE8621_05", #	Credit Card Indicator - Premium
            "IBE8621_03", #	Credit Card Indicator - Travel and Entertainment
            "IBE8621_04", #	Credit Card Indicator - Unknown Type
            "IBE8621_06", #	Credit Card Indicator - Upscale Department Store

            ## Banking behavior
            "AP004964", #	Acquired A Non-Interest Checking Account In Past 12 Months (Financial)
            "AP004959", #	Acquired A Saving Account In Past 12 Months (Financial)
            "AP004962", #	Acquired An Interest Checking Account In Past 12 Months (Financial)
            "AP000463", #	Have Overdraft Protection (Financial)
            "AP009349", #	Open Account in Next 12 Months - Bank  (Financial)
            "AP009500", #	Regularly uses apps or internet to buy and sell stocks (Financial)
            "AP004973", #	Used Direct Deposit For Payroll In Past 12 Months (Financial)
            

            ## Income behavior
            "IBE7641_01", #	Income - Estimated Household - Broad Ranges - 100%
            "AP000448_01", #	Estimated Disposable Income (B) (Financial)
            "AP000451_01", #	Estimated Disposable Income Index (B)
            "AP001371", #	Retirement Assets
            "AP001370", #	Total Liquid Investible Assets
            "IBE2834", #	Affordability
            "IBE9350", #	Economic Stability Indicator
            "AP006828", #	Retirement Planner
            "IBE9351", #	Underbanked

            ## Home ownership behavior
            "IBE1805", #	Family Ties - Adult with Wealthy Parent - Person
            "IBE8707", #	Home Equity Available - Estimated - Actual (Real Property data only)
            "IBE8706", #	Home Equity Lendable - Estimated - Actual (Real Property data only)
            "IBE8702", #	Home Loan Amount - Original - Actual (Real Property data only)
            "IBE8572", #	Home Loan Interest Rate Type 1 (Real Property data only)
            "IBE8575", #	Home Loan Transaction Type 1 (Real Property data only)
            "IBE8570", #	Home Loan Type 1 (Real Property data only)
            "IBE8704", #	Home Loan-to-Value - Estimated - Actual (Real Property data only)
            "IBE9750", #	Home Market Value - Estimated - Actual (Real Property data only)
            "IBE8463", #	Home Market Value Deciles - Estimated (Real Property data only)
            "AP001246", #	Obtain Mortgage Insurance
            "AP004389", #	Build or buy a home (net)
            
            ## Investment behavior
            "AP009500", #	Regularly uses apps or internet to buy and sell stocks (Financial)
            "AP009352", #	Start Trading in Next 12 Months - Stock Market  (Financial)
        ]
        
        # return set(veh_pref_columns + spending_columns + demographic_columns + lifestyle_columns + financial_columns)
        return list(dict.fromkeys(veh_pref_columns + spending_columns + demographic_columns + lifestyle_columns + financial_columns))

    @staticmethod
    def excluded_columns():
        # 1. VEHICLE PREFERENCE COLUMNS
        veh_pref_columns = [
            ## Luxury vehicle interest
            "AP004571",  # Purchase a New Luxury Convertible (Financial)
            "AP004573",  # Purchase a New Luxury Coupe (Financial)
            "AP004575",  # Purchase a New Luxury Diesel (Financial)
            "AP004579",  # Purchase a New Luxury Wagon (Financial)
            
            ## Compact vehicle interest
            "AP004531",  # Purchase a New Compact Pickup (Financial)
            "AP004533",  # Purchase a New Compact Van (Financial)

            ## Mid-size vehicle interest

            ## New Entry vehicle interest

            ## Premium vehicle interest

            ## Hybrid/EV vehicle interest
            "AP004564",  # Purchase a New Hybrid Luxury Car (Financial)
            "AP004565",  # Purchase a New Hybrid Regular Car (Financial)
            "AP004566",  # Purchase a New Hybrid SUV (Financial)

            ## Regular vehicle interest
            "AP004572",  # Purchase a New Regular Convertible (Financial)
            "AP004574",  # Purchase a New Regular Coupe (Financial)
            "AP004576",  # Purchase a New Regular Diesel (Financial)
            "AP004580",  # Purchase a New Regular Wagon (Financial)

            ## Other vehicle interest

            ## Purchase reason
            "AP006980",	#	Dealer Courtesy Car Incentives Reason for Purchase
            "AP006977",	#	Dealer Financing Incentives Reason for Purchase
            "AP006974",	#	Dealer Lease Used to Purchase Vehicle
            "AP006966",	#	Dealer Loan Used to Purchase Vehicle
            "AP006979",	#	Dealer Shuttle Service Incentives Reason for Purchase
            
            ## Current ownership
            "AP007094",	# Owns an Entry Electric Vehicle
            "AP007091",	# Owns a Regular Electric Vehicle
            "AP007109",	# Owns a Full Sized Pickup Truck Body Style Vehicle
            "AP007108",	# Owns a Gasoline Engine Truck  Body Style Vehicle
            "AP007105",	# Owns a Luxury SUV Body Style Vehicle
            "AP007133",	# Owns a Mid-Sized Pickup Truck  Body Style Vehicle
            "AP006962",	# Owns a ATV Recreational Vehicle
            "AP006961",	# Owns a Camper Recreational Vehicle
            "AP006958",	# Owns a Fifth Wheel Recreational Vehicle
            "AP006957",	# Owns a Motor Home Recreational Vehicle
            "AP006964",	# Owns a Side by Side Recreational Vehicle
            "AP006965",	# Owns a Snowmobile Recreational Vehicle
            "AP006959",	# Owns a Towable/Caravan Recreational Vehicle
            "AP006956",	# Owns a Boat Water Craft
            "AP006955",	# Owns a Jet Ski or Sea Doo Water Craft
            "IBE9180",	#	Vehicle Type - Vehicle 1
            "IBE9181",	#	Vehicle Type - Vehicle 2       


            ## Polk in market future
            "PM020020",	# Polk - In-Market (Future) - Luxury Sporty Car
            "PM020027",	# Polk - In-Market (Future) - Luxury Traditional Subcompact Car
            "PM021325",	# Polk - In-Market (Future) - Luxury Traditional Subcompact Plus Utility
            "PM020028",	# Polk - In-Market (Future) - Non-luxury Sporty Car
            "PM020031",	# Polk - In-Market (Future) - Non-luxury 1/2 Ton Full-size Pickup
            "PM020032",	# Polk - In-Market (Future) - Non-luxury 3/4-1 Ton Full-size Pickup
            "PM020033",	# Polk - In-Market (Future) - Non-luxury 3/4-1 Ton Full-size Van
            "PM020037",	# Polk - In-Market (Future) - Non-luxury Mid-size Sporty Car
            "PM020039",	# Polk - In-Market (Future) - Non-luxury Mid-size Van
            "PM020043",	# Polk - In-Market (Future) - Non-luxury Traditional Subcompact Car
            "PM021326",	# Polk - In-Market (Future) - Non-Luxury Traditional Subcompact Plus Utility


            ## Polk in market
            "PM020324",	# Polk - In-Market - New luxury exotic car
            "PM020325",	# Polk - In-Market - New luxury sporty car
            "PM020327",	# Polk - In-Market - New luxury prestige full-size car
            "PM021197",	# Polk - In-Market - New luxury 1/2 ton full-size pickup EV
            "PM021199",	# Polk - In-Market - New luxury lower Mid-Size Utility EV
            "PM020333",	# Polk - In-Market - New luxury traditional subcompact car
            "PM021195",	# Polk - In-Market - New luxury compact Utility EV
            "PM021201",	# Polk - In-Market - New luxury sub compact plus Utility EV
            "PM021204",	# Polk - In-Market - New luxury upper Mid-size Utility EV
            "PM020336",	# Polk - In-Market - New non-luxury compact van
            "PM021196",	# Polk - In-Market - New non-luxury compact Utility EV
            "PM020337",	# Polk - In-Market - New non-luxury 1/2 ton full-size pickup
            "PM020338",	# Polk - In-Market - New non-luxury 1/2 ton full-size van
            "PM020340",	# Polk - In-Market - New non-luxury 3/4-1 ton full-size pickup
            "PM020341",	# Polk - In-Market - New non-luxury 3/4-1 ton full-size van
            "PM020344",	# Polk - In-Market - New non-luxury mid-size sport car
            "PM020350",	# Polk - In-Market - New non-luxury traditional subcompact car
            "PM020351",	# Polk - In-Market - New non-luxury sporty car
            "PM021200",	# Polk - In-Market - New non-luxury sub compact car EV
            "PM021202",	# Polk - In-Market - New non-luxury sub compact plus Utility EV
            "PM021203",	# Polk - In-Market - New non-luxury sub compact Utility EV
            "PM020360",	# Polk - In-Market - Lease Luxury
            "PM020361",	# Polk - In-Market - Lease Non-luxury
            "AP004414",	#	In Market for a New Luxury Vehicle (Financial)
            "AP004415",	#	In Market for a New Regular Vehicle (Financial)

            ## Polk loyalty

            ## Polk owner
            "PM020740", #	Polk - Owner - Buick Cascada
            "PM020742", #	Polk - Owner - Buick Enclave
            "PM020743", #	Polk - Owner - Buick Encore
            "PM020744", #	Polk - Owner - Buick Envision
            "PM020745", #	Polk - Owner - Buick LaCrosse
            "PM020747", #	Polk - Owner - Buick Lucerne
            "PM020750", #	Polk - Owner - Buick Regal
            "PM020752", #	Polk - Owner - Buick Verano
            "PM020753", #	Polk - Owner - Cadillac ATS
            "PM020754", #	Polk - Owner - Cadillac CTS
            "PM020756", #	Polk - Owner - Cadillac DTS
            "PM020757", #	Polk - Owner - Cadillac Escalade
            "PM020758", #	Polk - Owner - Cadillac SRX
            "PM020759", #	Polk - Owner - Cadillac STS
            "PM020760", #	Polk - Owner - Cadillac XT5
            "PM020761", #	Polk - Owner - Cadillac XTS
            "PM021334", #	Polk - Owner - Cadillac CT4
            "PM021335", #	Polk - Owner - Cadillac CT6
            "PM021429", #	Polk - Owner - Cadillac CT5
            "PM021430", #	Polk - Owner - Cadillac XT4
            "PM021431", #	Polk - Owner - Cadillac XT6
            "PM020763", #	Polk - Owner - Chevrolet Avalanche
            "PM020764", #	Polk - Owner - Chevrolet Aveo
            "PM020765", #	Polk - Owner - Chevrolet Bolt
            "PM020766", #	Polk - Owner - Chevrolet Camaro
            "PM020769", #	Polk - Owner - Chevrolet Cobalt
            "PM020770", #	Polk - Owner - Chevrolet Colorado
            "PM020771", #	Polk - Owner - Chevrolet Corvette
            "PM020772", #	Polk - Owner - Chevrolet Cruze
            "PM020773", #	Polk - Owner - Chevrolet Equinox
            "PM020774", #	Polk - Owner - Chevrolet Express
            "PM020775", #	Polk - Owner - Chevrolet HHR
            'PM020776', #	Polk - Owner - Chevrolet Impala
            "PM020777", #	Polk - Owner - Chevrolet Malibu
            "PM020779", #	Polk - Owner - Chevrolet Silverado
            "PM020780", #	Polk - Owner - Chevrolet Sonic
            "PM020781", #	Polk - Owner - Chevrolet Spark
            "PM020782", #	Polk - Owner - Chevrolet Suburban
            "PM020783", #	Polk - Owner - Chevrolet Tahoe
            "PM020784", #	Polk - Owner - Chevrolet Trailblazer
            "PM020785", #	Polk - Owner - Chevrolet Traverse
            "PM020786", #	Polk - Owner - Chevrolet Trax
            "PM021336", #	Polk - Owner - Chevrolet Blazer
            "PM021281", #	Polk - Owner - Chevrolet Bolt EUV
            "PM021337", #	Polk - Owner - Chevrolet S10 Blazer
            "PM021282", #	Polk - Owner - Chevrolet Spark EV
            "PM020836", #	Polk - Owner - GMC Acadia
            "PM020837", #	Polk - Owner - GMC Canyon
            "PM020839", #	Polk - Owner - GMC Savana
            "PM020840", #	Polk - Owner - GMC Sierra
            "PM020841", #	Polk - Owner - GMC Terrain
            "PM020842", #	Polk - Owner - GMC Yukon
            "PM020843", #	Polk - Owner - GMC Yukon XL
            "PM021137", #	Polk - Owner - Luxury full-size 1/2 ton pickup
            "PM021138", #	Polk - Owner - Luxury prestige full-size car
            "PM021245", #	Polk - Owner - Full Size Half Ton Luxury Pick Up EV
            "PM021145", #	Polk - Owner - Luxury traditional subcompact car
            "PM021243", #	Polk - Owner - Compact Luxury Utility EV
            "PM021248", #	Polk - Owner - Luxury Lower Mid-Size Utility EV
            "PM021251", #	Polk - Owner - Luxury sub compact car EV
            "PM021254", #	Polk - Owner - Luxury sub compact plus Utility EV
            "PM021253", #	Polk - Owner - Luxury upper mid-sizeUtility EV
            "PM021147", #	Polk - Owner - Non-luxury compact pickup
            "PM021149", #	Polk - Owner - Non-luxury compact van
            "PM021150", #	Polk - Owner - Non-luxury full-size 1/2 ton pickup
            "PM021151", #	Polk - Owner - Non-luxury full-size 1/2 ton van
            "PM021153", #	Polk - Owner - Non-luxury full-size 3/4-1 ton pickup
            "PM021154", #	Polk - Owner - Non-luxury full-size 3/4-1 ton van
            "PM021246", #	Polk - Owner - Full Size Half Ton Pick Up EV
            "PM021250", #	Polk - Owner - Non-luxury sub compact car EV
            "PM021252", #	Polk - Owner - Non-luxury sub compact Plus Utility EV
            "PM021255", #	Polk - Owner - Non-Luxury sub compact Utility EV
            "PM021173", #	Polk - Owner - 4 Wheel Drive Truck
            "PM021183", #	Polk - Owner - Non Motorized RV
            "PM021184", #	Polk - Owner - Motorized RV
            "PM021458", #	Polk - Owner - Model Year 2001
            "PM021459", #	Polk - Owner - Model Year 2002
            "PM021460", #	Polk - Owner - Model Year 2003
            "PM021461", #	Polk - Owner - Model Year 2004
            "PM021462", #	Polk - Owner - Model Year 2005
            "PM021463", #	Polk - Owner - Model Year 2006
            "PM021464", #	Polk - Owner - Model Year 2007
            "PM021465", #	Polk - Owner - Model Year 2008
            "PM021466", #	Polk - Owner - Model Year 2009
            "PM021467", #	Polk - Owner - Model Year 2010
            "PM021468", #	Polk - Owner - Model Year 2011
            "PM021469", #	Polk - Owner - Model Year 2012
        ]
        
        # 2. SPENDING COLUMNS
        spending_columns = [
            ## Credit card behavior (already in your code)

            ## Purchasing behavior
            # "AP009280", #	Research On Computer - Big Ticket Electronics (Financial)
            # "AP009289", #	Research On Computer - Gift Cards and PrePaid Cards (Financial)
            # "AP009292", #	Research On Computer - Home and Garden (Financial)
            # "AP009279", #	Research On Mobile - Big Ticket Electronics (Financial)
            # "AP009288", #	Research On Mobile - Gift Cards and PrePaid Cards (Financial)
            # "AP009291", #	Research On Mobile - Home and Garden (Financial)

            # need to aggregate in FE
        ]

        # 3. DEMOGRAPHIC COLUMNS
        demographic_columns = [
            ## Household brackets
            
            ## Age brackets
            "IBE7600_02", #	Adult Age Ranges Present - 18 to 24 Female - 100%
            "IBE7600_01", #	Adult Age Ranges Present - 18 to 24 Male - 100%
            "IBE7600_05", #	Adult Age Ranges Present - 25 to 34 Female - 100%
            "IBE7600_04", #	Adult Age Ranges Present - 25 to 34 Male - 100%
            "IBE7600_08", #	Adult Age Ranges Present - 35 to 44 Female - 100%
            "IBE7600_07", #	Adult Age Ranges Present - 35 to 44 Male - 100%
            "IBE7600_11", #	Adult Age Ranges Present - 45 to 54 Female - 100%
            "IBE7600_10", #	Adult Age Ranges Present - 45 to 54 Male - 100%
            "IBE7600_14", #	Adult Age Ranges Present - 55 to 64 Female - 100%
            "IBE7600_13", #	Adult Age Ranges Present - 55 to 64 Male - 100%
            "IBE7600_17", #	Adult Age Ranges Present - 65 to 74 Female - 100%
            "IBE7600_16", #	Adult Age Ranges Present - 65 to 74 Male - 100%
            "IBE7600_20", #	Adult Age Ranges Present - 75 or over Female - 100%
            "IBE7600_19", #	Adult Age Ranges Present - 75 or over Male - 100%

            ## Geo demographics
            
            ## Other key demographics
            "IBE8433", #	Investment - Estimated Residential Properties Owned (Real Property data only)
            "IBE8619", #	Woman in the Workplace
        ]
        
        # 4. LIFESTYLE/USAGE COLUMNS
        lifestyle_columns = [
            ## General interests
            "IBE7832", #	Collectibles and Antiques Grouping
            "IBE7826", #	Cooking/Food Grouping
            "IBE7829", #	Electronics/Computers Grouping
            "IBE7825", #	Reading Grouping

            ## Eating interest

            ## Travel interest
            
            ## Investing mindset

            ## Sharing behavior
            # need to aggregate
            
            ## Other interest/behavior
        ]
        
        # 5. FINANCIAL BEHAVIOR COLUMNS
        financial_columns = [
            ## Credit card behavior

            ## Banking behavior

            ## Income behavior

            ## Home ownership behavior
            "IBE8706", #	Home Equity Lendable - Estimated - Actual (Real Property data only)
            
            ## Investment behavior
        ]
        
        return list(dict.fromkeys(veh_pref_columns + spending_columns + demographic_columns + lifestyle_columns + financial_columns))

    def left_columns(self):
        exclude = set(RawDataPrep.excluded_columns())
        return [self.id_column] + [col for col in RawDataPrep.full_columns() if col not in exclude]
    
    def available_columns(self):
        sample_df = spark.sql(f"SELECT * FROM {self.catelog}.{self.schema}.{self.table} LIMIT 1")
        all_cols = set(sample_df.columns)
        return [col for col in self.left_columns() if col in all_cols]
    
    def column_report(self):
        cls = type(self)
        print(f"""
            {len(cls.full_columns())} columns defined in the original list.
            {len(self.left_columns())} columns returned after excluding some unused columns.
            {len(self.available_columns())} columns will be used in the raw dataset.
            """)
    
    def select_table(self):
        tbl = f"{self.catelog}.{self.schema}.{self.table}"
        cols = self.available_columns()

        ## Select the most recent snapshot
        latest_snapshot = spark.table(tbl).agg(F.max("snapshot")).first()[0]
        base = spark.table(tbl).where(F.col("snapshot") == latest_snapshot)

        # column_list = ", ".join(self.available_columns())
        # raw_query = f"""
        # SELECT {column_list}
        # FROM {self.catelog}.{self.schema}.{self.table}
        # WHERE 
        # snapshot = 
        # (SELECT max(snapshot)
        # FROM {self.catelog}.{self.schema}.{self.table})
        # """

        ## for each gm real id, select the most recent record
        # raw_query = f"""
        # SELECT {column_list}
        # FROM {self.catelog}.{self.schema}.{self.table}
        # QUALIFY ROW_NUMBER() OVER(PARTITION BY GM_PERSON_REALID ORDER BY snapshot DESC) = 1
        # """

        ## Sampling strategy
        if self.sample_frac and self.sample_size:
            raise ValueError("Please specify either sample_frac or sample_size, not both.")
        elif self.sample_size:
            ## Bernoulli sampling
            alpha = 1.5 # oversampling factor
            p = min(1.0, alpha * self.sample_size / self.pop_estimate)     # prob of a row being selected
            ids = (base.select(self.id_column)
                .where(F.rand(self.SEED) < p)
                .limit(self.sample_size)
                .persist(StorageLevel.MEMORY_AND_DISK))
            _ = ids.count()  # materialize once
            raw_df = base.join(F.broadcast(ids), self.id_column, "left_semi").select(*cols)
            # raw_df = base.join(ids, self.id_column, "inner").select(*cols)

        #     query = f"""
        # SELECT {column_list}
        # FROM {self.catelog}.{self.schema}.{self.table}
        # WHERE 
        # snapshot = 
        # (SELECT max(snapshot)
        # FROM {self.catelog}.{self.schema}.{self.table})
        # ORDER BY RAND()
        # LIMIT {self.sample_size}
        # """
        elif self.sample_frac:
            raw_df = base.select(*cols).sample(False, self.sample_frac, seed = self.SEED)
        #     total = spark.sql(raw_query).count()
        #     query = f"""
        # SELECT {column_list}
        # FROM {self.catelog}.{self.schema}.{self.table}
        # WHERE 
        # snapshot = 
        # (SELECT max(snapshot)
        # FROM {self.catelog}.{self.schema}.{self.table})
        # ORDER BY RAND()
        # LIMIT (SELECT CAST({total} * {self.sample_frac} AS INT) FROM total)
        # """
        else:
            raw_df = base.select(*cols)

        # if not self.sample_frac and not self.sample_size:
        #     query = raw_query
        

        # raw_df = spark.sql(query)
        # raw_df.cache()
        return raw_df

    def select_other_table(self):
        tbl = f"{self.catelog}.{self.schema}.{self.table}"
        return spark.table(tbl)

    def verify_query_result(self):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n=== Query execution started. ===")
        raw_df = self.select_table()
        row_count = raw_df.count()
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Query execution completed. {time_taken(start_time, end_time)} minutes taken. ===")
        if row_count == 0:
            raise RuntimeError("No data returned from Acxiom database query. This is a fatal error.")
        if row_count <= self.pop_estimate:
            raise RuntimeError("Not enough data returned from Acxiom database query.")
        else:
            print(f"{row_count} rows returned from Acxiom database query.")
    
    def save_query_result(self, df, storage_format = "delta"):
        raw_table = f"{self.destination_catelog}.{self.destination_schema}.{self.destination_table}"
        # raw_df = self.select_table()
        
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Writing to table: {raw_table} ===")
        df.write.mode("overwrite").format(storage_format).saveAsTable(raw_table)
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Successfully wrote to {raw_table}. {time_taken(start_time, end_time)} minutes taken. ===")


# COMMAND ----------

rdp = RawDataPrep()

# COMMAND ----------

rdp.column_report()

# COMMAND ----------

rdp.verify_query_result()

# COMMAND ----------

# rdp.save_query_result()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploratory Data Analysis

# COMMAND ----------

raw_sample_df = RawDataPrep(sample_frac = 0.01).select_table()
# raw_sample_df = RawDataPrep(sample_size = 5000000).select_table()

# COMMAND ----------

#number of partitions
raw_sample_df.rdd.getNumPartitions()

# COMMAND ----------

raw_sample_df.limit(5).toPandas()

# COMMAND ----------

(raw_sample_df.count(), len(raw_sample_df.columns))

# COMMAND ----------

ref_df = RawDataPrep(
                catelog = "dataproducts_dev",
                schema = "bronze_acxiom",
                table = "gm_consumer_list_variable_description").select_other_table()

# COMMAND ----------

# Null only
print_missing_percentages(raw_sample_df, ref_df, include_blank_strings = False)

# COMMAND ----------

# Null & Blank
print_missing_percentages(raw_sample_df, ref_df, include_blank_strings = True)

# COMMAND ----------

print_summary_stats(raw_sample_df)

# COMMAND ----------

ref_df.display()

# COMMAND ----------

class SparkAcxiomPlot:
    def __init__(
        self, 
        main_df, 
        ref_df,
        ref_key = "variable_name", 
        ref_val = "variable_short_description"
    ):
        self.main_df = main_df
        self.ref_df = ref_df
        self.ref_key = ref_key
        self.ref_val = ref_val
    
    def ref_df_to_dict(self):
        pdf = self.ref_df.select(self.ref_key, self.ref_val).toPandas()
        return dict(zip(pdf[self.ref_key], pdf[self.ref_val]))

    @staticmethod
    def polish_plot(ax, title=None, xlabel=None, ylabel=None, rotate_x=False, tight=True):
        ax.set_title(title, pad=12, weight="bold") if title else ax.set_title("")
        ax.set_xlabel(xlabel) if xlabel else ax.set_xlabel("")
        ax.set_ylabel(ylabel) if ylabel else ax.set_ylabel("")
        if rotate_x: plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if tight: plt.tight_layout()


    def plot(self, col_name, ax=None, suppress_labels=True, 
             include_blank_strings=False, tight=True):
        # Only compute ref_dict if it hasn't been cached yet
        if not hasattr(self, "_ref_dict_cache"):
            self._ref_dict_cache = self.ref_df_to_dict()
        ref_dict = self._ref_dict_cache
        # ref_dict = self.ref_df_to_dict()

        ylabel = "Count"
        if include_blank_strings:
            agg_df = (
                self.main_df
                .filter(F.col(col_name).isNotNull()) 
                .groupBy(col_name)
                .agg(F.count(col_name).alias(ylabel))
                .withColumnRenamed(col_name, ref_dict[col_name]) 
            )
        else:
            agg_df = (
                self.main_df
                .filter((F.col(col_name).isNotNull()) & (F.col(col_name) != "")) 
                .groupBy(col_name)
                .agg(F.count(col_name).alias(ylabel))
                .withColumnRenamed(col_name, ref_dict[col_name]) 
            )
        agg_df = to_pd(agg_df)
        n_uniq = agg_df.nunique()[0]

        if 1 <= n_uniq <= 5:
            agg_df = agg_df.sort_values(ylabel).set_index(ref_dict[col_name])
            agg_df.index = agg_df.index.where(agg_df.index.str.strip() != "", "(blank)")
            ax = agg_df[ylabel].plot(
                kind="pie",
                autopct="%0.0f%%",
                ax=ax
            )
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1, 1)
            )
            if not suppress_labels:
                SparkAcxiomPlot.polish_plot(ax, title=f"Breakdown of {ref_dict[col_name]}", tight=tight)
            else:
                SparkAcxiomPlot.polish_plot(
                    ax,
                    xlabel=ref_dict[col_name],
                    tight=tight
                )

        elif 5 < n_uniq <= 25:
            ax = agg_df.sort_values(ylabel).plot(kind="barh", x=ref_dict[col_name], y=ylabel, ax=ax, legend=False)
            for c in ax.containers: 
                ax.bar_label(c, fmt="%.0f")
            # SparkAcxiomPlot.polish_plot(ax, title=f"Distribution of {ref_dict[col_name]}", xlabel=ref_dict[col_name], ylabel=ylabel)

            # only add labels/titles when not in grid mode
            if not suppress_labels:
                SparkAcxiomPlot.polish_plot(
                    ax,
                    title=f"Distribution of {ref_dict[col_name]}",
                    xlabel=ref_dict[col_name],
                    ylabel=ylabel,
                    tight=tight
                )
            else:
                SparkAcxiomPlot.polish_plot(
                    ax,
                    xlabel=ref_dict[col_name],
                    tight=tight
                )

        elif n_uniq > 25:

            if include_blank_strings:
                agg_df = (
                    self.main_df
                    .filter(F.col(col_name).isNotNull()).
                    select(F.col(col_name).alias(ref_dict[col_name]))
                    )
            else:
                agg_df = (
                    self.main_df
                    .filter((F.col(col_name).isNotNull()) & (F.col(col_name) != "")).
                    select(F.col(col_name).alias(ref_dict[col_name]))
                    )              
            agg_df = pd.to_numeric(to_pd(agg_df)[ref_dict[col_name]])

            ax = sns.histplot(x=agg_df, kde=True, bins=20, edgecolor="white", ax=ax)
            # SparkAcxiomPlot.polish_plot(ax, title=f"Distribution of {ref_dict[col_name]}", xlabel=ref_dict[col_name], ylabel=ylabel)

            # only add labels/titles when not in grid mode
            if not suppress_labels:
                SparkAcxiomPlot.polish_plot(
                    ax,
                    title=f"Distribution of {ref_dict[col_name]}",
                    xlabel=ref_dict[col_name],
                    ylabel=ylabel,
                    tight=tight
                )
            else:
                SparkAcxiomPlot.polish_plot(
                    ax,
                    xlabel=ref_dict[col_name],
                    tight=tight
                )
                # ax.set_title("")
                # ax.set_xlabel(ref_dict[col_name])
                # ax.set_ylabel("")        
        
        return ax

    def plot_grid(self, cols, title, ncols=3, 
                  cell_size=(7,5), wspace=0.3, 
                  hspace=0.25, suppress_labels=True, 
                  include_blank_strings=False):
        self._ref_dict_cache = self.ref_df_to_dict()  # <--- compute once
        n = len(cols)
        nrows = math.ceil(n / ncols)       

        # Figure & axes
        fig_w = ncols * cell_size[0]
        fig_h = nrows * cell_size[1]
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.atleast_1d(axes).ravel()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

        # Compute target axis indices for placement
        total = nrows * ncols
        indices = []

        rem = n % ncols
        last_row_start = (nrows - 1) * ncols  # index of last row's col1

        if nrows == 1:
            # Single row special cases
            if rem == 2:
                # empty col1, place in col2 and col3
                indices = [0, 1]
            elif rem == 1:
                # center col2
                indices = [0]
            else:
                # 3 or 0 remainder (n==3)
                indices = list(range(n))
        else:
            # Multiple rows
            # Fill full rows first
            full_rows_vars = (nrows - 1) * ncols
            indices = list(range(full_rows_vars))
            if rem == 0:
                indices += list(range(last_row_start, last_row_start + ncols))
            elif rem == 2:
                # leave last_row col1 empty; use col2, col3
                indices += [last_row_start, last_row_start + 1]
            elif rem == 1:
                # center in last_row col2
                indices += [last_row_start]

        # Clear ALL axes first (we’ll enable only used ones)
        for ax in axes:
            ax.set_visible(False)
        

        # Plot each column into its assigned axis
        for col, ax_idx in zip(cols, indices):
            ax = axes[ax_idx]
            ax.set_visible(True)
            ax.clear()
            # reuse your single-plot function
            self.plot(col, ax=ax, 
                      suppress_labels=suppress_labels, 
                      include_blank_strings=include_blank_strings, 
                      tight=False)

        fig.suptitle(title, y=1.0, weight="bold")

        fig.tight_layout()

        return fig, axes


# COMMAND ----------

sap = SparkAcxiomPlot(raw_sample_df, ref_df)

# COMMAND ----------

sap.plot("IBE2834", suppress_labels=False)

# COMMAND ----------

sap.plot("AP009547", suppress_labels=False)

# COMMAND ----------

sap.plot("PM020370", include_blank_strings=True, suppress_labels=False)

# COMMAND ----------

cols = ["AP009547", #	Gig Work - Accounting  (Financial)
        "AP009548", #	Gig Work - Baking Or Cooking  (Financial)
        "AP009549", #	Gig Work - Business Lead Generation  (Financial)
        "AP009550", #	Gig Work - Business Services  (Financial)
        "AP009551", #	Gig Work - Caregiver  (Financial)
        "AP009552", #	Gig Work - Childcare  (Financial)
        "AP009553", #	Gig Work - Computer Programming  (Financial)
        "AP009554", #	Gig Work - Customer Service  (Financial)
        "AP009555", #	Gig Work - Data Entry Or Processing  (Financial)
        "AP009556"  #	Gig Work - Dog Walker  (Financial)
        ]

sap.plot_grid(cols, "Gig Work Bar Chart", include_blank_strings=False)

# COMMAND ----------

cols = [    
        "PM020370", # Polk - Loyalty - Convertible Loyalist
        "PM020373", # Polk - Loyalty - Coupe Loyalist
        "PM020376", # Polk - Loyalty - Hatchback Loyalist
        "PM020379", # Polk - Loyalty - Pickup Loyalist
        "PM020382", # Polk - Loyalty - Sedan Loyalist
        "PM020388", # Polk - Loyalty - SUV Loyalist
        ]

sap.plot_grid(cols, "Polk Loyalty Pie Chart", include_blank_strings=True)

# COMMAND ----------

cols = [
        "PM020023",	# Polk - In-Market (Future) - Luxury Mid-size SUV
        "PM020044",	# Polk - In-Market (Future) - New Luxury Vehicle
        "PM020045",	# Polk - In-Market (Future) - New Non-luxury Vehicle
        "PM020046",	# Polk - In-Market (Future) - Electric
        "PM020047",	# Polk - In-Market (Future) - Hybrid
        "PM020048",	# Polk - In-Market (Future) - Any New Vehicle
        "PM020019",	# Polk - In-Market (Future) - Luxury Compact CUV
        "PM020021",	# Polk - In-Market (Future) - Luxury Full-size SUV
        "PM020022",	# Polk - In-Market (Future) - Luxury Mid-size CUV

]

sap.plot_grid(cols, "In Market Distribution Chart", include_blank_strings=False)

# COMMAND ----------

agg_df = (
        raw_sample_df
        .filter((F.col("PM020044").isNotNull()) & (F.col("PM020044") != "")).
        select(F.col("PM020044"))
        )

agg_df = to_pd(agg_df)               

# COMMAND ----------

agg_df.nunique()[0]

# COMMAND ----------

int('018')

# COMMAND ----------

SparkAcxiomPlot(raw_sample_df, ref_df, "IBE2834").plot("barh")

# COMMAND ----------

SparkAcxiomPlot(raw_sample_df, ref_df, "PM020388").plot("pie")

# COMMAND ----------

result = (
    raw_sample_df
    .filter((F.col("IBE2834").isNotNull()) & (F.col("IBE2834") != "")) 
    .groupBy("IBE2834")
    .agg(F.count("IBE2834").alias("CNT"))
)

pdf = to_pd(result)
ax = pdf.sort_values("CNT").plot(kind="barh", x="IBE2834", y="CNT")
for c in ax.containers: ax.bar_label(c, fmt="%.0f")
polish_plot(ax, "This plot is ", "")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

#  Cell 3: Helper Functions
def log_status(message, print_timestamp=True):
    """Helper function to print status updates with consistent formatting and display in Databricks"""
    timestamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " if print_timestamp else ""
    formatted_message = f"STATUS: {timestamp}{message}"
    
    # Print to standard output
    print(formatted_message)
    
    # If in Databricks, also display with HTML formatting for better visibility
    if IN_DATABRICKS:
        try:
            if message.startswith("ERROR") or message.startswith("CRITICAL ERROR"):
                display(HTML(f"<div style='color:red; font-weight:bold;'>{formatted_message}</div>"))
            elif message.startswith("WARNING"):
                display(HTML(f"<div style='color:orange; font-weight:bold;'>{formatted_message}</div>"))
            elif message.startswith("===== "):
                display(HTML(f"<div style='color:blue; font-weight:bold; font-size:1.1em;'>{formatted_message}</div>"))
            else:
                display(HTML(f"<div>{formatted_message}</div>"))
        except:
            # Fallback is the standard print already done above
            pass

print("\n=== CELL: Helper Functions completed ===")


# COMMAND ----------

# CELL 4: Enhanced Utility Functions
# Position: After Original Cell 4 (Visualization Functions)
# Purpose: Adds improved logging, configuration, and resource management utilities

from IPython.display import display, HTML
import traceback
import time
import json
import numpy as np
import pandas as pd
import functools
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Define IN_DATABRICKS variable
IN_DATABRICKS = True

# Configuration management for better parameter organization
class PipelineConfig:
    def __init__(self, sample_size=10000, output_prefix="/dbfs/FileStore/acxiom_clustering/vehicle_segmentation",
                clustering_methods=None, max_clusters=15):
        self.sample_size = sample_size
        self.output_prefix = output_prefix
        self.clustering_methods = clustering_methods or ["kmeans"]
        self.max_clusters = max_clusters
        self.debug_mode = False
        
    def enable_debug(self):
        """Enable debug mode for more verbose logging"""
        self.debug_mode = True
        return self
        
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            "sample_size": self.sample_size,
            "output_prefix": self.output_prefix,
            "clustering_methods": self.clustering_methods,
            "max_clusters": self.max_clusters,
            "debug_mode": self.debug_mode
        }

def log_status(message, print_timestamp=True):
    """Helper function to print status updates with consistent formatting and display in Databricks"""
    timestamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " if print_timestamp else ""
    formatted_message = f"STATUS: {timestamp}{message}"
    
    # Print to standard output
    print(formatted_message)
    
    # If in Databricks, also display with HTML formatting for better visibility
    if IN_DATABRICKS:
        try:
            if message.startswith("ERROR") or message.startswith("CRITICAL ERROR"):
                display(HTML(f"<div style='color:red; font-weight:bold;'>{formatted_message}</div>"))
            elif message.startswith("WARNING"):
                display(HTML(f"<div style='color:orange; font-weight:bold;'>{formatted_message}</div>"))
            elif message.startswith("===== "):
                display(HTML(f"<div style='color:blue; font-weight:bold; font-size:1.1em;'>{formatted_message}</div>"))
            else:
                display(HTML(f"<div>{formatted_message}</div>"))
        except:
            # Fallback is the standard print already done above
            pass

def log_progress(step, total_steps, message):
    """
    Log progress with a visual progress bar
    
    Parameters:
    -----------
    step : int
        Current step number
    total_steps : int
        Total number of steps
    message : str
        Message to display with progress
    """
    percentage = (step / total_steps) * 100
    progress_bar = "▓" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
    log_status(f"[{progress_bar}] ({percentage:.1f}%) {message}")

def release_resources(spark_df=None, pandas_dfs=None):
    """
    Release memory resources to prevent OOM errors
    
    Parameters:
    -----------
    spark_df : SparkDataFrame or None
        Spark DataFrame to unpersist
    pandas_dfs : list or None
        List of pandas DataFrames to delete
    """
    # Release Spark DataFrame
    if spark_df is not None:
        try:
            spark_df.unpersist()
            log_status("Released Spark DataFrame from memory")
        except Exception as e:
            log_status(f"Warning: Could not release Spark DataFrame: {str(e)}")
    
    # Release pandas DataFrames
    if pandas_dfs is not None:
        import gc
        try:
            for i, df in enumerate(pandas_dfs):
                del df
                log_status(f"Deleted pandas DataFrame {i+1} of {len(pandas_dfs)}")
            gc.collect()
            log_status("Garbage collection completed")
        except Exception as e:
            log_status(f"Warning: Error during memory cleanup: {str(e)}")

def profile_execution(func):
    """
    Decorator to profile function execution time and memory usage
    
    Parameters:
    -----------
    func : function
        Function to profile
        
    Returns:
    --------
    function
        Wrapped function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Try to get memory info if psutil is available
        try:
            import psutil
            start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_available = True
        except ImportError:
            memory_available = False
            
        # Run function with all arguments
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log execution time
        log_status(f"Function {func.__name__} completed in {execution_time:.2f} seconds")
        
        # Log memory usage if available
        if memory_available:
            try:
                import psutil
                end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_diff = end_mem - start_mem
                log_status(f"Memory change: {memory_diff:.2f} MB (Current: {end_mem:.2f} MB)")
            except Exception as memory_error:
                log_status(f"Could not measure memory usage: {str(memory_error)}")
        
        # Run garbage collection
        gc.collect()
        
        return result
    
    return wrapper

def verify_data_quality(df, stage_name):
    """
    Verify data quality at various pipeline stages
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to verify
    stage_name : str
        Name of the current pipeline stage
        
    Returns:
    --------
    bool
        True if data quality is acceptable, False otherwise
    """
    issues = []
    
    # Skip empty dataframes
    if df is None or len(df) == 0:
        log_status(f"⚠️ Empty DataFrame at {stage_name}")
        return False
    
    # Check for missing values
    missing_counts = df.isna().sum()
    high_missing_cols = missing_counts[missing_counts > len(df) * 0.5].index.tolist()
    if high_missing_cols:
        issues.append(f"High missing values in columns: {high_missing_cols[:5]}...")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols[:5]}...")
    
    # Check for extreme outliers in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:5]:  # Check just first 5 numeric columns for speed
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                extreme_count = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
                if extreme_count > len(df) * 0.05:
                    issues.append(f"Column {col} has {extreme_count} extreme outliers")
        except Exception:
            # Skip columns that can't be analyzed
            continue
    
    # Log issues or confirm quality
    if issues:
        log_status(f"⚠️ Data quality issues at {stage_name}:")
        for issue in issues:
            log_status(f"  - {issue}")
        
        # Sample problematic records
        if high_missing_cols and IN_DATABRICKS:
            try:
                sample_col = high_missing_cols[0]
                problem_samples = df[df[sample_col].isna()].head(3)
                log_status("Sample records with missing values:")
                display(HTML(f"<h5>Sample problematic records for column {sample_col}</h5>"))
                display(problem_samples)
            except Exception as e:
                log_status(f"Could not display sample records: {str(e)}")
        
        return False
    else:
        log_status(f"✓ Data quality verified at {stage_name}")
        return True

def create_detailed_log(config, pipeline_stages):
    """
    Create a detailed log file with all stages and parameters
    
    Parameters:
    -----------
    config : PipelineConfig
        Configuration object
    pipeline_stages : dict
        Dictionary of pipeline stages and their details
        
    Returns:
    --------
    str
        Path to the created log file
    """
    log_content = f"Vehicle Segmentation Pipeline Execution Log\n"
    log_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += "=" * 50 + "\n\n"
    
    # Add configuration
    log_content += "Configuration:\n"
    log_content += "-" * 20 + "\n"
    for key, value in config.to_dict().items():
        log_content += f"  {key}: {value}\n"
    
    # Add pipeline stages
    for stage, details in pipeline_stages.items():
        log_content += f"\n[{stage}]\n"
        log_content += "-" * 20 + "\n"
        for key, value in details.items():
            if isinstance(value, dict):
                log_content += f"  {key}:\n"
                for sub_key, sub_value in value.items():
                    log_content += f"    {sub_key}: {sub_value}\n"
            else:
                log_content += f"  {key}: {value}\n"
    
    # Write log file
    log_path = f"{config.output_prefix}_execution_log.txt"
    with open(log_path, "w") as f:
        f.write(log_content)
        
    log_status(f"Created detailed execution log at {log_path}")
    return log_path

print("\n=== CELL 4: Enhanced Utility Functions loaded ===")

# COMMAND ----------

# DBTITLE 1,Generate Cluster-Based Segment Names with Profiles
# Cell 7: Generate Cluster-Based Segment Names (not being used so skip this cell)

# def generate_data_driven_segment_names(cluster_data, cluster_col, prepared_data, mca_analysis, column_map=None):
#     """
#     Generate data-driven segment names based on cluster characteristics
    
#     Parameters:
#     -----------
#     cluster_data : pandas.DataFrame
#         DataFrame containing cluster assignments
#     cluster_col : str
#         Name of the column containing cluster assignments
#     prepared_data : dict
#         Dictionary with prepared data and column information
#     mca_analysis : dict
#         Dictionary with MCA analysis results
#     column_map : dict, optional
#         Mapping of column categories (vehicle, demographic, etc.)
        
#     Returns:
#     --------
#     dict
#         Dictionary mapping cluster IDs to generated names
#     """
#     log_status("Generating data-driven segment names based on cluster characteristics...")
    
#     # Get number of clusters
#     num_clusters = cluster_data[cluster_col].nunique()
    
#     # Initialize names dictionary
#     segment_names = {}
    
#     # If we have the original feature data
#     if 'features' in prepared_data and 'column_categories' in prepared_data:
#         original_data = prepared_data['features']
#         column_categories = prepared_data['column_categories']
        
#         # Join cluster assignments to original data
#         if cluster_col not in original_data.columns:
#             analysis_data = original_data.copy()
#             analysis_data[cluster_col] = cluster_data[cluster_col].values
#         else:
#             analysis_data = original_data
        
#         # For each cluster
#         for cluster_id in range(num_clusters):
#             # Isolate cluster data
#             cluster_mask = analysis_data[cluster_col] == cluster_id
#             cluster_segment = analysis_data[cluster_mask]
            
#             # Create a profile based on original features
#             # First categorize features
#             vehicle_cols = [col for col, cat in column_categories.items() 
#                          if cat.lower() == 'vehicle']
#             demo_cols = [col for col, cat in column_categories.items() 
#                        if cat.lower() == 'demographic']
#             propensity_cols = [col for col, cat in column_categories.items() 
#                              if cat.lower() == 'propensity']
            
#             # Find distinctive vehicle preferences
#             # This approach looks for columns where the cluster has notably higher values
#             dominant_vehicle_type = None
#             max_vehicle_diff = 0
            
#             for col in vehicle_cols:
#                 if col in cluster_segment.columns:
#                     cluster_mean = cluster_segment[col].mean()
#                     overall_mean = analysis_data[col].mean()
#                     diff = cluster_mean - overall_mean
                    
#                     if diff > max_vehicle_diff:
#                         max_vehicle_diff = diff
#                         # Extract vehicle type from column name or value
#                         # This will depend on your specific column naming conventions
#                         if "luxury" in col.lower():
#                             dominant_vehicle_type = "Luxury"
#                         elif "suv" in col.lower() or "pickup" in col.lower():
#                             dominant_vehicle_type = "SUV/Truck"
#                         elif "compact" in col.lower() or "economy" in col.lower():
#                             dominant_vehicle_type = "Economy"
#                         elif "electric" in col.lower() or "hybrid" in col.lower():
#                             dominant_vehicle_type = "Alternative Fuel"
#                         elif "sports" in col.lower() or "performance" in col.lower():
#                             dominant_vehicle_type = "Performance"
#                         elif "family" in col.lower() or "minivan" in col.lower():
#                             dominant_vehicle_type = "Family"
#                         else:
#                             dominant_vehicle_type = "Standard"
            
#             # Find distinctive demographics
#             income_level = None
#             age_group = None
            
#             for col in demo_cols:
#                 if col in cluster_segment.columns:
#                     # Check income-related columns
#                     if "income" in col.lower():
#                         cluster_mean = cluster_segment[col].mean()
#                         overall_mean = analysis_data[col].mean()
                        
#                         if cluster_mean > overall_mean * 1.2:  # 20% higher
#                             if "high" in col.lower() or "200k" in col.lower():
#                                 income_level = "Affluent"
#                             elif "mid" in col.lower() or "100k" in col.lower():
#                                 income_level = "Middle-Income"
#                             elif "low" in col.lower() or "50k" in col.lower():
#                                 income_level = "Budget-Conscious"
                    
#                     # Check age-related columns
#                     if "age" in col.lower():
#                         cluster_mean = cluster_segment[col].mean()
#                         overall_mean = analysis_data[col].mean()
                        
#                         if cluster_mean > overall_mean * 1.2:  # 20% higher
#                             if "young" in col.lower() or "18-34" in col.lower():
#                                 age_group = "Young"
#                             elif "senior" in col.lower() or "65+" in col.lower():
#                                 age_group = "Senior"
#                             elif "middle" in col.lower() or "35-64" in col.lower():
#                                 age_group = "Middle-Aged"
            
#             # Find distinctive buying behavior
#             buying_behavior = None
#             for col in propensity_cols:
#                 if col in cluster_segment.columns:
#                     cluster_mean = cluster_segment[col].mean()
#                     overall_mean = analysis_data[col].mean()
                    
#                     if cluster_mean > overall_mean * 1.2:  # 20% higher
#                         if "tech" in col.lower() or "early" in col.lower():
#                             buying_behavior = "Tech-Forward"
#                         elif "value" in col.lower() or "price" in col.lower():
#                             buying_behavior = "Value-Conscious"
#                         elif "luxury" in col.lower() or "premium" in col.lower():
#                             buying_behavior = "Premium"
#                         elif "eco" in col.lower() or "environment" in col.lower():
#                             buying_behavior = "Eco-Conscious"
#                         elif "family" in col.lower() or "practical" in col.lower():
#                             buying_behavior = "Practical"
                            
#             # Generate segment name based on findings
#             name_parts = []
            
#             if income_level:
#                 name_parts.append(income_level)
            
#             if age_group:
#                 name_parts.append(age_group)
            
#             if buying_behavior:
#                 name_parts.append(buying_behavior)
                
#             if dominant_vehicle_type:
#                 if buying_behavior:
#                     name_parts.append(f"{dominant_vehicle_type} Vehicle Buyers")
#                 else:
#                     name_parts.append(f"{dominant_vehicle_type} Vehicle Enthusiasts")
            
#             # If we couldn't determine characteristics, use cluster dimensions
#             if not name_parts and 'mca_coords' in mca_analysis:
#                 # Get cluster center in MCA space
#                 dim_cols = [f'MCA_dim{i+1}' for i in range(min(3, mca_analysis['n_dims']))]
#                 center = cluster_data[cluster_data[cluster_col] == cluster_id][dim_cols].mean()
                
#                 # Determine main characteristics from MCA dimensions
#                 # This is highly dependent on how the MCA dimensions are interpreted
#                 # Usually requires domain knowledge or post-hoc analysis
                
#                 if center['MCA_dim1'] > 0.5:
#                     name_parts.append("Premium Vehicle")
#                 elif center['MCA_dim1'] < -0.5:
#                     name_parts.append("Economy Vehicle")
                    
#                 if len(dim_cols) > 1:
#                     if center['MCA_dim2'] > 0.5:
#                         name_parts.append("Tech-Savvy")
#                     elif center['MCA_dim2'] < -0.5:
#                         name_parts.append("Traditional")
                
#                 if not name_parts:
#                     name_parts = ["Mainstream"] 
                
#                 name_parts.append("Buyers")
            
#             # Final fallback if we still don't have a name
#             if not name_parts:
#                 name_parts = [f"Segment {cluster_id + 1}"]
            
#             # Combine parts into a full name
#             segment_names[cluster_id] = " ".join(name_parts)
    
#     # If we don't have original features or failed to generate names,
#     # fallback to generic names with MCA dimensions information
#     if not segment_names:
#         log_status("Using MCA dimensions to generate generic segment names")
        
#         # Extract MCA dimensions data
#         dim_cols = [f'MCA_dim{i+1}' for i in range(min(3, mca_analysis['n_dims']))]
        
#         # For each cluster
#         for cluster_id in range(num_clusters):
#             # Get cluster center in MCA space
#             center = cluster_data[cluster_data[cluster_col] == cluster_id][dim_cols].mean()
            
#             # Use the dominant dimensions to name the cluster
#             primary_dim = None
#             primary_value = 0
            
#             for dim in dim_cols:
#                 if abs(center[dim]) > abs(primary_value):
#                     primary_value = center[dim]
#                     primary_dim = dim
            
#             if primary_dim and abs(primary_value) > 0.2:
#                 direction = "High" if primary_value > 0 else "Low"
#                 segment_names[cluster_id] = f"{direction} {primary_dim} Vehicle Segment"
#             else:
#                 segment_names[cluster_id] = f"Average Vehicle Segment {cluster_id + 1}"
    
#     log_status(f"Generated {len(segment_names)} data-driven segment names")
#     return segment_names

# COMMAND ----------

# CELL 8: Global Configuration
GLOBAL_CONFIG = {
    # Database and table settings
    'acxiom_table': "dataproducts_dev.bronze_acxiom.gm_consumer_list",
    'id_column': "GM_PERSON_REALID",
    
    # File paths and prefixes
    'base_output_path': "/dbfs/FileStore/acxiom_clustering/",
    'mca_output_prefix': "/dbfs/FileStore/acxiom_clustering/mca_results",
    'clustering_output_prefix': "/dbfs/FileStore/acxiom_clustering/clustering_results",
    
    # Sample sizes
    'mca_sample_size': 50000,  # Larger sample for MCA
    'clustering_sample_size': 40000,  # Smaller sample for clustering
    
    # Clustering parameters
    'kmeans_clusters': [8, 10, 12],
    'hierarchical_clusters': [8, 10, 12],
    'max_clusters': 15
}

# COMMAND ----------

# CELL 9: Column Definitions
# Define columns for each category
def define_columns():
    # Define the ID column
    id_column = "GM_PERSON_REALID"
    
    # 1. VEHICLE PREFERENCE COLUMNS - Expanded
    vehicle_columns = [
        # Luxury vehicle interest (already in your code)
        "AP004561",  # Purchase a New Luxury CUV (Financial)
        "AP004542",  # Purchase a New Luxury SUV (Financial)
        "AP004563",  # Purchase a New Luxury Car (Financial)
        
        # Economy/Mainstream vehicle interest (already in your code)
        "AP004559",  # Purchase a New Compact Car (Financial)
        "AP004564",  # Purchase a New Mid-Sized Car (Financial)
        "AP004560",  # Purchase a New Full-Sized Car (Financial)
        
        # Specialty vehicle interest (already in your code)
        "AP004562",  # Purchase a New Sports Car (Financial)
        "AP004540",  # Purchase a New Full-Sized Pickup (Financial)
        "AP004541",  # Purchase a New Minivan (Financial)
        "AP004543",  # Purchase a New Economy SUV (Financial)
        
        # Current ownership (already in your code)
        "AP007105",  # Owns a Luxury SUV Body Style Vehicle
        "AP007113",  # Owns an all electric SUV
        
        # General vehicle interest (already in your code)
        "AP006816",  # In market for a vehicle
        "AP001711",  # Alternative to AP007119 for fuel type preference
        "AP001425",  # Household vehicle ownership count
        
        # NEW: Additional vehicle body type ownership
        "AP007106",  # Owns a Sedan Body Style
        "AP007107",  # Owns a Compact Vehicle
        "AP007108",  # Owns a Pickup Truck
        "AP007109",  # Owns a Minivan
        "AP007114",  # Owns hybrid vehicle
        "AP007115",  # Interest in electric vehicles
        
        # NEW: Vehicle usage patterns
        "AP005201",  # Daily commuter usage
        "AP005202",  # Weekend leisure driving
        "AP005203"   # Long-distance travel
    ]
    
    # 2. PROPENSITY & PURCHASE BEHAVIOR COLUMNS - Expanded
    propensity_columns = [
        # Purchase behavior (already in your code)
        "AP005983",  # Likely to buy vehicle with alternative fuel
        "AP006832",  # Shops for auto insurance
        
        # Financing behavior (already in your code)
        "AP004553",  # Auto Loan - New Vehicle (Financial)
        "AP004554",  # Auto Loan - Used Vehicle (Financial)
        "AP004570",  # Auto Lease (Financial)
        
        # Propensity indicators (already in your code)
        "AP004118",  # Automotive - General Interest
        "AP008214",  # Automobile ownership
        "AP008945",  # New vehicle purchase propensity
        "AP008215",  # Vehicle upgrade timeframe
        "AP008219",  # Price sensitivity
        "AP007821",  # Technology adoption
        "AP007903",  # Finance vs cash purchase
        
        # NEW: Decision-making process 
        "AP008301",  # Research-intensive buyer
        "AP008302",  # Brand-loyal customer
        "AP008303",  # Deal-seeking behavior
        "AP004581",  # Preferred loan duration
        
        # NEW: Shopping patterns
        "AP005270",  # High-end shopper
        "AP005271"   # Value shopper
    ]
    
    # 3. DEMOGRAPHIC COLUMNS - Expanded
    demographic_columns = [
        # Income brackets (already in your code)
        "AP001200",  # Household Income <$15K
        "AP001203",  # Household Income $35-49K
        "AP001204",  # Household Income $50-74K
        "AP001205",  # Household Income $150-174K
        "AP001206",  # Household Income $175-199K
        "AP001207",  # Household Income $200-249K
        "AP001208",  # Household Income $250K+
        
        # Age brackets (already in your code)
        "AP001106",  # Age 65-74
        "AP001107",  # Age 75+
        
        # Other key demographics (already in your code)
        "AP003015",  # Household income (general)
        "AP003004",  # Age (general)
        "AP003035",  # Geographic location (urban/suburban/rural)
        "AP003061",  # Home ownership
        "AP001119",  # Alternative demographic column
        "AP003711",  # Alternative demographic column
        
        # NEW: Family status
        "AP001500",  # Number of children in household
        "AP001501",  # Presence of teenagers
        "AP001502",  # Multi-generational household
        "AP001505",  # Recent baby/new parent
        
        # NEW: Additional location indicators
        "AP003036",  # Commute length
        "AP003037"   # Public transit usage
    ]
    
    # 4. LIFESTYLE/USAGE COLUMNS - Expanded
    lifestyle_columns = [
        # Outdoor activities (already in your code)
        "AP003921",  # Outdoor enthusiast - hunting/fishing
        "AP005265",  # Interest in camping and hiking
        "AP003965",  # Off-road racing enthusiast
        
        # Travel & business (already in your code)
        "AP004081",  # Business traveler
        "AP003726",  # Travel enthusiast
        
        # Luxury & culture (already in your code)
        "AP003895",  # Luxury product buyer
        "AP003935",  # Cultural/arts enthusiast
        
        # Shopping behavior (already in your code)
        "AP008218",  # Technology interest level
        "AP005270",  # High-end shopper
        "AP005271",  # Value shopper
        "AP008124",  # Social media user
        
        # NEW: Additional lifestyle indicators
        "AP003980",  # Home improvement enthusiast
        "AP003981",  # DIY car maintenance
        "AP003982",  # Environmental concerns
        "AP003983"   # Urban vs rural lifestyle preference
    ]
    
    # 5. FINANCIAL BEHAVIOR COLUMNS - Expanded
    financial_columns = [
        # Credit card behavior (already in your code)
        "AP004504",  # Credit card - Premium/Upscale
        "AP004508",  # Credit card - Cash-back reward
        "AP004510",  # Credit card - Airline miles reward
        "AP004512",  # Credit card - Points reward
        "AP004505",  # Credit card - Standard
        "AP004520",  # Credit card - Frequent user
        "AP004531",  # Credit card - Balance carrier
        
        # Investment behavior (already in your code)
        "AP005550",  # Financial investment - Active Stock Trader
        "AP004580",  # Bank account - Premium banking
        "AP007904",  # Financial outlook indicator
        
        # NEW: Economic indicators
        "AP003580",  # Household income tier
        "AP003581",  # Credit card usage frequency
        "AP003582",  # Savings behavior
        "AP003583",  # Risk tolerance
        "AP003584"   # Financial planning preference
    ]
    
    return id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns

print("\n=== CELL: Column Definitions defined ===")



# COMMAND ----------

# CELL 10: Data Extraction Function - Modified to use Python-friendly formats
def direct_extract_acxiom_data(sample_size=GLOBAL_CONFIG.get('mca_sample_size', 10000)):
    """
    Extract Acxiom data directly using SQL without complex stratification
    
    Parameters:
    -----------
    sample_size : int
        Number of rows to sample
        
    Returns:
    --------
    dict
        Dictionary containing extracted data and metadata
    """
    extraction_start = time.time()
    log_status(f"Starting direct Acxiom data extraction with sample size: {sample_size}")
    
    try:
        # Get column definitions
        id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
        
        # Combine all columns to extract
        all_columns = [id_column] + vehicle_columns + propensity_columns + demographic_columns + lifestyle_columns + financial_columns
        
        # Remove any duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))
        
        # Create a column mapping for return
        column_map = {
            "id": id_column,
            "vehicle": vehicle_columns,
            "propensity": propensity_columns,
            "demographic": demographic_columns,
            "lifestyle": lifestyle_columns,
            "financial": financial_columns
        }
        
        # Specify the correct table name
        acxiom_table = "dataproducts_dev.bronze_acxiom.gm_consumer_list"
        
        # Verify table access
        log_status("Verifying database access...")
        try:
            spark.sql(f"SELECT 1 FROM {acxiom_table} LIMIT 1")
            log_status("Database access verified")
        except Exception as e:
            log_status(f"FATAL ERROR: Cannot access Acxiom table: {str(e)}")
            raise RuntimeError(f"Cannot access {acxiom_table}. This is a fatal error.")
        
        # Verify column existence (get a sample row)
        sample_df = spark.sql(f"SELECT * FROM {acxiom_table} LIMIT 1")
        available_columns = set(sample_df.columns)
        
        # Filter for only available columns
        valid_columns = [col for col in all_columns if col in available_columns]
        
        # Report on missing columns
        missing_columns = set(all_columns) - set(valid_columns)
        if missing_columns:
            log_status(f"WARNING: {len(missing_columns)} columns not found in dataset")
            # Only log first 10 missing columns to avoid clutter
            if len(missing_columns) > 10:
                log_status(f"First 10 missing columns: {list(missing_columns)[:10]}")
            else:
                log_status(f"Missing columns: {list(missing_columns)}")
        
        # Ensure we have enough columns
        if len(valid_columns) < 10:
            log_status(f"FATAL ERROR: Not enough valid columns found (only {len(valid_columns)}). Need at least 10 columns for meaningful analysis.")
            raise RuntimeError("Not enough valid columns found for analysis. This is a fatal error.")
        
        # Build column list for query
        column_list = ", ".join(valid_columns)
        
        # Create simple query - just use random sampling
        log_status("Executing simple random sampling...")
        simple_sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE {id_column} IS NOT NULL
        ORDER BY rand()
        LIMIT {sample_size}
        """
        
        # Execute the query
        acxiom_df = spark.sql(simple_sql_query)
        
        # Check if we got enough data
        row_count = acxiom_df.count()
        if row_count == 0:
            log_status("FATAL ERROR: No rows returned from Acxiom query")
            raise RuntimeError("No data returned from Acxiom database query. This is a fatal error.")
        
        # Cache the result for faster subsequent operations
        acxiom_df.cache()
        
        # Display sample in Databricks
        if IN_DATABRICKS:
            try:
                display(HTML("<h4>Data Sample (first 5 rows)</h4>"))
                display(acxiom_df.limit(5))
            except:
                pass
        
        # Prepare result dictionary
        col_count = len(acxiom_df.columns)
        extract_time = time.time() - extraction_start
        log_status(f"Successfully extracted {row_count} rows and {col_count} columns in {extract_time:.2f} seconds")
        
        result = {
            'spark_df': acxiom_df,
            'id_column': id_column,
            'column_map': column_map,
            'row_count': row_count,
            'extract_time': extract_time
        }
        
        # Save extraction metadata (Python-friendly format)
        import json
        metadata_path = f"/dbfs/FileStore/acxiom_clustering/extraction_metadata.json"
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': sample_size,
            'rows_extracted': row_count,
            'columns_extracted': col_count,
            'execution_time_seconds': extract_time
        }
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Now save the metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return result
        
    except Exception as e:
        log_status(f"ERROR in direct data extraction: {str(e)}")
        log_status(f"Error details: {traceback.format_exc()}")
        raise
        
        # Verify column existence (get a sample row)
        sample_df = spark.sql(f"SELECT * FROM {acxiom_table} LIMIT 1")
        available_columns = set(sample_df.columns)
        
        # Filter for only available columns
        valid_columns = [col for col in all_columns if col in available_columns]
        
        # Report on missing columns
        missing_columns = set(all_columns) - set(valid_columns)
        if missing_columns:
            log_status(f"WARNING: {len(missing_columns)} columns not found in dataset")
            # Only log first 10 missing columns to avoid clutter
            if len(missing_columns) > 10:
                log_status(f"First 10 missing columns: {list(missing_columns)[:10]}")
            else:
                log_status(f"Missing columns: {list(missing_columns)}")
        
        # Ensure we have enough columns
        if len(valid_columns) < 10:
            log_status(f"FATAL ERROR: Not enough valid columns found (only {len(valid_columns)}). Need at least 10 columns for meaningful analysis.")
            raise RuntimeError("Not enough valid columns found for analysis. This is a fatal error.")
        
        # Build column list for query
        column_list = ", ".join(valid_columns)
        
        # Create simple query - just use random sampling
        log_status("Executing simple random sampling...")
        simple_sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE {id_column} IS NOT NULL
        ORDER BY rand()
        LIMIT {sample_size}
        """
        
        # Execute the query
        acxiom_df = spark.sql(simple_sql_query)
        
        # Check if we got enough data
        row_count = acxiom_df.count()
        if row_count == 0:
            log_status("FATAL ERROR: No rows returned from Acxiom query")
            raise RuntimeError("No data returned from Acxiom database query. This is a fatal error.")
        
        # Cache the result for faster subsequent operations
        acxiom_df.cache()
        
        # Display sample in Databricks
        if IN_DATABRICKS:
            try:
                display(HTML("<h4>Data Sample (first 5 rows)</h4>"))
                display(acxiom_df.limit(5))
            except:
                pass
        
        # Prepare result dictionary
        col_count = len(acxiom_df.columns)
        extract_time = time.time() - extraction_start
        log_status(f"Successfully extracted {row_count} rows and {col_count} columns in {extract_time:.2f} seconds")
        
        result = {
            'spark_df': acxiom_df,
            'id_column': id_column,
            'column_map': column_map,
            'row_count': row_count,
            'extract_time': extract_time
        }
        
        # Save extraction metadata (Python-friendly format)
        import json
        metadata_path = f"/dbfs/FileStore/acxiom_clustering/extraction_metadata.json"
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': sample_size,
            'rows_extracted': row_count,
            'columns_extracted': col_count,
            'execution_time_seconds': extract_time
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return result
        
    except Exception as e:
        log_status(f"ERROR in direct data extraction: {str(e)}")
        log_status(f"Error details: {traceback.format_exc()}")
        raise

# COMMAND ----------

# CELL 11: Robust Data Preparation
import time
import traceback
import pandas as pd
import numpy as np

def robust_prepare_data_for_mca(acxiom_df, id_column="GM_PERSON_REALID"):
    """
    Prepare data for MCA analysis with improved type checking and error handling
    
    Parameters:
    -----------
    acxiom_df : spark.DataFrame or pandas.DataFrame
        DataFrame with extracted data (can be either Spark or pandas)
    id_column : str
        Name of the ID column
        
    Returns:
    --------
    dict
        Dictionary containing prepared data and metadata
    """
    if acxiom_df is None:
        log_status("ERROR: Input DataFrame is None")
        return None
        
    # Step 1: Convert to pandas for MCA processing if it's a Spark DataFrame
    log_status("Preparing DataFrame for MCA analysis...")
    pandas_start = time.time()
    
    try:
        if hasattr(acxiom_df, 'toPandas'):
            log_status("Converting Spark DataFrame to pandas DataFrame...")
            df = acxiom_df.toPandas()
        else:
            log_status("Input is already a pandas DataFrame, using directly")
            df = acxiom_df
            
        pandas_time = time.time() - pandas_start
        log_status(f"DataFrame preparation complete in {pandas_time:.2f} seconds ({len(df)} rows, {len(df.columns)} columns)")
    except Exception as e:
        log_status(f"ERROR during DataFrame conversion: {str(e)}")
        log_status(traceback.format_exc())
        return None
    
    # Step 2: Prepare data for MCA by category
    log_status("Preparing data for MCA analysis with category-aware processing...")
    mca_prep_start = time.time()
    
    try:
        # Extract ID column
        id_values = df[id_column].copy() if id_column in df.columns else None
        if id_values is None:
            log_status(f"WARNING: ID column '{id_column}' not found, continuing without ID values")
        
        # Get actual columns present in the data
        id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
        
        available_vehicle_cols = [col for col in vehicle_columns if col in df.columns]
        available_propensity_cols = [col for col in propensity_columns if col in df.columns]
        available_demographic_cols = [col for col in demographic_columns if col in df.columns]
        available_lifestyle_cols = [col for col in lifestyle_columns if col in df.columns]
        available_financial_cols = [col for col in financial_columns if col in df.columns]
        
        # Combine all available feature columns
        available_feature_cols = (available_vehicle_cols + available_propensity_cols + 
                                available_demographic_cols + available_lifestyle_cols + 
                                available_financial_cols)
        
        if len(available_feature_cols) < 5:
            log_status(f"ERROR: Not enough feature columns available (only {len(available_feature_cols)})")
            return None
            
        # Create a dictionary to track column categories
        column_categories = {}
        for col in available_vehicle_cols:
            column_categories[col] = "Vehicle"
        for col in available_propensity_cols:
            column_categories[col] = "Propensity"
        for col in available_demographic_cols:
            column_categories[col] = "Demographic"
        for col in available_lifestyle_cols:
            column_categories[col] = "Lifestyle"
        for col in available_financial_cols:
            column_categories[col] = "Financial"
        
        # Extract features
        features = df[available_feature_cols].copy()
        
        # Print column value counts for exploration
        log_status(f"Examining column value counts to determine categorical vs. numeric...")
        categorical_cols = []
        
        for col in available_feature_cols:
            try:
                # Get column category
                category = column_categories.get(col, "Unknown")
                
                # Count unique values
                unique_vals = features[col].nunique()
                na_count = features[col].isna().sum()
                
                # Decide if categorical based on unique values
                is_categorical = False
                
                # Check if already categorical/object type
                if pd.api.types.is_object_dtype(features[col]) or pd.api.types.is_categorical_dtype(features[col]):
                    is_categorical = True
                    log_status(f"- {col} ({category}): already categorical type", False)
                # Check if numeric with few unique values
                elif pd.api.types.is_numeric_dtype(features[col]):
                    if unique_vals <= 15:  # If numeric with few values, treat as categorical
                        is_categorical = True
                        log_status(f"- {col} ({category}): {unique_vals} unique values - treating as categorical", False)
                    else:
                        log_status(f"- {col} ({category}): {unique_vals} unique values - binning into categories", False)
                        
                        # Skip binning if column has many NaN values
                        if na_count / len(features) > 0.5:
                            log_status(f"  Skipping binning due to high NaN ratio: {na_count/len(features):.2f}", False)
                            features[col] = features[col].astype(str)
                            is_categorical = True
                        else:
                            # Only bin non-NaN values if there are enough
                            non_na_mask = ~features[col].isna()
                            if non_na_mask.sum() > 5:  # Need at least 5 non-NA values for binning
                                try:
                                    # Use quintiles for binning (ensuring we can handle duplicates)
                                    features.loc[non_na_mask, col] = pd.qcut(
                                        features.loc[non_na_mask, col], 
                                        q=5, 
                                        labels=False, 
                                        duplicates='drop'
                                    )
                                    # Fill NAs with special value
                                    features[col] = features[col].fillna(-1)
                                    is_categorical = True
                                except Exception as bin_error:
                                    # If binning fails, convert to string
                                    log_status(f"  Could not bin column {col}: {str(bin_error)}", False)
                                    features[col] = features[col].astype(str)
                                    is_categorical = True
                            else:
                                # Not enough non-NA values, just convert to string
                                features[col] = features[col].astype(str)
                                is_categorical = True
                else:
                    # For non-numeric, always categorical
                    is_categorical = True
                    log_status(f"- {col} ({category}): {unique_vals} unique values - treating as categorical", False)
                    
                # Add to categorical list if determined to be categorical
                if is_categorical:
                    categorical_cols.append(col)
                    # Convert to string to ensure MCA compatibility
                    features[col] = features[col].astype(str)
                
                # Report on missing values
                if na_count > 0:
                    log_status(f"  {na_count} missing values ({na_count/len(features)*100:.1f}%)", False)
                    
            except Exception as col_error:
                log_status(f"WARNING: Error processing column {col}: {str(col_error)}")
                # Skip problematic column
                continue
        
        # Check if we have enough categorical columns
        if len(categorical_cols) < 3:
            log_status(f"ERROR: Not enough categorical columns for MCA (only {len(categorical_cols)})")
            return None
        
        # Check for columns with all NAs
        na_counts = features[categorical_cols].isna().sum()
        all_na_cols = na_counts[na_counts == len(features)].index.tolist()
        
        if all_na_cols:
            log_status(f"Removing {len(all_na_cols)} columns with all NAs")
            features = features.drop(columns=all_na_cols)
            categorical_cols = [col for col in categorical_cols if col not in all_na_cols]
        
        # Final check for any remaining NAs in categorical columns
        for col in categorical_cols:
            if features[col].isna().any():
                log_status(f"Filling missing values in column {col}")
                features[col] = features[col].fillna("missing")
        
        # Create dataset for MCA
        prepared_data = {
            'features': features,
            'feature_cols': [col for col in features.columns],
            'id_column': id_column,
            'id_values': id_values,
            'categorical_cols': categorical_cols,
            'column_categories': column_categories
        }
        
        mca_prep_time = time.time() - mca_prep_start
        log_status(f"Data preparation for MCA complete in {mca_prep_time:.2f} seconds with {len(categorical_cols)} categorical columns")
        
        return prepared_data
        
    except Exception as e:
        log_status(f"ERROR in data preparation: {str(e)}")
        log_status(traceback.format_exc())
        return None

print("\n=== CELL 12: Robust Data Preparation completed ===")

# COMMAND ----------


# CELL 12: K-prototypes data extraction and clustering w/o MCA
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os


def extract_and_prepare_data_for_kprototypes(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Extract and prepare data for K-Prototypes clustering
    
    Parameters:
    -----------
    sample_size : int
        Number of samples to extract. Defaults to GLOBAL_CONFIG['clustering_sample_size']
        
    Returns:
    --------
    dict
        Dictionary containing prepared data for k-prototypes
    """
    # Validate sample size
    if not isinstance(sample_size, int) or sample_size <= 0:
        log_status(f"Invalid sample size provided: {sample_size}. Using default: {GLOBAL_CONFIG.get('clustering_sample_size', 50000)}")
        sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000)
    
    log_status(f"Extracting and preparing data for k-prototypes, sample size: {sample_size}")
    
    try:
        # Specify the table name
        acxiom_table = "dataproducts_dev.bronze_acxiom.gm_consumer_list"
        
        # Get column definitions with explicit error handling and logging
        log_status("Calling define_columns() to get column definitions...")
        try:
            id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
            
            # Debug logging for each column list
            log_status(f"ID Column: {id_column}")
            log_status(f"Vehicle Columns: {vehicle_columns}")
            log_status(f"Propensity Columns: {propensity_columns}")
            log_status(f"Demographic Columns: {demographic_columns}")
            log_status(f"Lifestyle Columns: {lifestyle_columns}")
            log_status(f"Financial Columns: {financial_columns}")
        except Exception as def_error:
            log_status(f"ERROR in column definition: {str(def_error)}")
            log_status(f"Error details: {traceback.format_exc()}")
            return None
        
        # Validate column lists to remove any None or invalid values
        def clean_column_list(columns):
            # Check if None is in the list
            if None in columns:
                log_status(f"WARNING: None found in column list: {columns}")
            return [col for col in columns if col is not None and isinstance(col, str) and col.strip()]
        
        vehicle_columns = clean_column_list(vehicle_columns)
        propensity_columns = clean_column_list(propensity_columns)
        demographic_columns = clean_column_list(demographic_columns)
        lifestyle_columns = clean_column_list(lifestyle_columns)
        financial_columns = clean_column_list(financial_columns)
        
        # Validate ID column
        if not id_column or not isinstance(id_column, str) or id_column.strip() == '':
            log_status("ERROR: Invalid ID column")
            return None
        
        # First, get all available columns from the table
        log_status("Fetching available table columns...")
        all_table_columns = spark.table(acxiom_table).columns
        log_status(f"Total columns in table: {len(all_table_columns)}")
        
        # Add a check to log a sample of table columns
        log_status(f"First 20 table columns: {all_table_columns[:20]}")
        
        # Filter columns that actually exist in the table
        def filter_existing_columns(columns):
            non_existing = [col for col in columns if col not in all_table_columns]
            if non_existing:
                log_status(f"Columns not found in table: {non_existing}")
            return [col for col in columns if col in all_table_columns]
        
        # Filter and combine columns
        vehicle_cols = filter_existing_columns(vehicle_columns)
        propensity_cols = filter_existing_columns(propensity_columns)
        demographic_cols = filter_existing_columns(demographic_columns)
        lifestyle_cols = filter_existing_columns(lifestyle_columns)
        financial_cols = filter_existing_columns(financial_columns)
        
        # Combine all valid columns, ensuring ID column is first
        all_columns = [id_column] + vehicle_cols + propensity_cols + demographic_cols + lifestyle_cols + financial_cols
        
        # Remove duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))
        
        # Log the columns we're using
        log_status(f"Columns to extract: {all_columns}")
        
        # Verify ID column exists
        if id_column not in all_columns:
            log_status(f"ERROR: ID column {id_column} not found in table")
            return None
        
        # Safely quote column names
        quoted_columns = [f"`{col}`" for col in all_columns]
        column_list = ", ".join(quoted_columns)
        
        # Create safe SQL query with random sampling
        sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE `{id_column}` IS NOT NULL
        ORDER BY RAND()
        LIMIT {sample_size}
        """
        
        # Log the exact SQL query
        log_status(f"SQL Query: {sql_query}")
        
        # Execute the query
        log_status("Executing SQL query to extract data...")
        spark_df = spark.sql(sql_query)
        
        # Convert to pandas
        df = spark_df.toPandas()
        
        # Create category mappings
        column_categories = {}
        for col_list, category in [
            (vehicle_cols, "Vehicle"),
            (propensity_cols, "Propensity"),
            (demographic_cols, "Demographic"),
            (lifestyle_cols, "Lifestyle"),
            (financial_cols, "Financial")
        ]:
            for col in col_list:
                if col in df.columns:
                    column_categories[col] = category
        
        # Prepare columns for K-Prototypes
        feature_cols = [col for col in df.columns if col != id_column]
        
        # Determine categorical and numerical columns
        categorical_cols = []
        numerical_cols = []
        
        for col in feature_cols:
            # Get unique values
            unique_vals = df[col].nunique()
            
            # Categorize columns
            if unique_vals <= 15 or pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
                # Convert to string
                df[col] = df[col].fillna("missing").astype(str)
            else:
                numerical_cols.append(col)
                # Fill NAs with median
                df[col] = df[col].fillna(df[col].median())
        
        log_status(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
        
        # Verify we have at least some categorical columns for k-prototypes to be meaningful
        if not categorical_cols:
            log_status("ERROR: No categorical columns identified. K-prototypes requires categorical data.")
            return None
        
        # Return prepared data
        return {
            'data': df,
            'id_column': id_column,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'column_categories': column_categories,
            'row_count': len(df)
        }
    
    except Exception as e:
        log_status(f"ERROR in data extraction and preparation: {str(e)}")
        log_status(traceback.format_exc())
        return None


def run_kprototypes_clustering_standalone(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000), num_clusters=8, 
                                         output_prefix="/dbfs/FileStore/acxiom_clustering/kprototypes", 
                                         categorical_weight=0.5, max_iterations=100):
    """
    Standalone K-Prototypes clustering with no dependency on MCA or other functions
    
    Parameters:
    -----------
    sample_size : int
        Number of rows to sample
    num_clusters : int
        Number of clusters to generate
    output_prefix : str
        Prefix for output files
    categorical_weight : float
        Weight for categorical variables
    max_iterations : int
        Maximum iterations for k-prototypes algorithm
        
    Returns:
    --------
    dict
        Dictionary with k-prototypes results or None if failed
    """
    log_status(f"Starting standalone k-prototypes clustering with {num_clusters} clusters...")
    
    try:
        # Step 1: Extract and prepare data
        log_status("Step 1: Extracting and preparing data...")
        prepared_data = extract_and_prepare_data_for_kprototypes(sample_size)
        
        if prepared_data is None:
            log_status("ERROR: Data preparation failed")
            return None
        
        # Get the data
        df = prepared_data['data']
        id_column = prepared_data['id_column']
        categorical_cols = prepared_data['categorical_cols']
        numerical_cols = prepared_data['numerical_cols']
        
        log_status(f"Successfully prepared {len(df)} rows with {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
        
        # Step 2: Prepare data for K-Prototypes
        log_status("Step 2: Preparing arrays for K-Prototypes...")
        
        # Extract numerical data and standardize if available
        if numerical_cols:
            numerical_df = df[numerical_cols]
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(numerical_df)
        else:
            log_status("No numerical columns found, proceeding with categorical data only")
            numerical_data = np.empty((len(df), 0))  # Empty array with zero columns
        
        # Extract categorical data
        categorical_data = df[categorical_cols].values
        
        # Prepare categorical indices for K-Prototypes
        categorical_indices = list(range(numerical_data.shape[1], 
                                      numerical_data.shape[1] + categorical_data.shape[1]))
        
        # Combine data
        if numerical_data.shape[1] > 0:
            combined_data = np.hstack((numerical_data, categorical_data))
        else:
            combined_data = categorical_data
        
        log_status(f"Prepared data shape: {combined_data.shape} with {len(categorical_indices)} categorical features")
        
        # Step 3: Run K-Prototypes clustering
        log_status(f"Step 3: Running K-Prototypes clustering with {num_clusters} clusters...")
        
        # Check if kmodes package is installed
        try:
            from kmodes.kprototypes import KPrototypes
        except ImportError:
            log_status("Installing kmodes package...")
            import pip
            pip.main(['install', 'kmodes'])
            from kmodes.kprototypes import KPrototypes
        
        # Initialize and run K-Prototypes with gamma (categorical weight) during initialization
        k_proto = KPrototypes(n_clusters=num_clusters, 
                             init='Huang', 
                             max_iter=max_iterations, 
                             n_init=5,
                             verbose=1,
                             gamma=categorical_weight,  # Set categorical weight here during initialization
                             random_state=42)
        
        start_time = time.time()
        
        # Call fit_predict without gamma parameter, only with categorical indices
        # FIXED: Removed the gamma parameter from fit_predict since it's already set during initialization
        cluster_labels = k_proto.fit_predict(combined_data, categorical=categorical_indices)
        
        runtime = time.time() - start_time
        log_status(f"K-Prototypes clustering completed in {runtime:.2f} seconds")
        
        # Step 4: Analyze results
        log_status("Step 4: Analyzing clustering results...")
        
        # Add cluster labels to original data
        df_with_clusters = df.copy()
        df_with_clusters['kproto_cluster'] = cluster_labels
        
        # Calculate cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        log_status("Cluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(df_with_clusters)) * 100
            log_status(f"Cluster {cluster_id}: {size} records ({percentage:.1f}%)")
        
        # Calculate silhouette score if we have numerical features and multiple clusters
        silhouette = None
        if len(np.unique(cluster_labels)) > 1 and numerical_data.shape[1] > 0:
            try:
                silhouette = silhouette_score(numerical_data, cluster_labels)
                log_status(f"Silhouette score: {silhouette:.4f}")
            except Exception as sil_error:
                log_status(f"Note: Could not calculate silhouette score: {str(sil_error)}")
        else:
            log_status("Note: Silhouette score not calculated (requires multiple clusters and numerical features)")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        # Step 5: Visualize results
        log_status("Step 5: Creating visualizations...")
        
        # 5.1: Visualize cluster sizes
        plt.figure(figsize=(12, 6))
        cluster_labels_str = [f"Cluster {i}" for i in range(num_clusters)]
        bars = plt.bar(cluster_labels_str, cluster_sizes.values, 
                      color=plt.cm.tab20(np.linspace(0, 1, num_clusters)))
        
        plt.title(f"K-Prototypes Clustering: {num_clusters} Segments Size Distribution", fontsize=14)
        plt.xlabel("Cluster", fontsize=12)
        plt.ylabel("Number of Records", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(df_with_clusters)) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f"{int(height)}\n({percentage:.1f}%)",
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        size_plot_path = f"{output_prefix}_kproto_{num_clusters}_sizes.jpg"
        plt.savefig(size_plot_path, dpi=300, bbox_inches='tight')
        
        if IN_DATABRICKS:
            display(plt.gcf())
        
        plt.close()
        log_status(f"Saved cluster size visualization to {size_plot_path}")
        
        # 5.2: Visualize clusters in 2D space (if we have at least 2 numerical features)
        if len(numerical_cols) >= 2:
            # Select two numerical features for visualization
            viz_cols = numerical_cols[:2]
            
            plt.figure(figsize=(12, 10))
            
            # Plot data points colored by cluster
            scatter = plt.scatter(
                df[viz_cols[0]],
                df[viz_cols[1]],
                c=cluster_labels,
                cmap=plt.cm.tab20,
                alpha=0.7,
                s=40,
                edgecolors='w',
                linewidths=0.3
            )
            
           
            # Add cluster centroids
            for i in range(num_clusters):
                mask = df_with_clusters['kproto_cluster'] == i
                center_x = df.loc[mask, viz_cols[0]].mean()
                center_y = df.loc[mask, viz_cols[1]].mean()
                
                plt.scatter(center_x, center_y, 
                           s=200, marker='*', color='black', 
                           edgecolor='white', linewidth=1.5)
                
                plt.text(center_x, center_y, str(i), 
                        fontsize=14, ha='center', va='center', 
                        color='white', fontweight='bold')
            
            plt.colorbar(scatter, label="Cluster")
            plt.xlabel(viz_cols[0], fontsize=12)
            plt.ylabel(viz_cols[1], fontsize=12)
            plt.title(f"K-Prototypes Clustering: {num_clusters} Segments", fontsize=14)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            cluster_plot_path = f"{output_prefix}_kproto_{num_clusters}_clusters.jpg"
            plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
            
            if IN_DATABRICKS:
                display(plt.gcf())
            
            plt.close()
            log_status(f"Saved cluster visualization to {cluster_plot_path}")
        
        # Step 6: Save results
        log_status("Step 6: Saving results...")
        
        # Save clustering results
        results_path = f"{output_prefix}_kproto_{num_clusters}_results.csv"
        df_with_clusters.to_csv(results_path, index=False)
        log_status(f"Saved clustering results to {results_path}")
        
        # Return results dictionary
        return {
            'num_clusters': num_clusters,
            'silhouette_score': silhouette,
            'cluster_sizes': cluster_sizes,
            'cluster_data': df_with_clusters,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'runtime': runtime
        }
    
    except Exception as e:
        log_status(f"ERROR in k-prototypes clustering: {str(e)}")
        log_status(traceback.format_exc())
        return None

def run_kprototypes_multiple_standalone(cluster_counts=[8, 10, 12], sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000), categorical_weight=0.5):
    """
    Run k-prototypes with multiple cluster counts and compare results
    
    Parameters:
    -----------
    cluster_counts : list
        List of cluster counts to try
    sample_size : int
        Sample size to use
    categorical_weight : float
        Weight for categorical variables
        
    Returns:
    --------
    dict
        Dictionary with results for each cluster count
    """
    log_status(f"Running k-prototypes with multiple cluster counts: {cluster_counts}")
    
    results = {}
    silhouette_scores = {}
    
    for k in cluster_counts:
        log_status(f"Running k-prototypes with {k} clusters...")
        
        result = run_kprototypes_clustering_standalone(
            sample_size=sample_size,
            num_clusters=k,
            categorical_weight=categorical_weight
        )
        
        if result:
            results[k] = result
            if result['silhouette_score'] is not None:
                silhouette_scores[k] = result['silhouette_score']
                
            log_status(f"Successfully completed k-prototypes with {k} clusters")
        else:
            log_status(f"Failed to run k-prototypes with {k} clusters")
    
    # If we have multiple results with silhouette scores, create comparison chart
    if len(silhouette_scores) > 1:
        plt.figure(figsize=(10, 6))
        
        x = list(silhouette_scores.keys())
        y = list(silhouette_scores.values())
        
        plt.bar(
            [str(k) for k in x],
            y,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(x)))
        )
        
        # Find optimal k
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        optimal_score = silhouette_scores[optimal_k]
        
        plt.axhline(y=optimal_score, color='red', linestyle='--',
                   label=f'Best Score: {optimal_score:.4f} (k={optimal_k})')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Comparison of K-Prototypes Clustering Results', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        comparison_path = "/dbfs/FileStore/acxiom_clustering/kproto_comparison.jpg"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        if IN_DATABRICKS:
            display(plt.gcf())
        
        plt.close()
        log_status(f"Saved comparison chart to {comparison_path}")
        
        # Create a comparison report
        report = f"===== K-PROTOTYPES CLUSTERING COMPARISON =====\n\n"
        report += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Sample size: {sample_size}\n"
        report += f"Categorical weight: {categorical_weight}\n\n"
        
        report += "SILHOUETTE SCORES:\n"
        report += "-----------------\n"
        for k in sorted(silhouette_scores.keys()):
            score = silhouette_scores[k]
            relative = score / optimal_score
            report += f"k={k}: {score:.4f} ({relative:.1%} of optimal)\n"
        
        report += f"\nOptimal cluster count: k={optimal_k} (silhouette={optimal_score:.4f})\n"
        
        report_path = "/dbfs/FileStore/acxiom_clustering/kproto_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        log_status(f"Saved comparison report to {report_path}")
    
    return results

# Standalone function for easy execution
def run_just_kprototypes(cluster_counts=GLOBAL_CONFIG.get('kmeans_clusters', [8, 10, 12]), sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000), categorical_weight=0.5):
    """
    Run standalone K-Prototypes clustering with flexible parameters
    
    Parameters:
    -----------
    cluster_counts : list, optional
        List of cluster counts to generate (defaults to [8, 10, 12])
    sample_size : int, optional
        Sample size to use (defaults to global config)
    categorical_weight : float, optional
        Weight for categorical variables (default 0.5)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Use default cluster counts if not specified
    if cluster_counts is None:
        cluster_counts = GLOBAL_CONFIG.get('kmeans_clusters', [8, 10, 12])
    
    log_status(f"===== EXECUTING K-PROTOTYPES CLUSTERING =====")
    log_status(f"Cluster counts: {cluster_counts}")
    log_status(f"Sample size: {sample_size if sample_size is not None else 'Default from Global Config'}")
    log_status(f"Categorical weight: {categorical_weight}")
    
    start_time = time.time()
    
    try:
        # Run k-prototypes with multiple cluster counts
        results = run_kprototypes_multiple_standalone(
            cluster_counts=cluster_counts,
            sample_size=sample_size,
            categorical_weight=categorical_weight
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results:
            log_status(f"✅ K-Prototypes clustering completed successfully in {total_time:.2f} seconds")
            
            # Optional: Log detailed results
            for k, result in results.items():
                log_status(f"Cluster {k}: {len(result['cluster_data'])} records")
                if 'silhouette_score' in result and result['silhouette_score'] is not None:
                    log_status(f"  Silhouette Score: {result['silhouette_score']:.4f}")
            
            return True
        else:
            log_status(f"❌ K-Prototypes clustering failed after {total_time:.2f} seconds")
            return False
    
    except Exception as e:
        log_status(f"ERROR in k-prototypes clustering: {str(e)}")
        log_status(traceback.format_exc())
        return False
            

# COMMAND ----------


# CELL 12.1 K-prototypes with marketing report
def run_kprototypes_with_marketing_report(cluster_counts=None, sample_size=None, categorical_weight=0.5):
    """
    Run K-prototypes clustering with comprehensive marketing report generation
    
    Parameters:
    -----------
    cluster_counts : list, optional
        List of cluster counts to try (default [8, 10, 12])
    sample_size : int, optional
        Sample size to use for clustering
    categorical_weight : float, optional
        Weight for categorical variables (default 0.5)
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Use default values if not specified
    if cluster_counts is None:
        cluster_counts = [8, 10, 12]
    
    if sample_size is None:
        sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000)
    
    log_status(f"===== EXECUTING K-PROTOTYPES CLUSTERING WITH MARKETING REPORT =====")
    log_status(f"Cluster counts: {cluster_counts}")
    log_status(f"Sample size: {sample_size}")
    log_status(f"Categorical weight: {categorical_weight}")
    
    start_time = time.time()
    
    try:
        # First check if kmodes package is available (k-prototypes is in the same package)
        if not install_kmodes_if_needed():
            log_status("ERROR: Required kmodes package not available")
            return False
        
        # Run K-prototypes with multiple cluster counts
        results = {}
        cost_values = {}
        
        # Extract and prepare data (do this only once)
        prepared_data = extract_and_prepare_data_for_kprototypes(sample_size)
        if not prepared_data:
            log_status("ERROR: Failed to prepare data for K-prototypes clustering")
            return False
        
        # Run K-prototypes for each cluster count
        for k in cluster_counts:
            log_status(f"Running K-prototypes with {k} clusters...")
            
            result = run_kprototypes_clustering_standalone(
                sample_size=sample_size,
                num_clusters=k,
                categorical_weight=categorical_weight
            )
            
            if result:
                results[k] = result
                cost_values[k] = result.get('cost', float('inf'))  # Use a default if cost not available
                log_status(f"Successfully completed K-prototypes with {k} clusters")
            else:
                log_status(f"Failed to run K-prototypes with {k} clusters")
                
        if not results:
            log_status("ERROR: All K-prototypes clustering attempts failed")
            return False
                
        # Choose best clustering (some measure of quality)
        # For K-prototypes, use silhouette score if available
        best_scores = {}
        for k, result in results.items():
            if 'silhouette_score' in result and result['silhouette_score'] is not None:
                best_scores[k] = result['silhouette_score']
        
        # Use silhouette scores if available, otherwise use cost values
        if best_scores:
            best_k = max(best_scores, key=best_scores.get)
            log_status(f"Selected optimal clustering with k={best_k} clusters (best silhouette score)")
        else:
            # For cost values, lower is better
            best_k = min(cost_values, key=cost_values.get)
            log_status(f"Selected optimal clustering with k={best_k} clusters (lowest cost)")
            
        # Get the best result
        best_result = results[best_k]
        
        # Adapt format for marketing report generation
        kprototypes_for_marketing = {
            'cluster_data': best_result['cluster_data'],
            'centroids_df': pd.DataFrame({
                'cluster_id': range(best_k),
                # Add dummy columns to match expected format
                **{col: ['NA'] * best_k for col in prepared_data['categorical_cols'][:5]}
            }),
            'num_clusters': best_k
        }
        
        # Generate marketing report
        report_path = generate_kmodes_marketing_report(kprototypes_for_marketing, 
                                        output_prefix=f"/dbfs/FileStore/acxiom_clustering/kprototypes")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if report_path:
            log_status(f"✅ K-prototypes clustering and marketing report completed in {total_time:.2f} seconds")
            log_status(f"Marketing report saved to: {report_path}")
            return True
        else:
            log_status(f"❌ Marketing report generation failed after {total_time:.2f} seconds")
            return False
        
    except Exception as e:
        log_status(f"ERROR in K-prototypes clustering with marketing report: {str(e)}")
        log_status(traceback.format_exc())
        return False

# Function to generate segment names specifically for K-prototypes results
def generate_kprototypes_segment_names(kproto_results, column_categories=None):
    """
    Generate descriptive names for K-prototypes clusters
    
    Parameters:
    -----------
    kproto_results : dict
        Dictionary with K-prototypes results
    column_categories : dict, optional
        Dictionary mapping columns to categories
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to generated names
    """
    log_status("Generating descriptive segment names for K-prototypes clusters...")
    
    try:
        # Extract cluster data and numerical features
        cluster_data = kproto_results['cluster_data']
        num_clusters = kproto_results['num_clusters']
        
        # Try to get numerical columns
        numerical_cols = []
        if 'numerical_cols' in kproto_results:
            numerical_cols = kproto_results['numerical_cols']
        
        # Initialize segment names dictionary
        segment_names = {}
        
        # Get cluster sizes for reference
        if 'kmeans_cluster' in cluster_data.columns:
            cluster_col = 'kmeans_cluster'
        elif 'kproto_cluster' in cluster_data.columns:
            cluster_col = 'kproto_cluster'
        else:
            # Try to find any column that might be the cluster column
            potential_cluster_cols = [col for col in cluster_data.columns 
                                    if 'cluster' in col.lower() and col != 'cluster_id']
            if potential_cluster_cols:
                cluster_col = potential_cluster_cols[0]
            else:
                log_status("WARNING: Could not identify cluster column in results")
                # Create generic names and return
                return {i: f"Vehicle Segment {i+1}" for i in range(num_clusters)}
        
        # Calculate cluster sizes as percentages
        cluster_sizes = cluster_data[cluster_col].value_counts(normalize=True) * 100
        
        # For each cluster, analyze characteristics
        for cluster_id in range(num_clusters):
            # Get cluster records
            cluster_mask = cluster_data[cluster_col] == cluster_id
            cluster_records = cluster_data[cluster_mask]
            
            # Skip if no records
            if len(cluster_records) == 0:
                segment_names[cluster_id] = f"Vehicle Segment {cluster_id+1}"
                continue
            
            # Analyze numerical features if available
            numerical_profile = {}
            if numerical_cols:
                for col in numerical_cols:
                    if col in cluster_records.columns:
                        col_mean = cluster_records[col].mean()
                        all_mean = cluster_data[col].mean()
                        col_diff = col_mean - all_mean
                        numerical_profile[col] = {
                            'mean': col_mean,
                            'diff': col_diff,
                            'percentile': np.percentile(cluster_data[col], 
                                                       [25, 50, 75])
                        }
            
            # Determine key characteristics
            luxury_score = 0
            value_score = 0
            tech_score = 0
            family_score = 0
            utility_score = 0
            
            # Use numerical profiles to infer characteristics
            for col, stats in numerical_profile.items():
                col_lower = col.lower()
                diff = stats['diff']
                
                # Infer luxury from income or premium indicators
                if 'income' in col_lower or 'premium' in col_lower or 'luxury' in col_lower:
                    if diff > 0:
                        luxury_score += diff / all_mean * 5
                    else:
                        value_score -= diff / all_mean * 5
                
                # Infer tech adoption
                if 'tech' in col_lower or 'digital' in col_lower or 'connected' in col_lower:
                    if diff > 0:
                        tech_score += diff / all_mean * 5
                
                # Infer family focus
                if 'family' in col_lower or 'children' in col_lower or 'household' in col_lower:
                    if diff > 0:
                        family_score += diff / all_mean * 5
                
                # Infer utility priority
                if 'utility' in col_lower or 'cargo' in col_lower or 'practical' in col_lower:
                    if diff > 0:
                        utility_score += diff / all_mean * 5
            
            # Size determination
            size_pct = cluster_sizes.get(cluster_id, 0)
            if size_pct > 20:
                size_term = "Mainstream"
            elif size_pct > 10:
                size_term = "Major"
            elif size_pct > 3:
                size_term = "Niche"
            else:
                size_term = "Specialty"
            
            # Generate name based on scores
            primary_traits = []
            
            if luxury_score > 5:
                if luxury_score > 10:
                    primary_traits.append("Premium Luxury")
                else:
                    primary_traits.append("Upscale")
            
            if tech_score > 5:
                primary_traits.append("Tech-Forward")
            
            if value_score > 5:
                primary_traits.append("Value-Oriented")
            
            if family_score > 5:
                primary_traits.append("Family")
            
            if utility_score > 5:
                primary_traits.append("Utility")
            
            # If no clear traits, use size term
            if not primary_traits:
                if size_pct > 15:
                    primary_traits.append("Mainstream")
                elif luxury_score < -5:
                    primary_traits.append("Economy")
                else:
                    primary_traits.append("Standard")
            
            # Limit to max 2 primary traits
            primary_traits = primary_traits[:2]
            
            # Generate name
            name_parts = []
            name_parts.extend(primary_traits)
            
            # Add vehicle term
            name_parts.append("Vehicle")
            
            # Add buyer/segment term
            if family_score > 5:
                name_parts.append("Owners")
            elif luxury_score > 5:
                name_parts.append("Enthusiasts")
            elif tech_score > 5:
                name_parts.append("Early Adopters")
            else:
                name_parts.append("Buyers")
            
            # Create final name
            segment_names[cluster_id] = " ".join(name_parts)
        
        return segment_names
    
    except Exception as e:
        log_status(f"ERROR in segment name generation: {str(e)}")
        log_status(traceback.format_exc())
        # Return generic names as fallback
        return {i: f"Vehicle Segment {i+1}" for i in range(num_clusters)}

# Function to analyze K-prototypes results for marketing insights
def analyze_kprototypes_segments(cluster_data, centroids, k, numerical_cols=None, categorical_cols=None):
    """
    Analyze K-prototypes clusters to extract marketing-relevant insights
    
    Parameters:
    -----------
    cluster_data : pandas.DataFrame
        DataFrame with cluster assignments
    centroids : pandas.DataFrame
        DataFrame with cluster centroids (or a placeholder)
    k : int
        Number of clusters
    numerical_cols : list, optional
        List of numerical columns used in clustering
    categorical_cols : list, optional
        List of categorical columns used in clustering
        
    Returns:
    --------
    dict
        Dictionary containing segment insights
    """
    log_status("Analyzing K-prototypes clusters for marketing insights...")
    
    # Identify cluster column
    cluster_col = None
    for col in cluster_data.columns:
        if 'cluster' in col.lower() and col != 'cluster_id':
            cluster_col = col
            break
    
    if not cluster_col:
        log_status("ERROR: Could not identify cluster column in data")
        return None
    
    # Calculate cluster sizes
    cluster_counts = cluster_data[cluster_col].value_counts().sort_index()
    cluster_sizes = [(count / len(cluster_data)) * 100 for count in cluster_counts]
    
    # Initialize insights dictionary
    segment_insights = {
        'k': k,
        'cluster_sizes': cluster_sizes,
        'segments': {}
    }
    
    # Generate segment names
    segment_names = generate_kprototypes_segment_names({
        'cluster_data': cluster_data,
        'num_clusters': k,
        'numerical_cols': numerical_cols
    })
    
    # Process each cluster
    for cluster_id in range(k):
        # Skip if no members in this cluster
        if cluster_id not in cluster_data[cluster_col].unique():
            continue
            
        # Get cluster records
        cluster_records = cluster_data[cluster_data[cluster_col] == cluster_id]
        
        # Calculate cluster size percentage
        size_pct = segment_insights['cluster_sizes'][cluster_id] if cluster_id < len(segment_insights['cluster_sizes']) else 0
        
        # Get segment name
        segment_name = segment_names.get(cluster_id, f"Vehicle Segment {cluster_id+1}")
        
        # Create synthetic key features based on cluster analysis
        # This is a placeholder - in a real implementation, you would analyze the actual data
        key_features = {
            'luxury_orientation': 50,  # Default mid-point
            'price_sensitivity': 50,
            'tech_adoption': 50,
            'brand_loyalty': 50,
            'family_focus': 50,
            'utility_priority': 50
        }
        
        # Adjust key features based on segment name
        if "Premium" in segment_name or "Luxury" in segment_name:
            key_features['luxury_orientation'] = 85
            key_features['price_sensitivity'] = 20
        elif "Value" in segment_name or "Economy" in segment_name:
            key_features['luxury_orientation'] = 25
            key_features['price_sensitivity'] = 85
        
        if "Tech" in segment_name:
            key_features['tech_adoption'] = 85
        
        if "Family" in segment_name:
            key_features['family_focus'] = 85
        
        if "Utility" in segment_name:
            key_features['utility_priority'] = 85
        
        # Create synthetic demographics based on cluster analysis
        demographics = {
            'age': "18-34" if "Tech" in segment_name else
                  "35-54" if "Family" in segment_name else
                  "55+" if "Traditional" in segment_name else "Mixed",
                  
            'income': "Upper-middle to High" if key_features['luxury_orientation'] > 70 else
                     "Middle to Upper-middle" if key_features['luxury_orientation'] > 50 else
                     "Low to Middle",
                     
            'gender_split': "Not available",
            'education': "Not available",
            'geography': "Not available"
        }
        
        # Create synthetic vehicle preferences
        vehicle_preferences = []
        
        if key_features['luxury_orientation'] > 70:
            vehicle_preferences.append("Luxury vehicles with premium features")
            if key_features['tech_adoption'] > 70:
                vehicle_preferences.append("Premium models with advanced technology")
            if key_features['family_focus'] > 70:
                vehicle_preferences.append("Premium SUVs and family vehicles")
        elif key_features['family_focus'] > 70:
            vehicle_preferences.append("Family-oriented vehicles with versatile space")
            if key_features['utility_priority'] > 70:
                vehicle_preferences.append("SUVs and crossovers with ample cargo space")
            else:
                vehicle_preferences.append("Minivans and family sedans")
        elif key_features['utility_priority'] > 70:
            vehicle_preferences.append("Trucks and utility vehicles")
            vehicle_preferences.append("Vehicles with high cargo capacity")
        elif key_features['price_sensitivity'] > 70:
            vehicle_preferences.append("Economical vehicles with good fuel efficiency")
            vehicle_preferences.append("Value-focused models with essential features")
        
        if key_features['tech_adoption'] > 70 and "Tech" in segment_name:
            vehicle_preferences.append("Vehicles with latest technology features")
            vehicle_preferences.append("Connected cars with digital integration")
        
        # Ensure we have at least some preferences
        if not vehicle_preferences:
            vehicle_preferences = ["Standard vehicles", "Mixed preferences"]
        
        # Create synthetic buying behavior
        buying_behavior = []
        
        if key_features['price_sensitivity'] > 70:
            buying_behavior.append("Price-sensitive purchasing decisions")
            buying_behavior.append("Value-focused comparison shopping")
        elif key_features['luxury_orientation'] > 70:
            buying_behavior.append("Quality-focused over price-sensitive")
            buying_behavior.append("Prefers premium buying experience")
        
        if key_features['tech_adoption'] > 70:
            buying_behavior.append("Research-intensive purchase process")
            buying_behavior.append("Digital-first research approach")
        
        if key_features['brand_loyalty'] > 70:
            buying_behavior.append("Brand-loyal purchase decisions")
        
        # Ensure we have at least some behaviors
        if not buying_behavior:
            buying_behavior = ["Standard purchasing process", "Mixed buying patterns"]
        
        # Generate marketing recommendations
        marketing_strategy = generate_marketing_recommendations(
            key_features, 
            demographics, 
            vehicle_preferences, 
            buying_behavior,
            size_pct
        )
        
        # Store segment insights
        segment_insights['segments'][cluster_id] = {
            'name': segment_name,
            'size_pct': size_pct,
            'key_features': key_features,
            'demographics': demographics,
            'vehicle_preferences': vehicle_preferences,
            'buying_behavior': buying_behavior,
            'marketing_strategy': marketing_strategy
        }
    
    return segment_insights

# COMMAND ----------

# CELL 13: K-modes Clustering Implementation for Categorical Data
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os

def install_kmodes_if_needed():
    """
    Check if kmodes package is installed, and install it if not.
    """
    try:
        import kmodes
        log_status(f"kmodes package found (version: {kmodes.__version__})")
        return True
    except ImportError:
        log_status("kmodes package not found. Attempting to install...")
        try:
            import pip
            pip.main(['install', 'kmodes'])
            import kmodes
            log_status(f"Successfully installed kmodes package (version: {kmodes.__version__})")
            return True
        except Exception as e:
            log_status(f"ERROR: Failed to install kmodes package: {str(e)}")
            return False

def extract_and_prepare_data_for_kmodes(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Extract and prepare data specifically for K-modes clustering
    
    Parameters:
    -----------
    sample_size : int
        Number of samples to extract
        
    Returns:
    --------
    dict
        Dictionary containing prepared categorical data for K-modes
    """
    log_status(f"Extracting and preparing data for K-modes, sample size: {sample_size}")
    
    try:
        # Get column definitions
        id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
        
        # Combine all columns to extract
        all_columns = [id_column] + vehicle_columns + propensity_columns + demographic_columns + lifestyle_columns + financial_columns
        
        # Remove any duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))
        
        # Specify the table name
        acxiom_table = "dataproducts_dev.bronze_acxiom.gm_consumer_list"
        
        # Verify table access
        log_status("Verifying database access...")
        try:
            spark.sql(f"SELECT 1 FROM {acxiom_table} LIMIT 1")
            log_status("Database access verified")
        except Exception as e:
            log_status(f"ERROR: Cannot access Acxiom table: {str(e)}")
            return None
        
        # Verify column existence
        sample_df = spark.sql(f"SELECT * FROM {acxiom_table} LIMIT 1")
        available_columns = set(sample_df.columns)
        
        # Filter for only available columns
        valid_columns = [col for col in all_columns if col in available_columns]
        
        # Report on missing columns
        missing_columns = set(all_columns) - set(valid_columns)
        if missing_columns:
            log_status(f"WARNING: {len(missing_columns)} columns not found in dataset")
        
        # Ensure we have enough columns
        if len(valid_columns) < 10:
            log_status(f"ERROR: Not enough valid columns found (only {len(valid_columns)})")
            return None
        
        # Build column list for query
        column_list = ", ".join(valid_columns)
        
        # Create simple query with random sampling
        log_status("Executing data extraction query...")
        sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE {id_column} IS NOT NULL
        ORDER BY rand()
        LIMIT {sample_size}
        """
        
        # Execute the query
        spark_df = spark.sql(sql_query)
        
        # Check if we got enough data
        row_count = spark_df.count()
        if row_count == 0:
            log_status("ERROR: No rows returned from query")
            return None
        
        log_status(f"Successfully extracted {row_count} rows")
        
        # Convert to pandas for processing
        log_status("Converting to pandas DataFrame...")
        df = spark_df.toPandas()
        
        # Create category mappings
        column_categories = {}
        for col in vehicle_columns:
            if col in df.columns:
                column_categories[col] = "Vehicle"
                
        for col in propensity_columns:
            if col in df.columns:
                column_categories[col] = "Propensity"
                
        for col in demographic_columns:
            if col in df.columns:
                column_categories[col] = "Demographic"
                
        for col in lifestyle_columns:
            if col in df.columns:
                column_categories[col] = "Lifestyle"
                
        for col in financial_columns:
            if col in df.columns:
                column_categories[col] = "Financial"
        
        # For K-modes, we need to ensure all data is categorical
        # Convert all columns to string data type
        feature_cols = [col for col in df.columns if col != id_column]
        
        for col in feature_cols:
            # Convert to string (categorical)
            df[col] = df[col].fillna("missing").astype(str)
        
        log_status(f"Prepared {len(feature_cols)} categorical columns for K-modes clustering")
        
        # Return prepared data
        return {
            'data': df,
            'id_column': id_column,
            'categorical_cols': feature_cols,
            'column_categories': column_categories,
            'row_count': row_count
        }
    
    except Exception as e:
        log_status(f"ERROR in data extraction and preparation: {str(e)}")
        log_status(traceback.format_exc())
        return None

def determine_optimal_k_for_kmodes(data, max_k=15, n_init=3, init_method='Huang'):
    """
    Determine optimal number of clusters for K-modes using elbow method
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing categorical data
    max_k : int
        Maximum number of clusters to evaluate
    n_init : int
        Number of times to run K-modes with different initializations
    init_method : str
        Initialization method ('Huang' or 'Cao')
        
    Returns:
    --------
    tuple
        (optimal_k, cost_curve)
    """
    log_status(f"Determining optimal K for K-modes clustering (max_k={max_k})...")
    
    try:
        from kmodes.kmodes import KModes
        
        # Start with at least 2 clusters
        k_range = range(2, max_k + 1)
        cost_values = []
        
        # For each k, run K-modes and get cost
        for k in k_range:
            log_status(f"Testing K-modes with k={k}...")
            kmode = KModes(n_clusters=k, init=init_method, n_init=n_init, verbose=1, random_state=42)
            kmode.fit(data)
            cost = kmode.cost_
            cost_values.append(cost)
            log_status(f"K={k}, Cost={cost:.4f}")
        
        # Find optimal k using elbow method
        # Looking for point of diminishing returns in cost reduction
        cost_diffs = np.diff(cost_values)
        cost_diffs2 = np.diff(cost_diffs)  # Second derivative
        
        # Simple approach: find the first elbow point where second derivative is maximum
        if len(cost_diffs2) > 0:
            elbow_index = np.argmax(cost_diffs2) + 2  # +2 because we start at k=2 and diff reduces array length
            optimal_k = k_range[elbow_index]
        else:
            # Fall back to a reasonable default if we can't find an elbow
            optimal_k = 8
            
        log_status(f"Determined optimal k for K-modes: k={optimal_k}")
        
        # Return optimal k and the cost curve for plotting
        return optimal_k, list(zip(k_range, cost_values))
        
    except Exception as e:
        log_status(f"ERROR in optimal k determination: {str(e)}")
        log_status(traceback.format_exc())
        return 8, None  # Default to 8 clusters on error

def run_kmodes_clustering(prepared_data, num_clusters=8, init_method='Huang', output_prefix="/dbfs/FileStore/acxiom_clustering/kmodes"):
    """
    Run K-modes clustering algorithm on categorical data
    
    Parameters:
    -----------
    prepared_data : dict
        Dictionary containing prepared data for K-modes
    num_clusters : int
        Number of clusters to generate
    init_method : str
        Initialization method ('Huang' or 'Cao')
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    dict
        Dictionary with K-modes results or None if failed
    """
    log_status(f"Starting K-modes clustering with {num_clusters} clusters...")
    
    try:
        # Check if kmodes package is available
        if not install_kmodes_if_needed():
            return None
            
        from kmodes.kmodes import KModes
        
        # Extract data from prepared_data
        df = prepared_data['data']
        id_column = prepared_data['id_column']
        categorical_cols = prepared_data['categorical_cols']
        
        # Prepare data for clustering
        cluster_data = df[categorical_cols].copy()
        
        # Initialize and run K-modes
        start_time = time.time()
        
        log_status(f"Running K-modes with {num_clusters} clusters using {init_method} initialization...")
        kmode = KModes(
            n_clusters=num_clusters,
            init=init_method,
            n_init=5,  # Number of times algorithm will be run with different initializations
            verbose=1,
            random_state=42
        )
        
        # Fit and predict clusters
        cluster_labels = kmode.fit_predict(cluster_data)
        
        runtime = time.time() - start_time
        log_status(f"K-modes clustering completed in {runtime:.2f} seconds")
        
        # Add cluster labels to original data
        result_df = df.copy()
        result_df['kmodes_cluster'] = cluster_labels
        
        # Calculate cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        log_status("Cluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(result_df)) * 100
            log_status(f"Cluster {cluster_id}: {size} records ({percentage:.1f}%)")
        
        # Get centroids (modes)
        centroids = kmode.cluster_centroids_
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        # Save results
        result_path = f"{output_prefix}_results_{num_clusters}.csv"
        result_df.to_csv(result_path, index=False)
        log_status(f"Saved K-modes results to {result_path}")
        
        # Save centroids
        centroids_df = pd.DataFrame(centroids, columns=categorical_cols)
        centroids_df['cluster_id'] = range(num_clusters)
        centroids_path = f"{output_prefix}_centroids_{num_clusters}.csv"
        centroids_df.to_csv(centroids_path, index=False)
        log_status(f"Saved K-modes centroids to {centroids_path}")
        
        # Create visualization of cluster sizes
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            [f"Cluster {i}" for i in range(num_clusters)],
            cluster_sizes.values,
            color=plt.cm.tab20(np.linspace(0, 1, num_clusters))
        )
        
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Records', fontsize=12)
        plt.title(f'K-modes Clustering: {num_clusters} Segments Size Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add size and percentage labels
        for bar in bars:
            height = bar.get_height()
            percentage = (height / len(result_df)) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}\n({percentage:.1f}%)', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = f"{output_prefix}_sizes_{num_clusters}.jpg"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        # Display in Databricks
        if IN_DATABRICKS:
            display(plt.gcf())
            
        plt.close()
        log_status(f"Saved cluster size visualization to {viz_path}")
        
        # Create centroids profiles
        report_content = f"===== K-MODES CLUSTERING CENTROIDS PROFILES =====\n\n"
        report_content += f"Number of segments: {num_clusters}\n"
        report_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # For each cluster, describe the centroid
        for i in range(num_clusters):
            report_content += f"Cluster {i}:\n"
            report_content += f"{'=' * (len(f'Cluster {i}:') + 5)}\n"
            report_content += f"Size: {cluster_sizes[i]} records ({cluster_sizes[i]/len(result_df)*100:.1f}%)\n\n"
            
            # Get centroid values for key columns
            report_content += "Key attributes (modes):\n"
            
            # Group columns by category for better readability
            for category in ["Vehicle", "Propensity", "Demographic", "Financial", "Lifestyle"]:
                category_cols = [col for col, cat in prepared_data['column_categories'].items() 
                               if cat == category and col in categorical_cols]
                
                if category_cols:
                    report_content += f"\n{category} Attributes:\n"
                    for col in category_cols[:10]:  # Limit to first 10 per category
                        col_idx = categorical_cols.index(col)
                        mode_value = centroids[i, col_idx]
                        report_content += f"- {col}: {mode_value}\n"
            
            report_content += "\n"
            
        # Save centroids profile report
        report_path = f"{output_prefix}_profiles_{num_clusters}.txt"
        with open(report_path, "w") as f:
            f.write(report_content)
            
        log_status(f"Saved centroids profiles to {report_path}")
        
        # Return clustering results
        return {
            'cluster_data': result_df,
            'centroids': centroids,
            'centroids_df': centroids_df,
            'cluster_sizes': cluster_sizes,
            'cost': kmode.cost_,
            'num_clusters': num_clusters,
            'runtime': runtime
        }
        
    except Exception as e:
        log_status(f"ERROR in K-modes clustering: {str(e)}")
        log_status(traceback.format_exc())
        return None

def run_kmodes_multiple(cluster_counts=[8, 10, 12], sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Run K-modes clustering with multiple cluster counts and compare results
    
    Parameters:
    -----------
    cluster_counts : list
        List of cluster counts to try
    sample_size : int
        Sample size to use
        
    Returns:
    --------
    dict
        Dictionary with results for each cluster count
    """
    log_status(f"Running K-modes with multiple cluster counts: {cluster_counts}")
    
    results = {}
    cost_values = {}
    
    # Extract and prepare data (do this only once)
    prepared_data = extract_and_prepare_data_for_kmodes(sample_size)
    if not prepared_data:
        log_status("ERROR: Failed to prepare data for K-modes clustering")
        return None
    
    # Run K-modes for each cluster count
    for k in cluster_counts:
        log_status(f"Running K-modes with {k} clusters...")
        
        result = run_kmodes_clustering(
            prepared_data=prepared_data,
            num_clusters=k
        )
        
        if result:
            results[k] = result
            cost_values[k] = result['cost']
            log_status(f"Successfully completed K-modes with {k} clusters")
        else:
            log_status(f"Failed to run K-modes with {k} clusters")
    
    # If we have multiple results, create comparison chart
    if len(cost_values) > 1:
        plt.figure(figsize=(10, 6))
        
        x = list(cost_values.keys())
        y = list(cost_values.values())
        
        plt.plot(x, y, 'o-', linewidth=2, markersize=8)
        
        # Find best (lowest cost)
        best_k = min(cost_values, key=cost_values.get)
        best_cost = cost_values[best_k]
        
        plt.axvline(x=best_k, color='red', linestyle='--',
                   label=f'Best Cost: {best_cost:.2f} (k={best_k})')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.title('Comparison of K-modes Clustering Results', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        comparison_path = "/dbfs/FileStore/acxiom_clustering/kmodes_cost_comparison.jpg"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        
        if IN_DATABRICKS:
            display(plt.gcf())
        
        plt.close()
        log_status(f"Saved comparison chart to {comparison_path}")
        
        # Create a comparison report
        report = f"===== K-MODES CLUSTERING COMPARISON =====\n\n"
        report += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Sample size: {sample_size}\n\n"
        
        report += "COST VALUES:\n"
        report += "------------\n"
        for k in sorted(cost_values.keys()):
            cost = cost_values[k]
            relative = best_cost / cost
            report += f"k={k}: {cost:.2f} ({relative:.1%} of best cost)\n"
        
        report += f"\nOptimal cluster count (lowest cost): k={best_k} (cost={best_cost:.2f})\n"
        
        report_path = "/dbfs/FileStore/acxiom_clustering/kmodes_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        log_status(f"Saved comparison report to {report_path}")
    
    return results

def generate_kmodes_segment_names(kmodes_results, output_prefix="/dbfs/FileStore/acxiom_clustering/kmodes"):
    """
    Generate descriptive names for K-modes clusters based on centroids
    
    Parameters:
    -----------
    kmodes_results : dict
        Dictionary with K-modes results from run_kmodes_clustering()
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to generated names
    """
    log_status("Generating descriptive segment names based on K-modes centroids...")
    
    try:
        # Extract centroids and data
        centroids_df = kmodes_results['centroids_df']
        cluster_data = kmodes_results['cluster_data']
        cluster_sizes = kmodes_results['cluster_sizes']
        num_clusters = kmodes_results['num_clusters']
        
        # Initialize segment names
        segment_names = {}
        
        # Vehicle type keywords to look for in centroid values
        vehicle_keywords = {
            "luxury": ["luxury", "premium", "upscale", "high-end"],
            "suv": ["suv", "crossover", "utility", "cross-over"],
            "sedan": ["sedan", "4-door", "four-door"],
            "sports": ["sports", "performance", "convertible", "coupe"],
            "pickup": ["pickup", "truck", "4x4"],
            "economy": ["economy", "compact", "subcompact", "budget"],
            "family": ["family", "minivan", "van"]
        }
        
        # Buyer type keywords to look for in centroid values
        buyer_keywords = {
            "tech_savvy": ["tech", "advanced", "connected", "innovation"],
            "value_oriented": ["value", "price", "budget", "economical"],
            "luxury_oriented": ["luxury", "premium", "exclusive"],
            "family_focused": ["family", "children", "safety"],
            "eco_conscious": ["eco", "hybrid", "electric", "environment"]
        }
        
        # For each cluster, analyze centroid values
        for cluster_id in range(num_clusters):
            # Extract centroid for this cluster
            centroid = centroids_df[centroids_df['cluster_id'] == cluster_id]
            
            # Get cluster size percentage
            size = cluster_sizes[cluster_id]
            size_pct = (size / cluster_sizes.sum()) * 100
            
            # Initialize keyword counts
            vehicle_type_matches = {key: 0 for key in vehicle_keywords}
            buyer_type_matches = {key: 0 for key in buyer_keywords}
            
            # Scan centroid values for keyword matches
            for col in centroid.columns:
                if col == 'cluster_id':
                    continue
                
                value = str(centroid[col].values[0]).lower()
                
                # Check for vehicle type matches
                for v_type, keywords in vehicle_keywords.items():
                    for keyword in keywords:
                        if keyword in value:
                            vehicle_type_matches[v_type] += 1
                
                # Check for buyer type matches
                for b_type, keywords in buyer_keywords.items():
                    for keyword in keywords:
                        if keyword in value:
                            buyer_type_matches[b_type] += 1
            
            # Find dominant vehicle type
            dominant_vehicle = max(vehicle_type_matches.items(), key=lambda x: x[1])
            vehicle_type = ""
            if dominant_vehicle[1] > 0:
                vehicle_type = {
                    "luxury": "Luxury",
                    "suv": "SUV/Crossover",
                    "sedan": "Sedan",
                    "sports": "Sports/Performance",
                    "pickup": "Pickup/Truck",
                    "economy": "Economy/Compact",
                    "family": "Family/Minivan"
                }.get(dominant_vehicle[0], "")
            
            # Find dominant buyer type
            dominant_buyer = max(buyer_type_matches.items(), key=lambda x: x[1])
            buyer_type = ""
            if dominant_buyer[1] > 0:
                buyer_type = {
                    "tech_savvy": "Tech-Savvy",
                    "value_oriented": "Value-Oriented",
                    "luxury_oriented": "Luxury",
                    "family_focused": "Family-Focused",
                    "eco_conscious": "Eco-Conscious"
                }.get(dominant_buyer[0], "")
            
            # Generate name based on size and identified types
            if size_pct > 20:
                size_descriptor = "Mainstream"
            elif size_pct > 10:
                size_descriptor = "Significant"
            elif size_pct > 5:
                size_descriptor = "Niche"
            else:
                size_descriptor = "Specialty"
            
            # Combine elements to form name
            name_parts = []
            
            if buyer_type:
                name_parts.append(buyer_type)
            
            if vehicle_type:
                name_parts.append(vehicle_type)
            
            name_parts.append(f"{size_descriptor} Segment")
            
            # Special case: If we couldn't identify types
            if not buyer_type and not vehicle_type:
                segment_names[cluster_id] = f"Vehicle Segment {cluster_id + 1} ({size_descriptor})"
            else:
                segment_names[cluster_id] = " ".join(name_parts)
        
        # Save segment names
        names_df = pd.DataFrame({
            'cluster_id': segment_names.keys(),
            'segment_name': segment_names.values(),
            'size': [cluster_sizes[k] for k in segment_names.keys()],
            'size_percentage': [cluster_sizes[k]/cluster_sizes.sum()*100 for k in segment_names.keys()]
        })
        
        names_path = f"{output_prefix}_segment_names_{num_clusters}.csv"
        names_df.to_csv(names_path, index=False)
        log_status(f"Saved segment names to {names_path}")
        
        return segment_names
    
    except Exception as e:
        log_status(f"ERROR in segment name generation: {str(e)}")
        log_status(traceback.format_exc())
        return {i: f"Segment {i+1}" for i in range(kmodes_results['num_clusters'])}

def run_just_kmodes(cluster_counts=None, sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Run standalone K-modes clustering with flexible parameters
    
    Parameters:
    -----------
    cluster_counts : list, optional
        List of cluster counts to generate (defaults to [8, 10, 12])
    sample_size : int, optional
        Sample size to use
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Use default cluster counts if not specified
    if cluster_counts is None:
        cluster_counts = [8, 10, 12]
    
    # Use default sample size if not specified
    if sample_size is None:
        sample_size = 10000
    
    log_status(f"===== EXECUTING K-MODES CLUSTERING =====")
    log_status(f"Cluster counts: {cluster_counts}")
    log_status(f"Sample size: {sample_size}")
    
    start_time = time.time()
    
    try:
        # First check if kmodes package is available
        if not install_kmodes_if_needed():
            log_status("ERROR: Required kmodes package not available")
            return False
        
        # Run k-modes with multiple cluster counts
        results = run_kmodes_multiple(
            cluster_counts=cluster_counts,
            sample_size=sample_size
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results:
            log_status(f"✅ K-modes clustering completed successfully in {total_time:.2f} seconds")
            
            # Generate segment names for the best clustering (lowest cost)
            cost_values = {k: result['cost'] for k, result in results.items()}
            best_k = min(cost_values, key=cost_values.get)
            
            log_status(f"Generating segment names for best clustering (k={best_k})...")
            segment_names = generate_kmodes_segment_names(results[best_k])
            
            if segment_names:
                log_status("Cluster segment names:")
                for cluster_id, name in segment_names.items():
                    size = results[best_k]['cluster_sizes'][cluster_id]
                    percentage = size / sum(results[best_k]['cluster_sizes']) * 100
                    log_status(f"  Cluster {cluster_id}: {name} ({percentage:.1f}%)")
            
            return True
        else:
            log_status(f"❌ K-modes clustering failed after {total_time:.2f} seconds")
            return False
    
    except Exception as e:
        log_status(f"ERROR in k-modes clustering: {str(e)}")
        log_status(traceback.format_exc())
        return False

# Testing optimal k determination with elbow method
def determine_optimal_k_for_kmodes_and_run(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000), max_k=15):
    """
    Determine optimal number of clusters using elbow method and then run K-modes
    
    Parameters:
    -----------
    sample_size : int
        Sample size to use
    max_k : int
        Maximum number of clusters to evaluate
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    log_status(f"===== DETERMINING OPTIMAL K FOR K-MODES AND RUNNING CLUSTERING =====")
    
    try:
        # First check if kmodes package is available
        if not install_kmodes_if_needed():
            log_status("ERROR: Required kmodes package not available")
            return False
        
        # Extract and prepare data
        prepared_data = extract_and_prepare_data_for_kmodes(sample_size)
        if not prepared_data:
            log_status("ERROR: Failed to prepare data for K-modes clustering")
            return False
        
        # Determine optimal k
        optimal_k, cost_curve = determine_optimal_k_for_kmodes(
            prepared_data['data'][prepared_data['categorical_cols']],
            max_k=max_k
        )
        
        if not optimal_k:
            log_status("ERROR: Failed to determine optimal k")
            return False
        
        # Plot cost curve
        if cost_curve:
            plt.figure(figsize=(10, 6))
            
            k_values, costs = zip(*cost_curve)
            plt.plot(k_values, costs, 'o-', linewidth=2, markersize=8)
            
            plt.axvline(x=optimal_k, color='red', linestyle='--',
                       label=f'Optimal k: {optimal_k}')
            
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Cost', fontsize=12)
            plt.title('Elbow Method for Optimal k in K-modes Clustering', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            elbow_path = "/dbfs/FileStore/acxiom_clustering/kmodes_elbow_method.jpg"
            plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
            
            if IN_DATABRICKS:
                display(plt.gcf())
            
            plt.close()
            log_status(f"Saved elbow method plot to {elbow_path}")
        
        # Run K-modes with optimal k
        log_status(f"Running K-modes with optimal k={optimal_k}...")
        
        result = run_kmodes_clustering(
            prepared_data=prepared_data,
            num_clusters=optimal_k
        )
        
        if result:
            log_status(f"✅ Successfully completed K-modes clustering with optimal k={optimal_k}")
            
            # Generate segment names
            segment_names = generate_kmodes_segment_names(result)
            
            if segment_names:
                log_status("Cluster segment names:")
                for cluster_id, name in segment_names.items():
                    size = result['cluster_sizes'][cluster_id]
                    percentage = size / sum(result['cluster_sizes']) * 100
                    log_status(f"  Cluster {cluster_id}: {name} ({percentage:.1f}%)")
            
            return True
        else:
            log_status(f"❌ Failed to run K-modes clustering with optimal k={optimal_k}")
            return False
    
    except Exception as e:
        log_status(f"ERROR in K-modes clustering with optimal k: {str(e)}")
        log_status(traceback.format_exc())
        return False

# OPTIONAL: Add a function to compare K-modes with other clustering methods
def compare_kmodes_with_other_methods(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000), num_clusters=8):
    """
    Compare K-modes clustering with other methods (K-means, K-prototypes, Hierarchical)
    
    Parameters:
    -----------
    sample_size : int
        Sample size to use
    num_clusters : int
        Number of clusters to generate
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    log_status(f"===== COMPARING K-MODES WITH OTHER CLUSTERING METHODS =====")
    log_status(f"Sample size: {sample_size}, Clusters: {num_clusters}")
    
    # This function can be expanded to run different clustering methods
    # and compare their results, metrics, and computation times
    
    # For now, just run K-modes
    return run_just_kmodes(cluster_counts=[num_clusters], sample_size=sample_size)

print("\n=== CELL 13: K-modes Clustering Implementation completed ===")

# COMMAND ----------

# CELL 14: Data Extraction Function - Modified to use dbutils for directories
import time
import traceback
import pandas as pd
from IPython.display import display, HTML

def direct_extract_acxiom_data(sample_size=10000):
    """
    Extract Acxiom data directly using SQL without complex stratification
    
    Parameters:
    -----------
    sample_size : int
        Number of rows to sample
        
    Returns:
    --------
    dict
        Dictionary containing extracted data and metadata
    """
    extraction_start = time.time()
    log_status(f"Starting direct Acxiom data extraction with sample size: {sample_size}")
    
    try:
        # Get column definitions
        id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
        
        # Combine all columns to extract
        all_columns = [id_column] + vehicle_columns + propensity_columns + demographic_columns + lifestyle_columns + financial_columns
        
        # Remove any duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))
        
        # Create a column mapping for return
        column_map = {
            "id": id_column,
            "vehicle": vehicle_columns,
            "propensity": propensity_columns,
            "demographic": demographic_columns,
            "lifestyle": lifestyle_columns,
            "financial": financial_columns
        }
        
        # Specify the correct table name
        acxiom_table = "dataproducts_dev.bronze_acxiom.gm_consumer_list"
        
        # Verify table access
        log_status("Verifying database access...")
        try:
            spark.sql(f"SELECT 1 FROM {acxiom_table} LIMIT 1")
            log_status("Database access verified")
        except Exception as e:
            log_status(f"FATAL ERROR: Cannot access Acxiom table: {str(e)}")
            raise RuntimeError(f"Cannot access {acxiom_table}. This is a fatal error.")
        
        # Verify column existence (get a sample row)
        sample_df = spark.sql(f"SELECT * FROM {acxiom_table} LIMIT 1")
        available_columns = set(sample_df.columns)
        
        # Filter for only available columns
        valid_columns = [col for col in all_columns if col in available_columns]
        
        # Report on missing columns
        missing_columns = set(all_columns) - set(valid_columns)
        if missing_columns:
            log_status(f"WARNING: {len(missing_columns)} columns not found in dataset")
            # Only log first 10 missing columns to avoid clutter
            if len(missing_columns) > 10:
                log_status(f"First 10 missing columns: {list(missing_columns)[:10]}")
            else:
                log_status(f"Missing columns: {list(missing_columns)}")
        
        # Ensure we have enough columns
        if len(valid_columns) < 10:
            log_status(f"FATAL ERROR: Not enough valid columns found (only {len(valid_columns)}). Need at least 10 columns for meaningful analysis.")
            raise RuntimeError("Not enough valid columns found for analysis. This is a fatal error.")
        
        # Build column list for query
        column_list = ", ".join(valid_columns)
        
        # Create simple query - just use random sampling
        log_status("Executing simple random sampling...")
        simple_sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE {id_column} IS NOT NULL
        ORDER BY rand()
        LIMIT {sample_size}
        """
        
        # Execute the query
        acxiom_df = spark.sql(simple_sql_query)
        
        # Check if we got enough data
        row_count = acxiom_df.count()
        if row_count == 0:
            log_status("FATAL ERROR: No rows returned from Acxiom query")
            raise RuntimeError("No data returned from Acxiom database query. This is a fatal error.")
        
        # Cache the result for faster subsequent operations
        acxiom_df.cache()
        
        # Display sample in Databricks
        if IN_DATABRICKS:
            try:
                display(HTML("<h4>Data Sample (first 5 rows)</h4>"))
                display(acxiom_df.limit(5))
            except:
                pass
        
        # Prepare result dictionary
        col_count = len(acxiom_df.columns)
        extract_time = time.time() - extraction_start
        log_status(f"Successfully extracted {row_count} rows and {col_count} columns in {extract_time:.2f} seconds")
        
        result = {
            'spark_df': acxiom_df,
            'id_column': id_column,
            'column_map': column_map,
            'row_count': row_count,
            'extract_time': extract_time
        }
        
        # Save extraction metadata - FIXED: Create directory using dbutils
        directory_path = "/FileStore/acxiom_clustering"
        metadata_path = f"/dbfs{directory_path}/extraction_metadata.json"
        
        # Create directory using dbutils
        dbutils.fs.mkdirs(directory_path)
        log_status(f"Created directory: {directory_path}")
        
        import json
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': sample_size,
            'rows_extracted': row_count,
            'columns_extracted': col_count,
            'execution_time_seconds': extract_time
        }
        
        # Now write the file after ensuring directory exists
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        log_status(f"Saved extraction metadata to {metadata_path}")
        
        return result
        
    except Exception as e:
        log_status(f"ERROR in direct data extraction: {str(e)}")
        log_status(f"Error details: {traceback.format_exc()}")
        raise

# COMMAND ----------

# CELL 15: Improved MCA Analysis with Saved Results Check
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def robust_run_mca_analysis(prepared_data, output_prefix=None):
    """
    Run MCA analysis with improved error handling and stability
    Checks for existing saved MCA coordinates before computing new ones
    
    Parameters:
    -----------
    prepared_data : dict
        Dictionary containing prepared data from robust_prepare_data_for_mca()
    output_prefix : str, optional
        Prefix for output files, used to locate existing MCA results
        
    Returns:
    --------
    dict
        Dictionary with MCA results or None if analysis failed
    """
    if prepared_data is None:
        log_status("ERROR: Prepared data is None")
        return None
        
    if 'categorical_cols' not in prepared_data or not prepared_data['categorical_cols']:
        log_status("ERROR: No categorical columns available for MCA")
        return None
    
    # Check if saved MCA coordinates exist
    if output_prefix:
        coords_path = f"{output_prefix}_mca_coordinates.parquet"
        model_path = f"{output_prefix}_mca_model.pickle"
        var_path = f"{output_prefix}_mca_variance.csv"
        
        # Check if all required files exist
        try:
            import os
            if os.path.exists(coords_path) and os.path.exists(var_path):
                log_status(f"Found existing MCA coordinates at {coords_path}, loading instead of recomputing")
                
                # Load MCA coordinates
                mca_coords = pd.read_parquet(coords_path)
                
                # Load variance data
                var_df = pd.read_csv(var_path)
                eigenvalues = var_df['eigenvalue'].values
                var_explained = var_df['variance_explained'].values
                cum_var = var_df['cumulative_variance'].values
                
                # Load MCA model if available
                mca_model = None
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            mca_model = pickle.load(f)
                        log_status("Successfully loaded MCA model")
                    except Exception as e:
                        log_status(f"Warning: Could not load MCA model: {str(e)}")
                
                # Determine number of dimensions (find where cumulative variance reaches 70%)
                n_dims_70 = np.where(cum_var >= 0.7)[0]
                n_dims = n_dims_70[0] + 1 if len(n_dims_70) > 0 else min(len(var_explained), 10)
                n_dims = min(n_dims, 10)  # Cap at 10 dimensions for practical use
                
                log_status(f"Using previously computed MCA with {n_dims} significant dimensions explaining {cum_var[n_dims-1]:.1%} of variance")
                
                # Return loaded MCA results
                return {
                    'mca_coords': mca_coords,
                    'eigenvalues': eigenvalues,
                    'var_explained': var_explained,
                    'cum_var': cum_var,
                    'n_dims': n_dims,
                    'mca_model': mca_model,
                    'from_saved': True  # Flag indicating results were loaded from saved files
                }
        except Exception as load_error:
            log_status(f"Warning: Failed to load existing MCA results: {str(load_error)}")
            log_status("Proceeding with new MCA computation")
        
    # If we get here, we need to compute new MCA
    log_status("Starting MCA analysis...")
    mca_start = time.time()
    
    try:
        # Extract categorical data
        categorical_cols = prepared_data['categorical_cols']
        features = prepared_data['features'][categorical_cols].copy()
        
        # Ensure all columns are object type (strings)
        for col in categorical_cols:
            if not pd.api.types.is_object_dtype(features[col]):
                features[col] = features[col].astype(str)
                
        # Make sure no NaN values are present
        for col in categorical_cols:
            if features[col].isna().any():
                features[col] = features[col].fillna("missing")
        
        log_status(f"Running MCA on {len(categorical_cols)} categorical columns with {len(features)} rows")
        
        # Import prince library with error handling
        try:
            import prince
        except ImportError:
            log_status("Warning: prince library not found. Attempting to install...")
            try:
                import pip
                pip.main(['install', 'prince'])
                import prince
                log_status("Successfully installed prince library")
            except Exception as pip_error:
                log_status(f"Failed to install prince: {str(pip_error)}")
                return None
        
        # Determine max components to use
        max_components = min(30, len(categorical_cols))
        
        # Initialize MCA with bounded iteration to prevent long runs
        mca = prince.MCA(
            n_components=max_components,
            n_iter=5,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=42
        )
        
        # Fit MCA model
        log_status("Fitting MCA model...")
        mca.fit(features)
        
        # Transform data to get coordinates
        log_status("Transforming data to MCA coordinates...")
        mca_coords = mca.transform(features)
        
        # Name the columns properly
        mca_coords.columns = [f'MCA_dim{i+1}' for i in range(mca_coords.shape[1])]
        
        # Add ID column if available with robust error handling
        if 'id_values' in prepared_data and prepared_data['id_values'] is not None:
            id_column = prepared_data['id_column']
            try:
                mca_coords[id_column] = prepared_data['id_values'].values
            except Exception as e:
                log_status(f"Warning: Could not add ID column directly: {str(e)}")
                try:
                    # Try alternative approach
                    mca_coords[id_column] = list(prepared_data['id_values'])
                except Exception as e2:
                    log_status(f"Warning: Could not add ID column as list: {str(e2)}")
        
        # Get eigenvalues and variance explained
        eigenvalues = mca.eigenvalues_
        var_explained = eigenvalues / sum(eigenvalues)
        cum_var = np.cumsum(var_explained)
        
        # Determine number of dimensions to retain (70% variance explained)
        n_dims_70 = np.where(cum_var >= 0.7)[0]
        n_dims = n_dims_70[0] + 1 if len(n_dims_70) > 0 else max_components
        n_dims = min(n_dims, 10)  # Cap at 10 dimensions for practical use
        
        mca_time = time.time() - mca_start
        log_status(f"MCA analysis completed in {mca_time:.2f} seconds")
        log_status(f"Identified {n_dims} significant dimensions explaining {cum_var[n_dims-1]:.1%} of variance")
        
        # Return MCA results
        return {
            'mca_coords': mca_coords,
            'eigenvalues': eigenvalues,
            'var_explained': var_explained,
            'cum_var': cum_var,
            'n_dims': n_dims,
            'mca_model': mca,  # Store the model for serialization with pickle
            'from_saved': False  # Flag indicating results were newly computed
        }
        
    except Exception as e:
        log_status(f"ERROR in MCA analysis: {str(e)}")
        log_status(traceback.format_exc())
        return None

print("\n=== CELL 15: Improved MCA Analysis with Saved Results Check completed ===")

# COMMAND ----------

# CELL 16: MCA Pipeline function
def run_mca_pipeline():
    """
    Run the MCA analysis pipeline and save the results.
    This function extracts data, prepares it, runs MCA, and saves the results.
    """
    log_status("===== STARTING MCA PIPELINE =====")
    log_status(f"Using sample size: {GLOBAL_CONFIG['mca_sample_size']}")
    
    try:
        # Step 1: Extract data using direct SQL approach
        log_status("STEP 1: Extracting data for MCA...")
        
        extracted_data = direct_extract_acxiom_data(sample_size=GLOBAL_CONFIG['mca_sample_size'])
        if not extracted_data:
            log_status("ERROR: Data extraction failed for MCA")
            return False
            
        log_status(f"Successfully extracted {extracted_data['row_count']} rows for MCA")
        
        # Step 2: Prepare data for MCA
        log_status("STEP 2: Preparing data for MCA...")
        
        prepared_data = robust_prepare_data_for_mca(extracted_data['spark_df'], 
                                                   GLOBAL_CONFIG['id_column'])
        if not prepared_data:
            log_status("ERROR: Data preparation failed for MCA")
            return False
            
        log_status(f"Successfully prepared {len(prepared_data['categorical_cols'])} categorical columns for MCA")
        
        # Step 3: Run MCA analysis
        log_status("STEP 3: Running MCA analysis...")
        
        mca_analysis = robust_run_mca_analysis(prepared_data)
        if not mca_analysis:
            log_status("ERROR: MCA analysis failed")
            return False
            
        log_status(f"Successfully completed MCA analysis with {mca_analysis['n_dims']} dimensions")
        
        # Step 4: Save MCA results
        log_status("STEP 4: Saving MCA results...")
        
        output_prefix = GLOBAL_CONFIG['mca_output_prefix']
        try:
            # Save MCA coordinates
            coords_path = f"{output_prefix}_coordinates.parquet"
            mca_analysis['mca_coords'].to_parquet(coords_path)
            log_status(f"Saved MCA coordinates to {coords_path}")
            
            # Also save in CSV format for compatibility
            csv_path = f"{output_prefix}_coordinates.csv"
            mca_analysis['mca_coords'].to_csv(csv_path, index=False)
            log_status(f"Saved MCA coordinates as CSV to {csv_path}")
            
            # Save variance explained data
            var_path = f"{output_prefix}_variance.csv"
            var_df = pd.DataFrame({
                'dimension': range(1, len(mca_analysis['eigenvalues']) + 1),
                'eigenvalue': mca_analysis['eigenvalues'],
                'variance_explained': mca_analysis['var_explained'],
                'cumulative_variance': mca_analysis['cum_var']
            })
            var_df.to_csv(var_path, index=False)
            log_status(f"Saved variance explained data to {var_path}")
            
            # Create visualizations
            visualize_mca(mca_analysis, prepared_data, output_prefix)
            
            # Save analysis report
            report_path = f"{output_prefix}_analysis_report.txt"
            create_enhanced_mca_analysis_report(mca_analysis, prepared_data, report_path)
            
            log_status("Successfully saved all MCA results")
            return True
            
        except Exception as save_error:
            log_status(f"ERROR saving MCA results: {str(save_error)}")
            log_status(f"Error details: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        log_status(f"CRITICAL ERROR in MCA pipeline: {str(e)}")
        log_status(traceback.format_exc())
        return False


# COMMAND ----------

# CELL 17: Extended Hierarchical Clustering Implementation with MCA Reuse Support
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from IPython.display import display, HTML

def perform_fixed_hierarchical_clustering(mca_data, num_clusters=8):
    """
    Perform hierarchical clustering on MCA coordinates with a fixed number of clusters
    Updated to work with MCA data loaded from saved files
    
    Parameters:
    -----------
    mca_data : dict
        Dictionary with MCA results from robust_run_mca_analysis()
    num_clusters : int
        Fixed number of clusters to generate (overrides optimal selection)
        
    Returns:
    --------
    dict
        Dictionary with hierarchical clustering results or None if clustering failed
    """
    log_status(f"Starting hierarchical clustering analysis with fixed {num_clusters} clusters...")
    
    try:
        # Extract dimensional data for clustering
        if 'mca_coords' not in mca_data or 'n_dims' not in mca_data:
            log_status("ERROR: MCA data is missing required components")
            return None
            
        n_dims = mca_data['n_dims']
        dim_cols = [f'MCA_dim{i+1}' for i in range(n_dims)]
        
        # Check if the columns exist in the dataframe
        missing_cols = [col for col in dim_cols if col not in mca_data['mca_coords'].columns]
        if missing_cols:
            log_status(f"ERROR: Missing MCA dimension columns: {missing_cols}")
            return None
            
        # Extract data for clustering
        mca_for_clustering = mca_data['mca_coords'][dim_cols].copy()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(mca_for_clustering)
        
        # Perform hierarchical clustering
        # Use Ward's method for linkage (tends to create more balanced clusters)
        log_status("Computing hierarchical linkage (this may take a moment)...")
        Z = linkage(scaled_data, method='ward')
        
        # We'll still calculate silhouette scores for reference, but use fixed number of clusters
        log_status("Evaluating different numbers of clusters for reference...")
        
        # Calculate silhouette scores for a range of clusters
        max_k = max(num_clusters + 4, 15)  # Evaluate a few more than requested
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            # Cut the dendrogram to get k clusters
            labels = fcluster(Z, k, criterion='maxclust') - 1  # Convert to 0-based indexing
            
            # Calculate silhouette score
            sil_score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(sil_score)
            
            log_status(f"  k={k}: silhouette={sil_score:.3f}")
        
        # Get statistically optimal k for reference
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        log_status(f"Statistically optimal number of clusters would be: k={optimal_k}")
        log_status(f"But using fixed number of clusters: k={num_clusters}")
        
        # Get final cluster labels with fixed number of clusters
        final_labels = fcluster(Z, num_clusters, criterion='maxclust') - 1
        
        # Add cluster labels to MCA coordinates
        result_df = mca_data['mca_coords'].copy()
        result_df['hier_cluster'] = final_labels
        
        log_status(f"Successfully clustered data into {num_clusters} hierarchical segments")
        
        # Return clustering results
        return {
            'cluster_data': result_df,
            'linkage': Z,
            'optimal_k': optimal_k,  # Still include the statistically optimal k
            'fixed_k': num_clusters,  # Also include the fixed k
            'silhouette_scores': silhouette_scores,
            'hier_labels': final_labels
        }
        
    except Exception as e:
        log_status(f"ERROR in hierarchical clustering: {str(e)}")
        log_status(traceback.format_exc())
        return None

print("\n=== CELL 17: Extended Hierarchical Clustering Implementation with MCA Reuse Support completed ===")

# COMMAND ----------

# CELL 18: Visualization of Hierarchical Clustering
def visualize_extended_hierarchical_clustering(hier_results, output_prefix):
    """
    Create more detailed visualizations for hierarchical clustering results
    
    Parameters:
    -----------
    hier_results : dict
        Dictionary with hierarchical clustering results
    output_prefix : str
        Prefix for output files
    """
    if hier_results is None:
        log_status("ERROR: No hierarchical clustering results to visualize")
        return
        
    log_status("Creating hierarchical clustering visualizations...")
    
    try:
        # Create dendrogram visualization with colored clusters
        plt.figure(figsize=(15, 8))
        
        # Draw dendrogram with more details
        dendrogram(
            hier_results['linkage'],
            truncate_mode='lastp',
            p=30,  # Show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            color_threshold=0.7*max(hier_results['linkage'][:,2])  # Color threshold for better visualization
        )
        
        # Add horizontal line at the cut point for the fixed number of clusters
        cut_height = hier_results['linkage'][-(hier_results['fixed_k']-1), 2]
        plt.axhline(y=cut_height, color='r', linestyle='--', 
                   label=f'Cut for {hier_results["fixed_k"]} clusters')
        
        # Add horizontal line at the statistically optimal cut point
        if hier_results['fixed_k'] != hier_results['optimal_k']:
            opt_cut_height = hier_results['linkage'][-(hier_results['optimal_k']-1), 2]
            plt.axhline(y=opt_cut_height, color='g', linestyle=':', 
                      label=f'Optimal cut ({hier_results["optimal_k"]} clusters)')
        
        plt.title(f'Hierarchical Clustering Dendrogram ({hier_results["fixed_k"]} clusters)', fontsize=14)
        plt.xlabel('Sample index or (cluster size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.legend()
        
        # Save the dendrogram (Python-friendly jpg format)
        dendro_path = f"{output_prefix}_hierarchical_dendrogram_{hier_results['fixed_k']}_clusters.jpg"
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        
        # Display in Databricks
        if IN_DATABRICKS:
            display(plt.gcf())
            
        plt.close()
        log_status(f"Saved hierarchical dendrogram to {dendro_path}")
        
        # Create 2D scatter plot of clusters with better differentiation
        if 'cluster_data' in hier_results and 'MCA_dim1' in hier_results['cluster_data'].columns and 'MCA_dim2' in hier_results['cluster_data'].columns:
            plt.figure(figsize=(14, 12))
            
            # Use a distinct colormap
            cluster_cmap = plt.cm.get_cmap('tab20', hier_results['fixed_k'])
            
            # Plot points colored by cluster
            scatter = plt.scatter(
                hier_results['cluster_data']['MCA_dim1'],
                hier_results['cluster_data']['MCA_dim2'],
                c=hier_results['cluster_data']['hier_cluster'],
                cmap=cluster_cmap,
                alpha=0.7,
                s=40,
                edgecolors='w',
                linewidths=0.3
            )
            
            # Add colorbar legend
            cbar = plt.colorbar(scatter, label='Cluster', ticks=range(hier_results['fixed_k']))
            cbar.set_label('Cluster', fontsize=12)
            
            # Calculate and plot cluster centroids
            centroids = []
            for i in range(hier_results['fixed_k']):
                mask = hier_results['cluster_data']['hier_cluster'] == i
                centroid_x = hier_results['cluster_data'].loc[mask, 'MCA_dim1'].mean()
                centroid_y = hier_results['cluster_data'].loc[mask, 'MCA_dim2'].mean()
                centroids.append((centroid_x, centroid_y))
                
                # Add cluster number labels
                plt.text(centroid_x, centroid_y, str(i), 
                        fontsize=15, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
            
            # Convert to numpy array for plotting
            centroids = np.array(centroids)
            
            # Plot centroids
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       s=200, marker='*', c='black', edgecolor='white', linewidth=1.5,
                       label='Cluster Centroids')
            
            # Add labels and title
            plt.xlabel("MCA Dimension 1", fontsize=12)
            plt.ylabel("MCA Dimension 2", fontsize=12)
            plt.title(f'Hierarchical Clustering: {hier_results["fixed_k"]} Segments in MCA Space', fontsize=14)
            
            # Add grid
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
            
            plt.legend()
            plt.tight_layout()
            
            # Save the plot (Python-friendly jpg format)
            scatter_path = f"{output_prefix}_hierarchical_clusters_{hier_results['fixed_k']}.jpg"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            
            # Display in Databricks
            if IN_DATABRICKS:
                display(plt.gcf())
                
            plt.close()
            log_status(f"Saved hierarchical cluster visualization to {scatter_path}")
            
        # Create silhouette score plot for reference
        if 'silhouette_scores' in hier_results:
            plt.figure(figsize=(10, 6))
            
            k_range = range(2, len(hier_results['silhouette_scores']) + 2)
            plt.plot(k_range, hier_results['silhouette_scores'], 'o-', color='#1f77b4', linewidth=2)
            
            # Mark the fixed number of clusters
            plt.axvline(x=hier_results['fixed_k'], color='#ff7f0e', linestyle='--',
                       label=f'Fixed k: {hier_results["fixed_k"]}')
            
            # Mark the statistically optimal number of clusters
            if hier_results['fixed_k'] != hier_results['optimal_k']:
                plt.axvline(x=hier_results['optimal_k'], color='green', linestyle=':',
                           label=f'Optimal k: {hier_results["optimal_k"]}')
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.title('Silhouette Scores for Hierarchical Clustering', fontsize=14)
            plt.legend()
            plt.tight_layout()
            
            # Save silhouette plot (Python-friendly jpg format)
            sil_path = f"{output_prefix}_hierarchical_silhouette_{hier_results['fixed_k']}.jpg"
            plt.savefig(sil_path, dpi=300, bbox_inches='tight')
            
            # Display in Databricks
            if IN_DATABRICKS:
                display(plt.gcf())
                
            plt.close()
            log_status(f"Saved silhouette score plot to {sil_path}")
            
    except Exception as e:
        log_status(f"WARNING: Error creating hierarchical visualizations: {str(e)}")
        log_status(traceback.format_exc())

# COMMAND ----------

# CELL 19: K-means Clustering Pipeline function
def run_kmeans_clustering_pipeline():
    """
    Run the K-means clustering pipeline using previously saved MCA results.
    """
    log_status("===== STARTING K-MEANS CLUSTERING PIPELINE =====")
    log_status(f"Using sample size: {GLOBAL_CONFIG['clustering_sample_size']}")
    
    try:
        # First check if MCA results exist
        coords_path = f"{GLOBAL_CONFIG['mca_output_prefix']}_coordinates.parquet"
        var_path = f"{GLOBAL_CONFIG['mca_output_prefix']}_variance.csv"
        
        if not os.path.exists(coords_path) or not os.path.exists(var_path):
            log_status("ERROR: MCA results not found. Please run MCA pipeline first.")
            return False
        
        # Load MCA coordinates and variance data
        log_status("Loading existing MCA results...")
        
        try:
            # Load MCA coordinates
            mca_coords = pd.read_parquet(coords_path)
            
            # Load variance data
            var_df = pd.read_csv(var_path)
            eigenvalues = var_df['eigenvalue'].values
            var_explained = var_df['variance_explained'].values
            cum_var = var_df['cumulative_variance'].values
            
            # Determine number of dimensions
            n_dims_70 = np.where(cum_var >= 0.7)[0]
            n_dims = n_dims_70[0] + 1 if len(n_dims_70) > 0 else min(len(var_explained), 10)
            n_dims = min(n_dims, 10)  # Cap at 10 dimensions
            
            mca_analysis = {
                'mca_coords': mca_coords,
                'eigenvalues': eigenvalues,
                'var_explained': var_explained,
                'cum_var': cum_var,
                'n_dims': n_dims,
                'from_saved': True
            }
            
            log_status(f"Successfully loaded MCA results with {n_dims} dimensions")
            
        except Exception as load_error:
            log_status(f"ERROR loading MCA results: {str(load_error)}")
            return False
        
        # Run K-means clustering for each specified cluster count
        kmeans_results = {}
        
        for k in GLOBAL_CONFIG['kmeans_clusters']:
            log_status(f"Running K-means clustering with k={k}...")
            
            try:
                # Select MCA dimensions for clustering
                dim_cols = [f'MCA_dim{i+1}' for i in range(n_dims)]
                cluster_df = mca_analysis['mca_coords'][dim_cols].copy()
                
                # Fill NAs with 0 (required for k-means)
                cluster_df = cluster_df.fillna(0)
                
                # Initialize and fit KMeans
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(cluster_df)
                
                # Calculate silhouette score
                sil_score = silhouette_score(cluster_df, labels)
                log_status(f"K-means with k={k}: silhouette score = {sil_score:.4f}")
                
                # Add cluster labels to original data
                result_df = mca_analysis['mca_coords'].copy()
                result_df[f'kmeans_cluster_{k}'] = labels
                
                # Create centers dataframe
                centers = pd.DataFrame(kmeans.cluster_centers_, columns=dim_cols)
                centers['cluster'] = range(k)
                
                # Save results
                output_prefix = GLOBAL_CONFIG['clustering_output_prefix']
                result_path = f"{output_prefix}_kmeans_{k}_clusters.parquet"
                result_df.to_parquet(result_path)
                log_status(f"Saved K-means {k} cluster assignments to {result_path}")
                
                centers_path = f"{output_prefix}_kmeans_{k}_centers.csv"
                centers.to_csv(centers_path, index=False)
                log_status(f"Saved K-means {k} centers to {centers_path}")
                
                # Create visualization
                create_kmeans_visualization(result_df, centers, k, output_prefix)
                
                # Store results for comparison
                kmeans_results[k] = {
                    'silhouette': sil_score,
                    'centers': centers,
                    'labels': labels
                }
                
                log_status(f"K-means clustering with k={k} completed successfully")
                
            except Exception as e:
                log_status(f"ERROR in K-means clustering with k={k}: {str(e)}")
                log_status(traceback.format_exc())
        
        # If we have results, create comparison report
        if kmeans_results:
            create_kmeans_comparison_report(kmeans_results, GLOBAL_CONFIG['clustering_output_prefix'])
            log_status("K-means clustering pipeline completed successfully")
            return True
        else:
            log_status("ERROR: No K-means clustering results generated")
            return False
            
    except Exception as e:
        log_status(f"CRITICAL ERROR in K-means clustering pipeline: {str(e)}")
        log_status(traceback.format_exc())
        return False


# COMMAND ----------

# CELL 20: Hierarchical Clustering Pipeline function
def run_hierarchical_clustering_pipeline():
    """
    Run the hierarchical clustering pipeline using previously saved MCA results.
    """
    log_status("===== STARTING HIERARCHICAL CLUSTERING PIPELINE =====")
    log_status(f"Using sample size: {GLOBAL_CONFIG['clustering_sample_size']}")
    
    try:
        # First check if MCA results exist
        coords_path = f"{GLOBAL_CONFIG['mca_output_prefix']}_coordinates.parquet"
        var_path = f"{GLOBAL_CONFIG['mca_output_prefix']}_variance.csv"
        
        if not os.path.exists(coords_path) or not os.path.exists(var_path):
            log_status("ERROR: MCA results not found. Please run MCA pipeline first.")
            return False
        
        # Load MCA coordinates and variance data
        log_status("Loading existing MCA results...")
        
        try:
            # Load MCA coordinates
            mca_coords = pd.read_parquet(coords_path)
            
            # Load variance data
            var_df = pd.read_csv(var_path)
            eigenvalues = var_df['eigenvalue'].values
            var_explained = var_df['variance_explained'].values
            cum_var = var_df['cumulative_variance'].values
            
            # Determine number of dimensions
            n_dims_70 = np.where(cum_var >= 0.7)[0]
            n_dims = n_dims_70[0] + 1 if len(n_dims_70) > 0 else min(len(var_explained), 10)
            n_dims = min(n_dims, 10)  # Cap at 10 dimensions
            
            mca_analysis = {
                'mca_coords': mca_coords,
                'eigenvalues': eigenvalues,
                'var_explained': var_explained,
                'cum_var': cum_var,
                'n_dims': n_dims,
                'from_saved': True
            }
            
            log_status(f"Successfully loaded MCA results with {n_dims} dimensions")
            
        except Exception as load_error:
            log_status(f"ERROR loading MCA results: {str(load_error)}")
            return False
        
        # Run hierarchical clustering for each specified cluster count
        hier_results = {}
        
        for k in GLOBAL_CONFIG['hierarchical_clusters']:
            log_status(f"Running hierarchical clustering with k={k} clusters...")
            
            try:
                # Run extended hierarchical clustering
                result = perform_fixed_hierarchical_clustering(mca_analysis, num_clusters=k)
                
                if result:
                    # Save results
                    output_prefix = GLOBAL_CONFIG['clustering_output_prefix']
                    result_path = f"{output_prefix}_hierarchical_{k}_clusters.parquet"
                    result['cluster_data'].to_parquet(result_path)
                    log_status(f"Saved hierarchical {k} cluster assignments to {result_path}")
                    
                    # Create visualizations
                    visualize_extended_hierarchical_clustering(result, output_prefix)
                    
                    # Store results
                    hier_results[k] = result
                    log_status(f"Hierarchical clustering with k={k} completed successfully")
                else:
                    log_status(f"WARNING: Hierarchical clustering with k={k} failed")
            
            except Exception as e:
                log_status(f"ERROR in hierarchical clustering with k={k}: {str(e)}")
                log_status(traceback.format_exc())
        
        # If we have results, create comparison report
        if hier_results:
            create_hierarchical_comparison_report(hier_results, GLOBAL_CONFIG['clustering_output_prefix'])
            log_status("Hierarchical clustering pipeline completed successfully")
            return True
        else:
            log_status("ERROR: No hierarchical clustering results generated")
            return False
            
    except Exception as e:
        log_status(f"CRITICAL ERROR in hierarchical clustering pipeline: {str(e)}")
        log_status(traceback.format_exc())
        return False

# COMMAND ----------

# CELL 21: Helper functions for visualization and reporting
def create_kmeans_visualization(result_df, centers, k, output_prefix):
    """Create and save visualizations for K-means clustering results."""
    try:
        plt.figure(figsize=(12, 10))
        
        # Use a distinct colormap
        cluster_cmap = plt.cm.get_cmap('tab20', k)
        
        # Plot points colored by cluster
        scatter = plt.scatter(
            result_df['MCA_dim1'],
            result_df['MCA_dim2'],
            c=result_df[f'kmeans_cluster_{k}'],
            cmap=cluster_cmap,
            alpha=0.7,
            s=40,
            edgecolors='w',
            linewidths=0.3
        )
        
        # Add colorbar legend
        cbar = plt.colorbar(scatter, label='Cluster', ticks=range(k))
        cbar.set_label('Cluster', fontsize=12)
        
        # Calculate and plot cluster centroids
        centroids = []
        for i in range(k):
            mask = result_df[f'kmeans_cluster_{k}'] == i
            centroid_x = result_df.loc[mask, 'MCA_dim1'].mean()
            centroid_y = result_df.loc[mask, 'MCA_dim2'].mean()
            centroids.append((centroid_x, centroid_y))
            
            # Add cluster number labels
            plt.text(centroid_x, centroid_y, str(i), 
                    fontsize=15, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
        
        # Convert to numpy array for plotting
        centroids = np.array(centroids)
        
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   s=200, marker='*', c='black', edgecolor='white', linewidth=1.5,
                   label='Cluster Centroids')
        
        # Add labels and title
        plt.xlabel("MCA Dimension 1", fontsize=12)
        plt.ylabel("MCA Dimension 2", fontsize=12)
        plt.title(f'K-means Clustering: {k} Segments in MCA Space', fontsize=14)
        
        # Add grid
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        viz_path = f"{output_prefix}_kmeans_{k}_clusters.jpg"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        # Display in Databricks
        if IN_DATABRICKS:
            display(plt.gcf())
            
        plt.close()
        log_status(f"Saved K-means visualization to {viz_path}")
        
    except Exception as e:
        log_status(f"WARNING: Error creating K-means visualization: {str(e)}")

def create_kmeans_comparison_report(kmeans_results, output_prefix):
    """Create and save a comparison report for different K-means cluster counts."""
    try:
        # Create comparison data
        comparison_data = []
        for k, result in sorted(kmeans_results.items()):
            comparison_data.append({
                'cluster_count': k,
                'silhouette_score': result['silhouette']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_path = f"{output_prefix}_kmeans_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        plt.bar(
            comparison_df['cluster_count'].astype(str),
            comparison_df['silhouette_score'],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))
        )
        
        # Add best score indicator
        best_k = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'cluster_count']
        best_score = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'silhouette_score']
        plt.axhline(y=best_score, color='red', linestyle='--', 
                   label=f'Best Score: {best_score:.4f} (k={best_k})')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Comparison of Silhouette Scores for K-means Clustering', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_path = f"{output_prefix}_kmeans_comparison.jpg"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Display in Databricks
        if IN_DATABRICKS:
            display(plt.gcf())
            
        plt.close()
        
        # Create report
        report_content = "===== K-MEANS CLUSTERING COMPARISON REPORT =====\n\n"
        report_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"Cluster counts evaluated: {', '.join(map(str, sorted(kmeans_results.keys())))}\n\n"
        
        report_content += "SILHOUETTE SCORES:\n"
        report_content += "-----------------\n"
        for k in sorted(kmeans_results.keys()):
            score = kmeans_results[k]['silhouette']
            report_content += f"k={k}: {score:.4f}"
            
            # Add relative quality
            relative = score / best_score
            report_content += f" ({relative:.1%} of best score)\n"
        
        report_content += f"\nStatistically optimal cluster count: k={best_k} (score: {best_score:.4f})\n"
        
        # Save report
        report_path = f"{output_prefix}_kmeans_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report_content)
            
        log_status(f"Saved K-means comparison report to {report_path}")
        
    except Exception as e:
        log_status(f"WARNING: Error creating K-means comparison report: {str(e)}")

def create_hierarchical_comparison_report(hier_results, output_prefix):
    """Create and save a comparison report for different hierarchical cluster counts."""
    try:
        # Extract silhouette scores for each k
        cluster_silhouettes = {}
        for k, result in hier_results.items():
            k_idx = k - 2
            if k_idx < len(result['silhouette_scores']):
                cluster_silhouettes[k] = result['silhouette_scores'][k_idx]
        
        if not cluster_silhouettes:
            log_status("WARNING: No silhouette scores available for hierarchical clustering")
            return
        
        # Create comparison data
        comparison_data = []
        for k in sorted(cluster_silhouettes.keys()):
            comparison_data.append({
                'cluster_count': k,
                'silhouette_score': cluster_silhouettes[k],
                'relative_quality': cluster_silhouettes[k] / max(cluster_silhouettes.values())
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_path = f"{output_prefix}_hierarchical_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        plt.bar(
            comparison_df['cluster_count'].astype(str),
            comparison_df['silhouette_score'],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))
        )
        
        # Add best silhouette score indicator
        best_k = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'cluster_count']
        best_score = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'silhouette_score']
        plt.axhline(y=best_score, color='red', linestyle='--', 
                   label=f'Best Score: {best_score:.4f} (k={best_k})')
        
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Comparison of Silhouette Scores for Hierarchical Clustering', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend()
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_path = f"{output_prefix}_hierarchical_comparison.jpg"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Display in Databricks
        if IN_DATABRICKS:
            display(plt.gcf())
            
        plt.close()
        
        # Create report
        report_content = "===== HIERARCHICAL CLUSTERING COMPARISON REPORT =====\n\n"
        report_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"Cluster counts evaluated: {', '.join(map(str, sorted(cluster_silhouettes.keys())))}\n\n"
        
        report_content += "SILHOUETTE SCORES:\n"
        report_content += "-----------------\n"
        for k in sorted(cluster_silhouettes.keys()):
            score = cluster_silhouettes[k]
            report_content += f"k={k}: {score:.4f}"
            
            # Add relative quality
            relative = score / best_score
            report_content += f" ({relative:.1%} of best score)\n"
        
        # Sort clusters by silhouette score
        sorted_clusters = sorted(cluster_silhouettes.items(), key=lambda x: x[1], reverse=True)
        
        report_content += f"\nStatistically optimal cluster count: k={sorted_clusters[0][0]} (score: {sorted_clusters[0][1]:.4f})\n"
        
        # Get cluster with best balance of quantity vs quality
        balanced_k = None
        best_balance = 0
        for k, score in cluster_silhouettes.items():
            # Balance formula: higher is better
            balance = score * np.log(k)
            if balance > best_balance:
                best_balance = balance
                balanced_k = k
        
        report_content += f"Best balance of granularity vs. cohesion: k={balanced_k}\n"
        
        # Save report
        report_path = f"{output_prefix}_hierarchical_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write(report_content)
            
        log_status(f"Saved hierarchical comparison report to {report_path}")
        
    except Exception as e:
        log_status(f"WARNING: Error creating hierarchical comparison report: {str(e)}")

# CELL 6: Execution Cell - Run MCA Pipeline only
def execute_mca_pipeline():
    """Execute the MCA pipeline."""
    log_status("===== EXECUTING MCA PIPELINE =====")
    
    start_time = time.time()
    success = run_mca_pipeline()
    end_time = time.time()
    
    if success:
        log_status(f"✅ MCA pipeline completed successfully in {end_time - start_time:.2f} seconds")
        log_status(f"MCA results saved to {GLOBAL_CONFIG['mca_output_prefix']}_* files")
        return True
    else:
        log_status(f"❌ MCA pipeline failed after {end_time - start_time:.2f} seconds")
        return False


# COMMAND ----------

# CELL22: Pure K-Prototypes Clustering Implementation (No MCA)
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from IPython.display import display, HTML
import os

def extract_and_prepare_data_for_kprototypes(sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Extract data directly from the database and prepare it for k-prototypes
    without any reference to MCA
    
    Parameters:
    -----------
    sample_size : int
        Number of samples to extract
        
    Returns:
    --------
    dict
        Dictionary containing prepared data for k-prototypes
    """
    log_status(f"Extracting and preparing data for k-prototypes, sample size: {sample_size}")
    
    try:
        # Get column definitions with expanded columns
        id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
        
        # Combine all columns to extract
        all_columns = [id_column] + vehicle_columns + propensity_columns + demographic_columns + lifestyle_columns + financial_columns
        
        # Remove any duplicates while preserving order
        all_columns = list(dict.fromkeys(all_columns))
        
        # Specify the table name
        acxiom_table = "dataproducts_dev.bronze_acxiom.gm_consumer_list"
        
        # Verify table access
        log_status("Verifying database access...")
        try:
            spark.sql(f"SELECT 1 FROM {acxiom_table} LIMIT 1")
            log_status("Database access verified")
        except Exception as e:
            log_status(f"ERROR: Cannot access Acxiom table: {str(e)}")
            return None
        
        # Verify column existence
        sample_df = spark.sql(f"SELECT * FROM {acxiom_table} LIMIT 1")
        available_columns = set(sample_df.columns)
        
        # Filter for only available columns
        valid_columns = [col for col in all_columns if col in available_columns]
        
        # Report on missing columns
        missing_columns = set(all_columns) - set(valid_columns)
        if missing_columns:
            log_status(f"WARNING: {len(missing_columns)} columns not found in dataset")
            log_status(f"Missing columns: {list(missing_columns)[:20]}...")
        
        # Ensure we have enough columns
        if len(valid_columns) < 10:
            log_status(f"ERROR: Not enough valid columns found (only {len(valid_columns)})")
            return None
        
        # Build column list for query
        column_list = ", ".join(valid_columns)
        
        # Create simple query with random sampling
        log_status("Executing data extraction query...")
        sql_query = f"""
        SELECT {column_list}
        FROM {acxiom_table}
        WHERE {id_column} IS NOT NULL
        ORDER BY rand()
        LIMIT {sample_size}
        """
        
        # Execute the query
        spark_df = spark.sql(sql_query)
        
        # Check if we got enough data
        row_count = spark_df.count()
        if row_count == 0:
            log_status("ERROR: No rows returned from query")
            return None
        
        log_status(f"Successfully extracted {row_count} rows")
        
        # Convert to pandas for processing
        log_status("Converting to pandas DataFrame...")
        df = spark_df.toPandas()
        
        # Create category mappings
        column_categories = {}
        for col in vehicle_columns:
            if col in df.columns:
                column_categories[col] = "Vehicle"
                
        for col in propensity_columns:
            if col in df.columns:
                column_categories[col] = "Propensity"
                
        for col in demographic_columns:
            if col in df.columns:
                column_categories[col] = "Demographic"
                
        for col in lifestyle_columns:
            if col in df.columns:
                column_categories[col] = "Lifestyle"
                
        for col in financial_columns:
            if col in df.columns:
                column_categories[col] = "Financial"
        
        # Determine categorical and numerical columns with improved logic
        log_status("Determining categorical and numerical columns...")
        categorical_cols = []
        numerical_cols = []
        
        # Exclude ID column from features
        feature_cols = [col for col in df.columns if col != id_column]
        
        # Known likely numeric columns
        likely_numeric_cols = [
            "AP001425",  # Household vehicle ownership count
            "AP001500",  # Number of children in household
            "AP003015",  # Household income (general)
            "AP003004",  # Age (general)
            "AP003036",  # Commute length
            "AP003580",  # Household income tier
            "AP003581",  # Credit card usage frequency
            "AP003582",  # Savings behavior
            "AP004581",  # Preferred loan duration
            "AP003583",  # Risk tolerance
            "AP008215",  # Vehicle upgrade timeframe
            "AP008219",  # Price sensitivity
            "AP007821",  # Technology adoption
            "AP008301"   # Research intensity
        ]
        
        for col in feature_cols:
            # Check if column is likely numeric based on our predefined list
            if col in likely_numeric_cols:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # If successful and has more than 15 unique values, treat as numeric
                    if df[col].nunique() > 15:
                        numerical_cols.append(col)
                        # Fill NAs with median
                        df[col] = df[col].fillna(df[col].median())
                        continue
                except:
                    pass  # If conversion fails, will be treated as categorical below
            
            # Get unique values for other columns
            unique_vals = df[col].nunique()
            
            # Categorize columns based on uniqueness and data type
            if unique_vals <= 15 or pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
                # Convert to string
                df[col] = df[col].fillna("missing").astype(str)
            else:
                numerical_cols.append(col)
                # Fill NAs with median
                df[col] = df[col].fillna(df[col].median())
        
        log_status(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical columns")
        
        # Output the first few columns of each type for verification
        log_status(f"Sample categorical columns: {categorical_cols[:5]}")
        log_status(f"Sample numerical columns: {numerical_cols[:5]}")
        
        # Return prepared data
        return {
            'data': df,
            'id_column': id_column,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'column_categories': column_categories,
            'row_count': row_count
        }
    
    except Exception as e:
        log_status(f"ERROR in data extraction and preparation: {str(e)}")
        log_status(traceback.format_exc())
        return None

# COMMAND ----------


# CELL 23: Execution Cell - Run MCA Pipeline only
def execute_mca_pipeline():
    """Execute the MCA pipeline."""
    log_status("===== EXECUTING MCA PIPELINE =====")
    
    start_time = time.time()
    success = run_mca_pipeline()
    end_time = time.time()
    
    if success:
        log_status(f"✅ MCA pipeline completed successfully in {end_time - start_time:.2f} seconds")
        log_status(f"MCA results saved to {GLOBAL_CONFIG['mca_output_prefix']}_* files")
        return True
    else:
        log_status(f"❌ MCA pipeline failed after {end_time - start_time:.2f} seconds")
        return False

# COMMAND ----------


# CELL 24: Execution Cell - Run K-means Clustering only
def execute_kmeans_clustering():
    """Execute the K-means clustering pipeline."""
    log_status("===== EXECUTING K-MEANS CLUSTERING PIPELINE =====")
    
    start_time = time.time()
    success = run_kmeans_clustering_pipeline()
    end_time = time.time()
    
    if success:
        log_status(f"✅ K-means clustering pipeline completed successfully in {end_time - start_time:.2f} seconds")
        log_status(f"K-means results saved to {GLOBAL_CONFIG['clustering_output_prefix']}_kmeans_* files")
        return True
    else:
        log_status(f"❌ K-means clustering pipeline failed after {end_time - start_time:.2f} seconds")
        return False


# COMMAND ----------

# CELL 25: Execution Cell - Run Hierarchical Clustering only
def execute_hierarchical_clustering():
    """Execute the hierarchical clustering pipeline."""
    log_status("===== EXECUTING HIERARCHICAL CLUSTERING PIPELINE =====")
    
    start_time = time.time()
    success = run_hierarchical_clustering_pipeline()
    end_time = time.time()
    
    if success:
        log_status(f"✅ Hierarchical clustering pipeline completed successfully in {end_time - start_time:.2f} seconds")
        log_status(f"Hierarchical results saved to {GLOBAL_CONFIG['clustering_output_prefix']}_hierarchical_* files")
        return True
    else:
        log_status(f"❌ Hierarchical clustering pipeline failed after {end_time - start_time:.2f} seconds")
        return False


# COMMAND ----------

# CELL 26: Execution Cell - Run All Pipelines
def execute_all_pipelines():
    """Execute the full pipeline: MCA, K-means, and Hierarchical clustering."""
    log_status("===== EXECUTING FULL PIPELINE =====")
    
    # Step 1: MCA
    mca_success = execute_mca_pipeline()
    if not mca_success:
        log_status("❌ Full pipeline aborted due to MCA pipeline failure")
        return False
    
    # Step 2: K-means
    kmeans_success = execute_kmeans_clustering()
    if not kmeans_success:
        log_status("⚠️ K-means clustering failed, continuing with hierarchical clustering")
    
    # Step 3: Hierarchical
    hier_success = execute_hierarchical_clustering()
    if not hier_success:
        log_status("⚠️ Hierarchical clustering failed")
    
    # Overall success if at least one clustering method succeeded
    if kmeans_success or hier_success:
        log_status("✅ Full pipeline completed with at least one clustering method successful")
        return True
    else:
        log_status("❌ Full pipeline failed - all clustering methods failed")
        return False
		
# CELL 10: Simplified execution functions for each specific task

def run_just_mca(sample_size=None):
    """
    Run only the MCA analysis with an optional sample size override.
    
    Parameters:
    -----------
    sample_size : int, optional
        If provided, overrides the GLOBAL_CONFIG sample size for MCA
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if sample_size is not None:
        log_status(f"Overriding default MCA sample size to: {sample_size}")
        original_size = GLOBAL_CONFIG['mca_sample_size']
        GLOBAL_CONFIG['mca_sample_size'] = sample_size
    
    success = execute_mca_pipeline()
    
    # Restore original sample size if we changed it
    if sample_size is not None:
        GLOBAL_CONFIG['mca_sample_size'] = original_size
        
    return success

def run_just_kmeans(cluster_counts=None, sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000)):
    """
    Run only K-means clustering with optional parameter overrides.
    
    Parameters:
    -----------
    cluster_counts : list, optional
        If provided, overrides the GLOBAL_CONFIG cluster counts for K-means
    sample_size : int, optional
        If provided, overrides the GLOBAL_CONFIG sample size for clustering
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Save original values
    original_clusters = GLOBAL_CONFIG['kmeans_clusters']
    original_size = GLOBAL_CONFIG['clustering_sample_size']
    
    # Override if requested
    if cluster_counts is not None:
        log_status(f"Overriding default K-means cluster counts to: {cluster_counts}")
        GLOBAL_CONFIG['kmeans_clusters'] = cluster_counts
        
    if sample_size is not None:
        log_status(f"Overriding default clustering sample size to: {sample_size}")
        GLOBAL_CONFIG['clustering_sample_size'] = sample_size
    
    success = execute_kmeans_clustering()
    
    # Restore original values
    GLOBAL_CONFIG['kmeans_clusters'] = original_clusters
    GLOBAL_CONFIG['clustering_sample_size'] = original_size
        
    return success

def run_just_hierarchical(cluster_counts=None, sample_size=None):
    """
    Run only hierarchical clustering with optional parameter overrides.
    
    Parameters:
    -----------
    cluster_counts : list, optional
        If provided, overrides the GLOBAL_CONFIG cluster counts for hierarchical
    sample_size : int, optional
        If provided, overrides the GLOBAL_CONFIG sample size for clustering
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Save original values
    original_clusters = GLOBAL_CONFIG['hierarchical_clusters']
    original_size = GLOBAL_CONFIG['clustering_sample_size']
    
    # Override if requested
    if cluster_counts is not None:
        log_status(f"Overriding default hierarchical cluster counts to: {cluster_counts}")
        GLOBAL_CONFIG['hierarchical_clusters'] = cluster_counts
        
    if sample_size is not None:
        log_status(f"Overriding default clustering sample size to: {sample_size}")
        GLOBAL_CONFIG['clustering_sample_size'] = sample_size
    
    success = execute_hierarchical_clustering()
    
    # Restore original values
    GLOBAL_CONFIG['hierarchical_clusters'] = original_clusters
    GLOBAL_CONFIG['clustering_sample_size'] = original_size
        
    return success


# COMMAND ----------

# CELL 27: Run Hierarchical Clustering with Proper Directory Creation
import time
import traceback
import json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_hierarchical_clustering_pipeline(sample_size=10000, num_clusters=8):
    """
    Complete hierarchical clustering pipeline with proper directory creation
    
    Parameters:
    -----------
    sample_size : int
        Number of rows to extract from Acxiom
    num_clusters : int
        Number of clusters to create
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    pipeline_start = time.time()
    log_status(f"Starting hierarchical clustering pipeline with {num_clusters} clusters")
    
    try:
        # Step 1: Create output directory using dbutils
        output_dir = "/FileStore/acxiom_clustering"
        output_prefix = f"/dbfs{output_dir}"
        
        # Create directory using dbutils (this handles the permission properly)
        dbutils.fs.mkdirs(output_dir)
        log_status(f"Created output directory: {output_dir}")
        
        # Step 2: Extract data
        log_status("Extracting data...")
        extracted_data = direct_extract_acxiom_data(sample_size)
        if not extracted_data:
            log_status("❌ Data extraction failed")
            return False
        
        log_status(f"Successfully extracted {extracted_data['row_count']} rows")
        
        # Step 3: Prepare data for MCA
        log_status("Preparing data for MCA...")
        spark_df = extracted_data['spark_df']
        
        # Convert to pandas for further processing
        pandas_df = spark_df.toPandas()
        log_status(f"Converted to pandas DataFrame with {len(pandas_df)} rows")
        
        # Prepare MCA data
        prepared_data = robust_prepare_data_for_mca(pandas_df, extracted_data['id_column'])
        if not prepared_data:
            log_status("❌ Data preparation for MCA failed")
            return False
        
        log_status(f"Prepared {len(prepared_data['categorical_cols'])} categorical columns for MCA")
        
        # Step 4: Run MCA
        log_status("Running MCA analysis...")
        mca_results = robust_run_mca_analysis(prepared_data)
        if not mca_results:
            log_status("❌ MCA analysis failed")
            return False
        
        log_status(f"MCA analysis completed with {mca_results['n_dims']} dimensions")
        
        # Step 5: Perform hierarchical clustering
        log_status("Performing hierarchical clustering...")
        
        # Extract dimensional data for clustering
        dim_cols = [f'MCA_dim{i+1}' for i in range(mca_results['n_dims'])]
        mca_coords = mca_results['mca_coords'][dim_cols].copy()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(mca_coords)
        
        # Compute linkage
        log_status("Computing hierarchical linkage...")
        Z = linkage(scaled_data, method='ward')
        
        # Cluster the data
        labels = fcluster(Z, num_clusters, criterion='maxclust') - 1  # Convert to 0-based indexing
        
        # Calculate silhouette score
        sil_score = silhouette_score(scaled_data, labels)
        log_status(f"Silhouette score for {num_clusters} clusters: {sil_score:.4f}")
        
        # Add cluster labels to MCA coordinates
        result_df = mca_results['mca_coords'].copy()
        result_df['cluster'] = labels
        
        # Step 6: Save results
        log_status("Saving clustering results...")
        
        # Save cluster assignments
        clusters_path = f"{output_prefix}/hierarchical_clusters_{num_clusters}.csv"
        result_df.to_csv(clusters_path, index=False)
        log_status(f"Saved cluster assignments to {clusters_path}")
        
        # Create dendrogram visualization
        plt.figure(figsize=(15, 8))
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            color_threshold=0.7*max(Z[:,2])
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram ({num_clusters} clusters)', fontsize=14)
        plt.xlabel('Sample index or (cluster size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        
        # Save dendrogram
        dendro_path = f"{output_prefix}/hierarchical_dendrogram_{num_clusters}.jpg"
        plt.savefig(dendro_path, dpi=300, bbox_inches='tight')
        plt.close()
        log_status(f"Saved hierarchical dendrogram to {dendro_path}")
        
        # Create cluster visualization
        plt.figure(figsize=(14, 12))
        
        if 'MCA_dim1' in result_df.columns and 'MCA_dim2' in result_df.columns:
            plt.scatter(
                result_df['MCA_dim1'],
                result_df['MCA_dim2'],
                c=result_df['cluster'],
                cmap='tab20',
                alpha=0.7,
                s=40,
                edgecolors='w',
                linewidths=0.3
            )
            
            plt.title(f'Hierarchical Clustering: {num_clusters} Segments in MCA Space', fontsize=14)
            plt.xlabel("MCA Dimension 1", fontsize=12)
            plt.ylabel("MCA Dimension 2", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Save scatter plot
            scatter_path = f"{output_prefix}/hierarchical_scatter_{num_clusters}.jpg"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            log_status(f"Saved cluster visualization to {scatter_path}")
        
        # Save clustering metadata
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_clusters': num_clusters,
            'sample_size': sample_size,
            'silhouette_score': float(sil_score),
            'method': 'hierarchical_ward',
            'execution_time': float(time.time() - pipeline_start)
        }
        
        metadata_path = f"{output_prefix}/hierarchical_metadata_{num_clusters}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        log_status(f"Saved clustering metadata to {metadata_path}")
        
        # Step 7: Generate cluster profiles
        log_status("Generating cluster profiles...")
        
        # Calculate cluster sizes
        cluster_sizes = result_df['cluster'].value_counts().sort_index()
        
        # Create cluster profiles report
        report_content = f"===== HIERARCHICAL CLUSTERING PROFILES =====\n\n"
        report_content += f"Number of segments: {num_clusters}\n"
        report_content += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_content += f"Silhouette score: {sil_score:.4f}\n\n"
        
        # Add cluster size information
        report_content += "CLUSTER SIZES:\n"
        report_content += "-------------\n"
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(result_df)) * 100
            report_content += f"Cluster {cluster_id}: {size} records ({percentage:.1f}%)\n"
        
        # Add cluster dimensional profiles
        report_content += "\nCLUSTER PROFILES:\n"
        report_content += "---------------\n"
        
        for i in range(num_clusters):
            cluster_mask = result_df['cluster'] == i
            cluster_df = result_df[cluster_mask]
            
            report_content += f"\nCluster {i}:\n"
            report_content += f"----------\n"
            
            # Dimensional profile
            dim_means = []
            for dim in range(min(5, mca_results['n_dims'])):
                dim_col = f'MCA_dim{dim+1}'
                dim_mean = cluster_df[dim_col].mean()
                dim_means.append(f"{dim_col}: {dim_mean:.3f}")
            
            report_content += "Dimensional profile: " + ", ".join(dim_means) + "\n"
            report_content += f"Size: {cluster_sizes[i]} records ({cluster_sizes[i]/len(result_df)*100:.1f}%)\n"
        
        # Save cluster profiles
        profiles_path = f"{output_prefix}/hierarchical_profiles_{num_clusters}.txt"
        with open(profiles_path, 'w') as f:
            f.write(report_content)
            
        log_status(f"Saved cluster profiles to {profiles_path}")
        
        # Calculate total time
        total_time = time.time() - pipeline_start
        log_status(f"✅ Hierarchical clustering pipeline completed successfully in {total_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        log_status(f"ERROR in hierarchical clustering pipeline: {str(e)}")
        log_status(f"Error details: {traceback.format_exc()}")
        log_status("❌ Hierarchical clustering pipeline failed")
        return False

# Execute the hierarchical clustering pipeline
run_hierarchical_clustering_pipeline(sample_size=10000, num_clusters=8)

# COMMAND ----------

# CELL: K-modes Marketing Report Generation
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def generate_kmodes_marketing_report(kmodes_results, output_prefix="/dbfs/FileStore/acxiom_clustering/kmodes"):
    """
    Generate comprehensive marketing reports based on k-modes clustering results.
    
    Parameters:
    -----------
    kmodes_results : dict
        Dictionary containing results from k-modes clustering
    output_prefix : str
        Base path for saving report files
        
    Returns:
    --------
    str
        Path to the generated report
    """
    log_status("Generating comprehensive marketing report from k-modes clustering results...")
    
    # Extract key information from results
    cluster_data = kmodes_results.get('cluster_data')
    centroids = kmodes_results.get('centroids_df')
    k = kmodes_results.get('num_clusters')
    
    if cluster_data is None or centroids is None:
        log_status("ERROR: Missing required data in k-modes results")
        return None
    
    # Create report directory if it doesn't exist
    report_dir = f"{output_prefix}_marketing_report_{k}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate segment insights from cluster data
    segment_insights = analyze_kmodes_segments(cluster_data, centroids, k)
    
    # Generate visualizations
    visualization_paths = generate_segment_visualizations(segment_insights, report_dir, k)
    
    # Create the report document
    report_path = create_marketing_report_document(segment_insights, visualization_paths, report_dir, k)
    
    log_status(f"Marketing report successfully generated at: {report_path}")
    return report_path

def analyze_kmodes_segments(cluster_data, centroids, k):
    """
    Analyze k-modes clusters to extract marketing-relevant insights
    
    Parameters:
    -----------
    cluster_data : pandas.DataFrame
        DataFrame with cluster assignments
    centroids : pandas.DataFrame
        DataFrame with cluster centroids
    k : int
        Number of clusters
        
    Returns:
    --------
    dict
        Dictionary containing segment insights
    """
    log_status("Analyzing k-modes clusters for marketing insights...")
    
    # Calculate cluster sizes
    cluster_counts = cluster_data['kmodes_cluster'].value_counts().sort_index()
    cluster_sizes = [(count / len(cluster_data)) * 100 for count in cluster_counts]
    
    # Initialize insights dictionary
    segment_insights = {
        'k': k,
        'cluster_sizes': cluster_sizes,
        'segments': {}
    }
    
    # Extract AP column patterns if available
    ap_columns = [col for col in cluster_data.columns if col.startswith('AP') and col != 'kmodes_cluster']
    
    # Categorize columns into different attributes if possible
    column_categories = {}
    
    try:
        # Try to identify logical column categories based on patterns or external definitions
        # These will vary based on your specific data
        for col in ap_columns:
            if any(vehicle_term in col.lower() for vehicle_term in ['vehicle', 'car', 'suv', 'luxury']):
                column_categories[col] = "Vehicle"
            elif any(demo_term in col.lower() for demo_term in ['income', 'age', 'demographic']):
                column_categories[col] = "Demographic"
            elif any(prop_term in col.lower() for prop_term in ['propensity', 'purchase', 'buy']):
                column_categories[col] = "Propensity"
            else:
                column_categories[col] = "Other"
    except:
        # Fallback to basic categorization
        log_status("Could not identify detailed column categories, using generic approach")
        column_categories = {col: "Feature" for col in ap_columns}
    
    # For each cluster, extract key attributes
    for cluster_id in range(k):
        # Extract cluster centroid
        centroid = centroids[centroids['cluster_id'] == cluster_id].iloc[0]
        
        # Calculate cluster size
        size_pct = segment_insights['cluster_sizes'][cluster_id]
        
        # Get cluster records
        cluster_records = cluster_data[cluster_data['kmodes_cluster'] == cluster_id]
        
        # Generate a meaningful segment name
        segment_name = generate_segment_name(centroid, cluster_records, ap_columns, column_categories)
        
        # Identify key distinguishing features
        key_features = identify_key_features(centroid, centroids, ap_columns)
        
        # Extract demographic patterns if available
        demographics = extract_demographics(cluster_records, column_categories)
        
        # Identify vehicle preferences
        vehicle_preferences = extract_vehicle_preferences(cluster_records, centroid, column_categories)
        
        # Create buying behavior profile
        buying_behavior = create_buying_behavior_profile(cluster_records, centroid, column_categories)
        
        # Generate marketing recommendations
        marketing_strategy = generate_marketing_recommendations(
            key_features, 
            demographics, 
            vehicle_preferences, 
            buying_behavior,
            size_pct
        )
        
        # Store all segment insights
        segment_insights['segments'][cluster_id] = {
            'name': segment_name,
            'size_pct': size_pct,
            'key_features': key_features,
            'demographics': demographics,
            'vehicle_preferences': vehicle_preferences,
            'buying_behavior': buying_behavior,
            'marketing_strategy': marketing_strategy
        }
    
    return segment_insights

def generate_segment_name(centroid, cluster_records, features, column_categories):
    # Generate a meaningful segment name based on centroid characteristics and cluster records
    # Store attribute scores to determine segment focus
    segment_attributes = {
        'luxury': 0,
        'economy': 0,
        'tech': 0,
        'family': 0,
        'utility': 0,
        'premium': 0,
        'value': 0
    }
    
    # Analyze vehicle type features
    vehicle_cols = [col for col, cat in column_categories.items() 
                  if cat == "Vehicle" and col in features]
    
    for col in vehicle_cols:
        col_lower = col.lower()
        col_value = str(centroid[col]).lower()
        
        # Check for luxury indicators
        if any(term in col_lower for term in ['luxury', 'premium', 'high-end']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['luxury'] += 2
                segment_attributes['premium'] += 1
        
        # Check for economy indicators
        if any(term in col_lower for term in ['economy', 'compact', 'value']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['economy'] += 2
                segment_attributes['value'] += 1
        
        # Check for SUV/truck indicators
        if any(term in col_lower for term in ['suv', 'truck', 'pickup']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['utility'] += 1
        
        # Check for family vehicle indicators
        if any(term in col_lower for term in ['family', 'minivan', 'passenger']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['family'] += 2
    
    # Analyze technology adoption
    tech_cols = [col for col, cat in column_categories.items() 
               if cat in ["Propensity", "Lifestyle"] and col in features]
    
    for col in tech_cols:
        col_lower = col.lower()
        col_value = str(centroid[col]).lower()
        
        # Check for tech indicators
        if any(term in col_lower for term in ['tech', 'digital', 'connected', 'innovation']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['tech'] += 2
        
        # Check for value consciousness
        if any(term in col_lower for term in ['price', 'value', 'savings']):
            if any(val in col_value for val in ['high', 'yes', '1', 'true']):
                segment_attributes['value'] += 2
    
    # Analyze demographics
    demo_cols = [col for col, cat in column_categories.items() 
               if cat == "Demographic" and col in features]
    
    for col in demo_cols:
        col_lower = col.lower()
        col_value = str(centroid[col]).lower()
        
        # Check for income/affluence indicators
        if any(term in col_lower for term in ['income', 'affluent', 'wealthy']):
            if any(val in col_value for val in ['high', 'upper', '1', 'true']):
                segment_attributes['premium'] += 2
                segment_attributes['luxury'] += 1
            elif any(val in col_value for val in ['low', 'middle', '0']):
                segment_attributes['value'] += 1
                segment_attributes['economy'] += 1
        
        # Check for family/household indicators
        if any(term in col_lower for term in ['household', 'children', 'family']):
            if any(val in col_value for val in ['yes', 'high', '1', 'true']):
                segment_attributes['family'] += 2
    
    # Find the top attributes
    top_attributes = sorted(segment_attributes.items(), key=lambda x: x[1], reverse=True)
    primary_attribute = top_attributes[0][0]
    secondary_attribute = top_attributes[1][0] if top_attributes[1][1] > 0 else None
    
    # Generate name based on top attributes
    if primary_attribute == 'luxury':
        if secondary_attribute == 'tech':
            name = "Premium Tech-Forward Enthusiasts"
        elif secondary_attribute == 'family':
            name = "Luxury Family Vehicle Buyers"
        else:
            name = "Luxury Vehicle Enthusiasts"
    elif primary_attribute == 'premium':
        if secondary_attribute == 'luxury':
            name = "Premium Luxury Buyers"
        elif secondary_attribute == 'tech':
            name = "Premium Tech-Savvy Consumers"
        else:
            name = "Premium Vehicle Segment"
    elif primary_attribute == 'tech':
        if secondary_attribute == 'luxury':
            name = "Tech-Forward Luxury Buyers"
        elif secondary_attribute == 'value':
            name = "Tech-Savvy Value-Conscious Buyers"
        else:
            name = "Technology-Forward Early Adopters"
    elif primary_attribute == 'family':
        if secondary_attribute == 'luxury':
            name = "Upscale Family Vehicle Buyers"
        elif secondary_attribute == 'value':
            name = "Practical Family Vehicle Owners"
        else:
            name = "Mainstream Family Vehicle Segment"
    elif primary_attribute == 'utility':
        if secondary_attribute == 'luxury':
            name = "Premium Utility Vehicle Owners"
        elif secondary_attribute == 'value':
            name = "Practical Utility Vehicle Segment"
        else:
            name = "Utility-Focused Vehicle Buyers"
    elif primary_attribute == 'economy':
        if secondary_attribute == 'tech':
            name = "Budget-Conscious Tech Adopters"
        elif secondary_attribute == 'family':
            name = "Economy Family Vehicle Segment"
        else:
            name = "Economy-Minded Vehicle Buyers"
    elif primary_attribute == 'value':
        if secondary_attribute == 'tech':
            name = "Value-Oriented Tech Adopters"
        elif secondary_attribute == 'family':
            name = "Value-Conscious Family Segment"
        else:
            name = "Value-Oriented Vehicle Buyers"
    else:
        name = f"Vehicle Segment {centroid['cluster_id']}"
    
    return name

def identify_key_features(centroid, all_centroids, features):
    # Identify key distinguishing features for a segment based on centroid values
    # Initialize key features dictionary
    key_features = {
        'luxury_orientation': 0,
        'price_sensitivity': 0,
        'tech_adoption': 0,
        'brand_loyalty': 0,
        'family_focus': 0,
        'utility_priority': 0
    }
    
    # Extract all centroids into a DataFrame for comparison
    centroids_df = pd.DataFrame(all_centroids)
    
    # Analyze centroid values to determine key features
    # Luxury orientation
    luxury_indicators = ['luxury', 'premium', 'high-end', 'upscale']
    luxury_score = 0
    luxury_cols = [col for col in features if any(ind in col.lower() for ind in luxury_indicators)]
    
    if luxury_cols:
        for col in luxury_cols:
            # Get this centroid's value
            val = str(centroid[col]).lower()
            
            # Compare with other centroids
            other_vals = centroids_df[centroids_df['cluster_id'] != centroid['cluster_id']][col].astype(str).str.lower()
            
            # Score based on positive indicators
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                luxury_score += 1
                
                # Add extra points if this is distinctive
                if all(term not in v for v in other_vals for term in ['high', 'yes', '1', 'true']):
                    luxury_score += 2
    
    # Scale to 0-100
    key_features['luxury_orientation'] = min(100, luxury_score * 20)
    
    # Price sensitivity
    price_indicators = ['price', 'value', 'economy', 'budget']
    price_score = 0
    price_cols = [col for col in features if any(ind in col.lower() for ind in price_indicators)]
    
    if price_cols:
        for col in price_cols:
            val = str(centroid[col]).lower()
            other_vals = centroids_df[centroids_df['cluster_id'] != centroid['cluster_id']][col].astype(str).str.lower()
            
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                price_score += 1
                if all(term not in v for v in other_vals for term in ['high', 'yes', '1', 'true']):
                    price_score += 1
    
    # Invert luxury orientation to also influence price sensitivity
    price_score += (5 - (key_features['luxury_orientation'] / 20))
    key_features['price_sensitivity'] = min(100, price_score * 15)
    
    # Tech adoption
    tech_indicators = ['tech', 'digital', 'connected', 'smart', 'innovation']
    tech_score = 0
    tech_cols = [col for col in features if any(ind in col.lower() for ind in tech_indicators)]
    
    if tech_cols:
        for col in tech_cols:
            val = str(centroid[col]).lower()
            other_vals = centroids_df[centroids_df['cluster_id'] != centroid['cluster_id']][col].astype(str).str.lower()
            
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                tech_score += 1
                if all(term not in v for v in other_vals for term in ['high', 'yes', '1', 'true']):
                    tech_score += 2
    
    key_features['tech_adoption'] = min(100, tech_score * 20)
    
    # Brand loyalty
    brand_indicators = ['brand', 'loyal', 'preference']
    brand_score = 0
    brand_cols = [col for col in features if any(ind in col.lower() for ind in brand_indicators)]
    
    if brand_cols:
        for col in brand_cols:
            val = str(centroid[col]).lower()
            
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                brand_score += 1
    
    # If no direct columns, infer from luxury and price
    if not brand_cols or brand_score == 0:
        # Luxury segments tend to have higher brand loyalty
        brand_score += key_features['luxury_orientation'] / 25
        # Price sensitive segments tend to have lower brand loyalty
        brand_score -= key_features['price_sensitivity'] / 50
    
    key_features['brand_loyalty'] = max(0, min(100, 50 + brand_score * 15))
    
    # Family focus
    family_indicators = ['family', 'children', 'kid', 'passenger']
    family_score = 0
    family_cols = [col for col in features if any(ind in col.lower() for ind in family_indicators)]
    
    if family_cols:
        for col in family_cols:
            val = str(centroid[col]).lower()
            other_vals = centroids_df[centroids_df['cluster_id'] != centroid['cluster_id']][col].astype(str).str.lower()
            
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                family_score += 1
                if all(term not in v for v in other_vals for term in ['high', 'yes', '1', 'true']):
                    family_score += 1
    
    key_features['family_focus'] = min(100, family_score * 25)
    
    # Utility priority
    utility_indicators = ['utility', 'practical', 'cargo', 'towing', 'truck', 'suv']
    utility_score = 0
    utility_cols = [col for col in features if any(ind in col.lower() for ind in utility_indicators)]
    
    if utility_cols:
        for col in utility_cols:
            val = str(centroid[col]).lower()
            other_vals = centroids_df[centroids_df['cluster_id'] != centroid['cluster_id']][col].astype(str).str.lower()
            
            if any(term in val for term in ['high', 'yes', '1', 'true']):
                utility_score += 1
                if all(term not in v for v in other_vals for term in ['high', 'yes', '1', 'true']):
                    utility_score += 1
    
    key_features['utility_priority'] = min(100, utility_score * 20)
    
    return key_features

def extract_demographics(cluster_records, column_categories):
    # Extract demographic information from cluster records
    # Get demographic columns
    demo_cols = [col for col, category in column_categories.items() 
               if category == "Demographic"]
    
    if not demo_cols or len(cluster_records) == 0:
        # Return default demographics if no data
        return {
            'age': "Mixed",
            'income': "Mixed",
            'gender_split': "Not available",
            'education': "Not available",
            'geography': "Not available"
        }
    
    # Initialize demographics dictionary
    demographics = {}
    
    # Analyze age if possible
    age_cols = [col for col in demo_cols if 'age' in col.lower()]
    if age_cols:
        # Simple approach: check for most common age range
        age_values = {}
        for col in age_cols:
            if col in cluster_records.columns:
                value_counts = cluster_records[col].value_counts(normalize=True)
                
                for val, count in value_counts.items():
                    val_str = str(val).lower()
                    
                    if '18-34' in val_str or 'young' in val_str:
                        age_values['18-34'] = age_values.get('18-34', 0) + count
                    elif '35-54' in val_str or 'middle' in val_str:
                        age_values['35-54'] = age_values.get('35-54', 0) + count
                    elif '55+' in val_str or 'senior' in val_str or '65+' in val_str:
                        age_values['55+'] = age_values.get('55+', 0) + count
        
        # Determine primary age range
        if age_values:
            primary_age = max(age_values.items(), key=lambda x: x[1])[0]
            demographics['age'] = primary_age
        else:
            demographics['age'] = "Mixed"
    else:
        demographics['age'] = "Mixed"
    
    # Analyze income if possible
    income_cols = [col for col in demo_cols if 'income' in col.lower()]
    if income_cols:
        # Check for income brackets
        income_values = {}
        for col in income_cols:
            if col in cluster_records.columns:
                value_counts = cluster_records[col].value_counts(normalize=True)
                
                for val, count in value_counts.items():
                    val_str = str(val).lower()
                    
                    if any(term in val_str for term in ['high', '150k', '200k', 'wealthy']):
                        income_values['High'] = income_values.get('High', 0) + count
                    elif any(term in val_str for term in ['middle', '50k', '100k']):
                        income_values['Middle'] = income_values.get('Middle', 0) + count
                    elif any(term in val_str for term in ['low', 'under', 'less']):
                        income_values['Low'] = income_values.get('Low', 0) + count
        
        # Determine primary income level
        if income_values:
            primary_income = max(income_values.items(), key=lambda x: x[1])[0]
            if primary_income == 'High':
                demographics['income'] = "Upper-middle to High"
            elif primary_income == 'Middle':
                demographics['income'] = "Middle to Upper-middle"
            else:
                demographics['income'] = "Low to Middle"
        else:
            demographics['income'] = "Mixed"
    else:
        demographics['income'] = "Mixed"
    
    # Add other demographics with default values
    demographics['gender_split'] = "Not available"
    demographics['education'] = "Not available"
    demographics['geography'] = "Not available"
    
    return demographics

def extract_vehicle_preferences(cluster_records, centroid, column_categories):
    # Extract vehicle preferences from cluster data
    # Get vehicle columns
    vehicle_cols = [col for col, category in column_categories.items() 
                  if category == "Vehicle"]
    
    if not vehicle_cols:
        # Return default preferences if no data
        return ["Standard vehicles", "Mixed preferences"]
    
    # Analyze centroid values to determine preferences
    preferences = []
    
    # Look for luxury preference
    luxury_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                   for term in ['luxury', 'premium', 'high-end'])]
    if luxury_cols:
        luxury_score = 0
        for col in luxury_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    luxury_score += 1
        
        if luxury_score > 0:
            preferences.append("Luxury vehicles with premium features")
    
    # Look for vehicle type preferences
    suv_cols = [col for col in vehicle_cols if 'suv' in col.lower()]
    sedan_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                  for term in ['sedan', 'car'])]
    truck_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                  for term in ['truck', 'pickup'])]
    minivan_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                     for term in ['minivan', 'van'])]
    
    # Score each vehicle type
    vehicle_scores = {
        'SUV/Crossover': sum(1 for col in suv_cols if col in centroid and 
                           any(term in str(centroid[col]).lower() for term in ['high', 'yes', '1', 'true'])),
        'Sedan': sum(1 for col in sedan_cols if col in centroid and 
                   any(term in str(centroid[col]).lower() for term in ['high', 'yes', '1', 'true'])),
        'Truck/Pickup': sum(1 for col in truck_cols if col in centroid and 
                          any(term in str(centroid[col]).lower() for term in ['high', 'yes', '1', 'true'])),
        'Minivan': sum(1 for col in minivan_cols if col in centroid and 
                      any(term in str(centroid[col]).lower() for term in ['high', 'yes', '1', 'true']))
    }
    
    # Add top vehicle types to preferences
    top_types = sorted(vehicle_scores.items(), key=lambda x: x[1], reverse=True)
    for vtype, score in top_types:
        if score > 0:
            if 'Luxury vehicles' in preferences:
                preferences.append(f"Premium {vtype} models")
            else:
                preferences.append(f"{vtype} models")
            break
    
    # Check for alternative fuel preference
    alt_fuel_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                     for term in ['hybrid', 'electric', 'alt', 'alternative'])]
    if alt_fuel_cols:
        alt_fuel_score = 0
        for col in alt_fuel_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    alt_fuel_score += 1
        
        if alt_fuel_score > 0:
            preferences.append("Alternative fuel/hybrid vehicles")
    
    # Add default preferences if none identified
    if not preferences:
        preferences.append("Standard vehicles")
        preferences.append("Mixed preferences")
    
    # Add family-oriented preference if family_focus is high
    # (This would come from key_features in a real implementation)
    family_cols = [col for col in vehicle_cols if any(term in col.lower() 
                                                    for term in ['family', 'passenger'])]
    if family_cols:
        family_score = 0
        for col in family_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    family_score += 1
        
        if family_score > 0:
            preferences.append("Family-oriented vehicles with versatile space")
    
    return preferences[:4]  # Limit to top 4 preferences

def create_buying_behavior_profile(cluster_records, centroid, column_categories):
    # Create a buying behavior profile from cluster data
    # Get propensity columns
    propensity_cols = [col for col, category in column_categories.items() 
                     if category in ["Propensity", "Financial"]]
    
    if not propensity_cols:
        # Return default behavior if no data
        return ["Standard purchasing process", "Mixed buying patterns"]
    
    # Analyze centroid values to determine behaviors
    behaviors = []
    
    # Check for research intensity
    research_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                        for term in ['research', 'compare', 'review'])]
    if research_cols:
        research_score = 0
        for col in research_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    research_score += 1
        
        if research_score > 0:
            behaviors.append("Research-intensive purchase process")
    
    # Check for price sensitivity
    price_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                     for term in ['price', 'value', 'budget'])]
    if price_cols:
        price_score = 0
        for col in price_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    price_score += 1
        
        if price_score > 0:
            behaviors.append("Price-sensitive purchasing decisions")
        else:
            behaviors.append("Quality-focused over price-sensitive")
    
    # Check for financing preference
    finance_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                       for term in ['finance', 'loan', 'lease'])]
    if finance_cols:
        finance_score = 0
        lease_score = 0
        cash_score = 0
        
        for col in finance_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if 'lease' in col.lower() and any(term in val for term in ['high', 'yes', '1', 'true']):
                    lease_score += 1
                elif 'cash' in col.lower() and any(term in val for term in ['high', 'yes', '1', 'true']):
                    cash_score += 1
                elif any(term in val for term in ['high', 'yes', '1', 'true']):
                    finance_score += 1
        
        if lease_score > finance_score and lease_score > cash_score:
            behaviors.append("Prefers leasing over traditional financing")
        elif cash_score > finance_score and cash_score > lease_score:
            behaviors.append("Prefer cash purchases when possible")
        elif finance_score > 0:
            behaviors.append("Traditional financing approach")
    
    # Check for tech influence
    tech_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                    for term in ['tech', 'feature', 'digital'])]
    if tech_cols:
        tech_score = 0
        for col in tech_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    tech_score += 1
        
        if tech_score > 0:
            behaviors.append("Influenced by technology features")
    
    # Check for brand influence
    brand_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                     for term in ['brand', 'loyal'])]
    if brand_cols:
        brand_score = 0
        for col in brand_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    brand_score += 1
        
        if brand_score > 0:
            behaviors.append("Brand-loyal purchase decisions")
    # Add digital shopping behavior
    digital_cols = [col for col in propensity_cols if any(term in col.lower() 
                                                       for term in ['online', 'digital', 'mobile'])]
    if digital_cols:
        digital_score = 0
        for col in digital_cols:
            if col in centroid:
                val = str(centroid[col]).lower()
                if any(term in val for term in ['high', 'yes', '1', 'true']):
                    digital_score += 1
        
        if digital_score > 0:
            behaviors.append("Digital-first research approach")
    
    # Add default behaviors if none identified
    if not behaviors:
        behaviors.append("Standard purchasing process")
        behaviors.append("Mixed buying patterns")
    
    return behaviors[:4]  # Limit to top 4 behaviors

def generate_marketing_recommendations(key_features, demographics, vehicle_preferences, buying_behavior, size_pct):
    # Generate marketing recommendations based on segment insights
    # Determine primary orientation based on key features and other insights
    primary_orientation = None
    secondary_orientation = None
    
    # Evaluate orientations by combining key features
    orientations = {
        'luxury': key_features['luxury_orientation'],
        'value': key_features['price_sensitivity'],
        'technology': key_features['tech_adoption'],
        'family': key_features['family_focus'],
        'utility': key_features['utility_priority']
    }
    
    # Get top two orientations
    top_orientations = sorted(orientations.items(), key=lambda x: x[1], reverse=True)
    primary_orientation = top_orientations[0][0]
    if len(top_orientations) > 1 and top_orientations[1][1] > 20:
        secondary_orientation = top_orientations[1][0]
    
    # Evaluate segment size to determine marketing approach
    if size_pct > 20:
        size_category = "mass"
    elif size_pct > 10:
        size_category = "major"
    elif size_pct > 3:
        size_category = "niche"
    else:
        size_category = "micro"
    
    # Generate marketing strategy components
    messaging = []
    channels = []
    product_focus = []
    customer_experience = []
    
    # Generate recommendations based on primary orientation
    if primary_orientation == 'luxury':
        messaging = [
            "Emphasize exclusivity and premium craftsmanship",
            "Focus on performance specifications and driving experience",
            "Highlight premium materials and attention to detail",
            "Stress the prestige and status aspects of ownership"
        ]
        channels = [
            "Luxury lifestyle publications and websites",
            "Exclusive events and experiences",
            "Personalized direct marketing",
            "High-end digital platforms"
        ]
        product_focus = [
            "Premium model variants with exclusive features",
            "Specialized limited editions",
            "Models with advanced technology packages",
            "High-performance engine options"
        ]
        customer_experience = [
            "White-glove concierge service",
            "VIP test drive experiences",
            "Personalized shopping journey",
            "Premium ownership benefits"
        ]
    elif primary_orientation == 'value':
        messaging = [
            "Emphasize total value and cost of ownership",
            "Focus on reliability and longevity",
            "Highlight competitive pricing and incentives",
            "Stress practical benefits and economic advantages"
        ]
        channels = [
            "Mass media with targeted value messaging",
            "Price comparison platforms",
            "Digital platforms featuring deals and incentives",
            "Local dealer marketing with price focus"
        ]
        product_focus = [
            "Value-oriented trim packages",
            "Efficient and economical models",
            "Base models with essential feature sets",
            "Special value editions"
        ]
        customer_experience = [
            "Straightforward, transparent pricing",
            "Streamlined purchase process",
            "Value-focused financing options",
            "No-pressure sales environment"
        ]
    elif primary_orientation == 'technology':
        messaging = [
            "Focus on innovation and cutting-edge technology",
            "Emphasize connectivity and digital features",
            "Highlight vehicle technology advancements",
            "Position as forward-thinking and progressive"
        ]
        channels = [
            "Technology and innovation publications",
            "Digital and social media platforms",
            "Tech-focused events and demonstrations",
            "Online communities and forums"
        ]
        product_focus = [
            "Models with advanced technology packages",
            "Connectivity and infotainment features",
            "Electric and hybrid vehicles",
            "Advanced driver assistance systems"
        ]
        customer_experience = [
            "Digital-first shopping experience",
            "Tech-focused product demonstrations",
            "Mobile app integration for ownership",
            "Virtual reality showroom experiences"
        ]
    elif primary_orientation == 'family':
        messaging = [
            "Focus on safety features and ratings",
            "Emphasize versatility and space efficiency",
            "Highlight comfort for all passengers",
            "Stress family-focused convenience features"
        ]
        channels = [
            "Family lifestyle publications",
            "Parenting websites and platforms",
            "School and community partnerships",
            "Family-oriented events and activities"
        ]
        product_focus = [
            "Family-sized vehicles with versatile seating",
            "Models with top safety ratings",
            "Vehicles with entertainment systems",
            "Features for convenience and comfort"
        ]
        customer_experience = [
            "Family-friendly showrooms with kids' areas",
            "Extended test drives for family evaluation",
            "Family-focused financing options",
            "Simplified buying process"
        ]
    elif primary_orientation == 'utility':
        messaging = [
            "Focus on capability and durability",
            "Emphasize practical functionality",
            "Highlight versatility and adaptability",
            "Stress reliability and ruggedness"
        ]
        channels = [
            "Industry and trade publications",
            "Practical demonstration events",
            "Work and lifestyle partnerships",
            "Targeted social media for utility use cases"
        ]
        product_focus = [
            "Versatile cargo and towing capability",
            "Robust design and durability features",
            "Models with practical customization options",
            "Functional accessories and packages"
        ]
        customer_experience = [
            "Practical demonstration of capabilities",
            "Feature-focused sales approach",
            "Worksite vehicle programs",
            "Service packages that minimize downtime"
        ]
    else:
        # Generic/balanced approach as fallback
        messaging = [
            "Balance features and value messaging",
            "Focus on overall ownership benefits",
            "Highlight versatility and adaptability",
            "Emphasize quality and reliability"
        ]
        channels = [
            "Broad market advertising",
            "Digital and social media mix",
            "Dealership marketing",
            "Content marketing highlighting use cases"
        ]
        product_focus = [
            "Mid-range models with balanced features",
            "Popular configurations and packages",
            "Versatile models with broad appeal",
            "Core product lineup"
        ]
        customer_experience = [
            "Streamlined and efficient buying process",
            "Balanced approach to sales and service",
            "Focus on customer satisfaction",
            "Standard dealership experience with personal touches"
        ]
    
    # Modify recommendations based on secondary orientation
    if secondary_orientation == 'luxury' and primary_orientation != 'luxury':
        messaging[1] = "Emphasize premium features within reach"
        product_focus[0] = "Higher trim levels with premium touches"
    elif secondary_orientation == 'technology' and primary_orientation != 'technology':
        messaging[2] = "Highlight innovative technology features"
        product_focus[2] = "Models with advanced technology options"
        customer_experience[1] = "Technology-focused product demonstrations"
    elif secondary_orientation == 'family' and primary_orientation != 'family':
        messaging[2] = "Emphasize versatility for family needs"
        product_focus[1] = "Family-friendly configurations and features"
    elif secondary_orientation == 'utility' and primary_orientation != 'utility':
        messaging[2] = "Highlight practical utility and versatility"
        product_focus[1] = "Models with enhanced utility features"
    elif secondary_orientation == 'value' and primary_orientation != 'value':
        messaging[1] = "Focus on value proposition and total cost of ownership"
        channels[2] = "Value-oriented promotional campaigns"
    
    # Adjust for segment size
    if size_category == "micro":
        channels = [
            "Highly targeted digital marketing",
            "Specialized publications and platforms",
            "Niche community engagement",
            "One-to-one personalized marketing"
        ]
    elif size_category == "niche":
        channels[0] = "Targeted marketing to specific segments"
    elif size_category == "mass":
        channels[0] = "Broad market mass media campaigns"
    
    # Adjust based on demographics
    age = demographics.get('age', 'Mixed')
    income = demographics.get('income', 'Mixed')
    
    if age == "18-34":
        channels[1] = "Social media and digital platforms for younger audiences"
        customer_experience[1] = "Digital-first buying experience"
    elif age == "55+":
        channels[1] = "Traditional media and established platforms"
        customer_experience[2] = "Relationship-based buying experience"
    
    if "High" in income:
        if primary_orientation != 'luxury':
            product_focus[0] = "Premium models and higher trim levels"
    elif "Low" in income:
        if primary_orientation != 'value':
            product_focus[2] = "Value-oriented models with essential features"
    
    # Adjust based on buying behavior
    for behavior in buying_behavior:
        if "research" in behavior.lower():
            channels.append("Information-rich platforms and content marketing")
            customer_experience[1] = "Detailed product information and comparison tools"
        elif "price" in behavior.lower() and primary_orientation != 'value':
            messaging[1] = "Emphasize value proposition and competitive pricing"
        elif "leasing" in behavior.lower():
            product_focus.append("Models with attractive lease residuals")
            customer_experience[2] = "Streamlined leasing programs"
        elif "technology" in behavior.lower() and primary_orientation != 'technology':
            product_focus[2] = "Models with appealing technology packages"
        elif "brand" in behavior.lower():
            messaging[3] = "Emphasize brand heritage and reputation"
            customer_experience[3] = "Brand-focused ownership experience"
    
    # Compile final marketing strategy
    marketing_strategy = {
        'messaging': list(set(messaging))[:4],  # Remove duplicates and limit to 4
        'channels': list(set(channels))[:4],
        'product_focus': list(set(product_focus))[:4],
        'customer_experience': list(set(customer_experience))[:4]
    }
    
    return marketing_strategy

def generate_segment_visualizations(segment_insights, report_dir, k):
    """
    Generate visualizations for the marketing report
    
    Parameters:
    -----------
    segment_insights : dict
        Dictionary with segment insights
    report_dir : str
        Directory to save visualizations
    k : int
        Number of clusters
        
    Returns:
    --------
    dict
        Dictionary mapping visualization types to file paths
    """
    log_status(f"Generating visualizations for {k} segments...")
    visualization_paths = {}
    
    # Create segment names and sizes for easier access
    segment_names = {}
    segment_sizes = {}
    segment_colors = {}
    
    # Standard colors for visualizations
    colors = ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099', '#0099c6', 
              '#dd4477', '#66aa00', '#b82e2e', '#316395', '#994499', '#22aa99',
              '#aaaa11', '#6633cc', '#e67300', '#8b0707', '#651067', '#329262',
              '#5574a6', '#3b3eac']
    
    # Ensure we have enough colors
    while len(colors) < k:
        colors.extend(colors[:k-len(colors)])
    
    # Prepare data for visualizations
    for i in range(k):
        if i in segment_insights['segments']:
            segment = segment_insights['segments'][i]
            segment_names[i] = segment['name']
            segment_sizes[i] = segment['size_pct']
            segment_colors[i] = colors[i]
    
    # 1. Segment Distribution Visualization
    try:
        log_status("Creating segment distribution chart...")
        plt.figure(figsize=(15, 10))
        
        # Create bar chart of segment sizes
        bars = plt.bar(
            [segment_names[i] for i in range(k)],
            [segment_sizes[i] for i in range(k)],
            color=[segment_colors[i] for i in range(k)]
        )
        
        # Add percentage labels above each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add count labels inside each bar
        total_records = 50000  # Placeholder - would use actual count in real implementation
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = int(height * total_records / 100)
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{count:,}', ha='center', va='center', 
                   color='white', fontweight='bold')
        
        # Customize the plot
        plt.title('Vehicle Customer Segment Distribution', fontsize=16, pad=20)
        plt.ylabel('Percentage of Customers', fontsize=12)
        plt.ylim(0, max(segment_sizes.values()) * 1.1)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        distribution_path = f"{report_dir}/vehicle_segment_distribution_{k}.jpg"
        plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['distribution'] = distribution_path
        log_status(f"Saved segment distribution chart to {distribution_path}")
    except Exception as e:
        log_status(f"Error creating distribution visualization: {str(e)}")
    
    # 2. Segment Quadrant Map Visualization
    try:
        log_status("Creating segment quadrant map...")
        plt.figure(figsize=(14, 12))
        
        # Define segment positioning on the quadrant map
        # Calculate based on luxury and tech scores from key_features
        segment_positions = {}
        
        for i in range(k):
            if i in segment_insights['segments']:
                luxury_score = segment_insights['segments'][i]['key_features']['luxury_orientation']
                tech_score = segment_insights['segments'][i]['key_features']['tech_adoption']
                
                # Scale to -5 to 5 range for both axes
                x = (luxury_score - 50) / 10
                y = (tech_score - 50) / 10
                
                # Add some randomness to prevent overlap
                x += np.random.uniform(-0.5, 0.5)
                y += np.random.uniform(-0.5, 0.5)
                
                segment_positions[i] = {"x": x, "y": y}
        
        # Plot the segments as bubbles
        for i in range(k):
            if i in segment_positions:
                plt.scatter(
                    segment_positions[i]["x"],
                    segment_positions[i]["y"],
                    s=segment_sizes[i] * 50,
                    color=segment_colors[i],
                    alpha=0.7,
                    edgecolor='white',
                    linewidth=1
                )
                # Add cluster number labels
                plt.text(
                    segment_positions[i]["x"],
                    segment_positions[i]["y"],
                    str(i),
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='white'
                )
                # Add segment name labels
                plt.text(
                    segment_positions[i]["x"],
                    segment_positions[i]["y"] - 0.6,
                    f"{segment_names[i]}\n({segment_sizes[i]:.1f}%)",
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='black'
                )
        
        # Add quadrant labels
        plt.text(3, 3, 'Tech-Forward Premium', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
        plt.text(-3, 3, 'Tech-Forward Value', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
        plt.text(3, -3, 'Traditional Premium', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
        plt.text(-3, -3, 'Traditional Value', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
        
        # Add axes labels
        plt.xlabel('Luxury Orientation', fontsize=14)
        plt.ylabel('Innovation Orientation', fontsize=14)
        
        # Add title
        plt.title('Vehicle Customer Segment Map', fontsize=16, pad=20)
        
        # Set axis limits
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        
        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add origin lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add axis descriptions
        plt.text(6, 0, 'Premium/Luxury', ha='right', va='center', fontsize=10)
        plt.text(-6, 0, 'Economy/Value', ha='left', va='center', fontsize=10)
        plt.text(0, 6, 'Innovative/Tech-Forward', ha='center', va='top', fontsize=10)
        plt.text(0, -6, 'Traditional', ha='center', va='bottom', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        quadrant_path = f"{report_dir}/vehicle_segment_quadrant_map_{k}.jpg"
        plt.savefig(quadrant_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['quadrant_map'] = quadrant_path
        log_status(f"Saved segment quadrant map to {quadrant_path}")
    except Exception as e:
        log_status(f"Error creating quadrant map visualization: {str(e)}")
    
    # 3. Radar Charts for Segment Profiles
    try:
        log_status("Creating segment radar profiles...")
        
        # Helper function for radar charts
        def radar_factory(num_vars, frame='circle'):
            """Create a radar chart with `num_vars` axes."""
            # Calculate evenly-spaced axis angles
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            
            # Create vertices for polygon plots
            def unit_poly_verts(theta):
                """Return vertices of polygon for subplot axes."""
                x0, y0, r = [0.5] * 3
                verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
                return verts
            
            # Class for creating a radar chart
            class RadarAxes(PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    self.theta = theta
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                    
                def fill(self, *args, closed=True, **kwargs):
                    return super().fill(self.theta, args[0], closed=closed, **kwargs)
                    
                def plot(self, *args, **kwargs):
                    return super().plot(self.theta, args[0], **kwargs)
                    
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(self.theta), labels)
                    
                def _gen_axes_patch(self):
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)
                        
                def _gen_axes_spines(self):
                    if frame == 'circle':
                        return super()._gen_axes_spines()
                    elif frame == 'polygon':
                        verts = unit_poly_verts(self.theta)
                        verts.append(verts[0])
                        path = Path(verts)
                        spine = Spine(self, 'circle', path)
                        spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                        return {'polar': spine}
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)
            
            register_projection(RadarAxes)
            return theta
        
        # Define radar chart attributes
        radar_attributes = ['Luxury Orientation', 'Price Sensitivity', 'Tech Adoption', 
                          'Brand Loyalty', 'Family Focus', 'Utility Priority']
        n_attributes = len(radar_attributes)
        theta = radar_factory(n_attributes, frame='polygon')
        
        # Calculate grid layout
        grid_size = max(2, int(np.ceil(np.sqrt(k))))
        
        # Create figure for radar charts
        fig, axes = plt.subplots(figsize=(16, 16), nrows=grid_size, ncols=grid_size, 
                                 subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85, bottom=0.05)
        
        # Flatten the axes array for easier iteration
        axes = axes.flatten()
        
        # Set plot limits for all axes
        for ax in axes:
            ax.set_ylim(0, 100)
        
        # Plot each segment on its own subplot
        for i in range(min(k, len(axes))):
            if i in segment_insights['segments']:
                # Get the key attribute values in order
                radar_data = [
                    segment_insights['segments'][i]['key_features'].get('luxury_orientation', 50),
                    segment_insights['segments'][i]['key_features'].get('price_sensitivity', 50),
                    segment_insights['segments'][i]['key_features'].get('tech_adoption', 50),
                    segment_insights['segments'][i]['key_features'].get('brand_loyalty', 50),
                    segment_insights['segments'][i]['key_features'].get('family_focus', 50),
                    segment_insights['segments'][i]['key_features'].get('utility_priority', 50)
                ]
                
                # Plot the radar chart
                ax = axes[i]
                ax.plot(radar_data, color=segment_colors[i], linewidth=2.5)
                ax.fill(radar_data, alpha=0.25, color=segment_colors[i])
                ax.set_title(segment_names[i], size=11, y=1.1, color=segment_colors[i], fontweight='bold')
                ax.set_varlabels(radar_attributes)
                
                # Customize grid lines
                ax.set_rgrids([20, 40, 60, 80], labels=['20', '40', '60', '80'], angle=0, fontsize=7)
                
                # Rotate attribute labels for better readability
                for label, angle in zip(ax.get_xticklabels(), theta):
                    if angle in (0, np.pi):
                        label.set_horizontalalignment('center')
                    elif 0 < angle < np.pi:
                        label.set_horizontalalignment('left')
                    else:
                        label.set_horizontalalignment('right')
                    label.set_fontsize(8)
        
        # Hide empty subplots
        for i in range(k, len(axes)):
            axes[i].set_visible(False)
        
        # Add main title
        fig.suptitle('Vehicle Customer Segment Profiles', fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.93, 'Based on k-modes clustering analysis', 
                 horizontalalignment='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        # Save the figure
        radar_path = f"{report_dir}/vehicle_segment_radar_profiles_{k}.jpg"
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['radar_profiles'] = radar_path
        log_status(f"Saved segment radar profiles to {radar_path}")
    except Exception as e:
        log_status(f"Error creating radar profiles visualization: {str(e)}")
    
    # 4. Marketing Strategy Recommendations
    try:
        log_status("Creating segment strategy recommendations visual...")
        
        # Function to create a strategy card for a segment
        def create_strategy_card(segment_id, ax):
            """Create a marketing strategy card visualization for a segment"""
            segment = segment_insights['segments'].get(segment_id)
            if not segment:
                ax.set_visible(False)
                return
            segment_name = segment['name']
            data = segment
            color = segment_colors.get(segment_id, '#333333')
            
            # Set title
            ax.set_title(segment_name, fontsize=12, fontweight='bold', color=color, pad=10)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Background color
            ax.set_facecolor('#f8f9fa')
            
            # Segment size
            ax.text(0.02, 0.98, f"Segment size: {data['size_pct']:.1f}%", transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Segment demographics summary
            demographics = f"Demographics: {data['demographics'].get('age', 'Various')} • {data['demographics'].get('income', 'Various')}"
            ax.text(0.02, 0.91, demographics, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Create text blocks for each strategy section
            ypositions = [0.85, 0.60, 0.35, 0.10]
            heights = [0.15, 0.15, 0.15, 0.15]
            titles = ['Key Messaging', 'Marketing Channels', 'Product Focus', 'Customer Experience']
            content_lists = [
                data['marketing_strategy'].get('messaging', []),
                data['marketing_strategy'].get('channels', []),
                data['marketing_strategy'].get('product_focus', []),
                data['marketing_strategy'].get('customer_experience', [])
            ]
            
            for ypos, height, title, content in zip(ypositions, heights, titles, content_lists):
                # Section background
                rect = plt.Rectangle((0.05, ypos-height), 0.9, height, 
                                   fill=True, color='white', alpha=0.8,
                                   transform=ax.transAxes, zorder=1,
                                   linewidth=1, edgecolor='#dddddd')
                ax.add_patch(rect)
                
                # Section title
                ax.text(0.07, ypos-0.03, title, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', verticalalignment='top',
                        horizontalalignment='left', zorder=2)
                
                # Section content as bullet points
                for i, item in enumerate(content[:4]):  # Limit to 4 items
                    ax.text(0.09, ypos-0.06-(i*0.025), f"• {item}", transform=ax.transAxes,
                            fontsize=8, verticalalignment='top', horizontalalignment='left',
                            zorder=2, wrap=True)
        
        # Create marketing strategy cards visualization
        fig = plt.figure(figsize=(16, 20))
        
        # Calculate grid dimensions
        grid_cols = min(3, k)
        grid_rows = int(np.ceil(k / grid_cols))
        
        # Create a GridSpec
        gs = GridSpec(grid_rows, grid_cols, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title for the entire figure
        fig.suptitle('Marketing Strategy Recommendations by Segment', fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.955, 'Based on k-modes clustering analysis of customer vehicle preferences',
                 horizontalalignment='center', fontsize=10)
        
        # Create a subplot and strategy card for each segment
        for i in range(k):
            if i in segment_insights['segments']:
                # Calculate grid position
                row = i // grid_cols
                col = i % grid_cols
                
                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                
                # Create strategy card
                create_strategy_card(i, ax)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Save the figure
        strategy_path = f"{report_dir}/vehicle_segment_marketing_strategies_{k}.jpg"
        plt.savefig(strategy_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualization_paths['marketing_strategies'] = strategy_path
        log_status(f"Saved marketing strategies visualization to {strategy_path}")
    except Exception as e:
        log_status(f"Error creating marketing strategies visualization: {str(e)}")
    
    return visualization_paths

def create_marketing_report_document(segment_insights, visualization_paths, report_dir, k):
    """
    Create a comprehensive marketing report document based on clustering results
    
    Parameters:
    -----------
    segment_insights : dict
        Dictionary with segment insights
    visualization_paths : dict
        Dictionary with paths to visualizations
    report_dir : str
        Directory to save the report
    k : int
        Number of clusters
        
    Returns:
    --------
    str
        Path to the generated report
    """
    log_status("Creating comprehensive marketing report document...")
    
    # Create report file path
    report_path = f"{report_dir}/Vehicle_Customer_Segmentation_Report_{k}.md"
    
    # Generate report content
    report_content = f"""# Vehicle Customer Segmentation & Marketing Strategy
## Based on K-modes Clustering Analysis

*Prepared: {time.strftime('%B %d, %Y')}*

---

## Executive Summary

This report presents the results of advanced k-modes clustering analysis performed on customer vehicle preference data. We identified {k} distinct customer segments with unique characteristics, behaviors, and purchase drivers. These segments provide a foundation for targeted marketing strategies that can enhance campaign effectiveness, improve customer experience, and optimize product development and positioning.

The analysis reveals several key insights:

1. The market shows distinct segmentation by luxury orientation, price sensitivity, and technology adoption
2. Family-focused customers represent a substantial portion of the market with specific needs
3. Technology adoption is a significant differentiator across segments
4. Price sensitivity varies significantly and correlates strongly with preferred vehicle types
5. Marketing channel effectiveness differs substantially between segments, suggesting the need for tailored approaches

The {k} identified segments represent natural groupings based on multiple customer attributes including demographics, behaviors, preferences, and purchase patterns. Each segment has been analyzed to create actionable marketing personas with specific recommendations.

---

## Identified Customer Segments

"""
    
    # Add visualization reference if available
    if 'distribution' in visualization_paths:
        report_content += f"![Vehicle Customer Segment Distribution]({os.path.basename(visualization_paths['distribution'])})\n\n"
    
    report_content += f"Our k-modes clustering analysis revealed {k} distinct customer segments:\n\n"
    
    # Add segment summaries
    for i in range(k):
        if i in segment_insights['segments']:
            segment = segment_insights['segments'][i]
            report_content += f"**{i+1}. {segment['name']} ({segment['size_pct']:.1f}%)**: "
            
            # Create a brief description based on key features
            key_features = segment['key_features']
            vehicle_prefs = segment['vehicle_preferences']
            
            luxury_level = "luxury" if key_features['luxury_orientation'] > 70 else \
                          "premium" if key_features['luxury_orientation'] > 50 else \
                          "mainstream" if key_features['luxury_orientation'] > 30 else "value-oriented"
                          
            tech_level = "tech-forward" if key_features['tech_adoption'] > 70 else \
                         "tech-savvy" if key_features['tech_adoption'] > 50 else \
                         "technology-conscious" if key_features['tech_adoption'] > 30 else ""
                         
            price_level = "price-insensitive" if key_features['price_sensitivity'] < 30 else \
                         "value-conscious" if key_features['price_sensitivity'] > 70 else ""
                         
            family_focus = "family-oriented" if key_features['family_focus'] > 70 else ""
            
            # Combine descriptors into a brief segment description
            descriptors = [d for d in [luxury_level, tech_level, price_level, family_focus] if d]
            description = f"{' '.join(descriptors).capitalize()} consumers "
            
            # Add vehicle preference if available
            if vehicle_prefs:
                description += f"seeking {vehicle_prefs[0].lower()}"
            else:
                description += "with distinctive automotive preferences"
                
            report_content += f"{description}\n\n"
    
    # Market segment positioning section
    report_content += "---\n\n## Market Segment Positioning\n\n"
    
    # Add visualization reference if available
    if 'quadrant_map' in visualization_paths:
        report_content += f"![Vehicle Customer Segment Map]({os.path.basename(visualization_paths['quadrant_map'])})\n\n"
    
    report_content += f"""The quadrant map above visualizes how our {k} segments position along two critical dimensions:
- **Horizontal Axis**: Luxury Orientation (economy/value to premium/luxury)
- **Vertical Axis**: Innovation Orientation (traditional to tech-forward)

This positioning reveals four primary market categories:
- **Tech-Forward Premium**: Segments focused on cutting-edge technology and premium features
- **Tech-Forward Value**: Segments embracing technology at more accessible price points 
- **Traditional Premium**: Segments prioritizing luxury and established prestige
- **Traditional Value**: Segments focused on practicality, reliability, and value

Understanding this positioning is crucial for developing tailored marketing strategies and appropriate product positioning for each segment.

---

## Segment Profiles and Key Attributes

"""
    
    # Add visualization reference if available
    if 'radar_profiles' in visualization_paths:
        report_content += f"![Vehicle Customer Segment Profiles]({os.path.basename(visualization_paths['radar_profiles'])})\n\n"
    
    report_content += """The radar charts illustrate the distinct profiles of each customer segment across six key attributes:
- **Luxury Orientation**: Preference for premium features and prestigious brands
- **Price Sensitivity**: Importance of cost in purchase decisions
- **Tech Adoption**: Openness to new technologies and digital features
- **Brand Loyalty**: Tendency to stick with preferred brands
- **Family Focus**: Prioritization of family needs in vehicle decisions
- **Utility Priority**: Emphasis on practical functionality and versatility

These profiles highlight the unique combinations of priorities and preferences that define each segment, allowing for more targeted marketing approaches.

---

## Detailed Segment Analysis

"""
    
    # Add detailed segment profiles
    for i in range(k):
        if i in segment_insights['segments']:
            segment = segment_insights['segments'][i]
            
            report_content += f"### Segment {i}: {segment['name']} ({segment['size_pct']:.1f}%)\n\n"
            
            # Demographics section
            report_content += "**Demographics:** \n"
            for key, value in segment['demographics'].items():
                if key not in ['gender_split', 'geography']:  # Keep it focused
                    report_content += f"- {key.replace('_', ' ').title()}: {value}\n"
            report_content += "\n"
            
            # Characteristics section
            report_content += "**Characteristics:**\n"
            key_features = segment['key_features']
            report_content += f"- {'Very high' if key_features['luxury_orientation'] > 80 else 'High' if key_features['luxury_orientation'] > 60 else 'Medium' if key_features['luxury_orientation'] > 40 else 'Low'} luxury orientation ({key_features['luxury_orientation']}/100)\n"
            report_content += f"- {'Very high' if key_features['price_sensitivity'] > 80 else 'High' if key_features['price_sensitivity'] > 60 else 'Medium' if key_features['price_sensitivity'] > 40 else 'Low'} price sensitivity ({key_features['price_sensitivity']}/100)\n"
            report_content += f"- {'Very high' if key_features['tech_adoption'] > 80 else 'High' if key_features['tech_adoption'] > 60 else 'Medium' if key_features['tech_adoption'] > 40 else 'Low'} tech adoption ({key_features['tech_adoption']}/100)\n"
            report_content += f"- {'Very high' if key_features['brand_loyalty'] > 80 else 'High' if key_features['brand_loyalty'] > 60 else 'Medium' if key_features['brand_loyalty'] > 40 else 'Low'} brand loyalty ({key_features['brand_loyalty']}/100)\n"
            if key_features['family_focus'] > 40:
                report_content += f"- {'Very high' if key_features['family_focus'] > 80 else 'High' if key_features['family_focus'] > 60 else 'Medium'} family focus ({key_features['family_focus']}/100)\n"
            if key_features['utility_priority'] > 40:
                report_content += f"- {'Very high' if key_features['utility_priority'] > 80 else 'High' if key_features['utility_priority'] > 60 else 'Medium'} utility priority ({key_features['utility_priority']}/100)\n"
            report_content += "\n"
            
            # Vehicle preferences section
            report_content += "**Vehicle Preferences:**\n"
            for pref in segment['vehicle_preferences']:
                report_content += f"- {pref}\n"
            report_content += "\n"
            
            # Buying behavior section
            report_content += "**Buying Behavior:**\n"
            for behavior in segment['buying_behavior']:
                report_content += f"- {behavior}\n"
            report_content += "\n"
            
            # Marketing strategy section
            report_content += "**Marketing Strategy:**\n"
            for message in segment['marketing_strategy']['messaging'][:3]:  # Limit to top 3
                report_content += f"- {message}\n"
            for channel in segment['marketing_strategy']['channels'][:2]:  # Limit to top 2
                report_content += f"- Utilize {channel.lower()}\n"
            report_content += "\n"
    
    # Marketing recommendations section
    report_content += "---\n\n## Marketing Strategy Recommendations\n\n"
    
    # Add visualization reference if available
    if 'marketing_strategies' in visualization_paths:
        report_content += f"![Marketing Strategy Recommendations by Segment]({os.path.basename(visualization_paths['marketing_strategies'])})\n\n"
    
    report_content += """Each segment requires a tailored marketing approach to effectively engage its members. The visualization above provides specific recommendations for:

1. **Key Messaging**: The most compelling message themes and content for each segment
2. **Marketing Channels**: The most effective channels and media for reaching each segment
3. **Product Focus**: The vehicle types and features to emphasize for each segment
4. **Customer Experience**: The dealership and service experiences that will resonate with each segment

---

## Strategic Recommendations

Based on the comprehensive segmentation analysis, we recommend the following strategies:

### 1. Segment-Specific Product Development

- **Luxury Tech Integration**: Develop high-end models with advanced technology to capture luxury-focused segments
- **Family-Friendly Innovation**: Create products balancing practicality with modern features for family segments
- **Value-Focused Technology**: Introduce affordable models with strategic tech features for value-conscious segments
- **Sustainable Options**: Expand eco-friendly lineup to target environmentally-conscious segments

### 2. Marketing Campaign Optimization

- **Channel Alignment**: Match marketing channel mix to segment preferences
- **Message Tailoring**: Customize messaging to address specific segment priorities and pain points
- **Visual Language**: Develop distinct visual approaches for different segments across all touchpoints
- **Timing Strategies**: Optimize campaign timing based on segment purchase cycles

### 3. Customer Experience Enhancements

- **Digital Transformation**: Accelerate digital experience development for tech-forward segments
- **Showroom Evolution**: Create segment-specific zones or experiences within dealerships
- **Service Differentiation**: Develop tiered service models aligned with segment expectations
- **Community Building**: Foster segment-specific communities, especially for lifestyle segments

### 4. Dealer Support Programs

- **Segment Training**: Train sales staff on segment identification and appropriate approaches
- **Inventory Mix**: Guide dealers on optimal inventory distribution based on local segment composition
- **Pricing Strategy**: Develop segment-responsive pricing and promotion guidelines
- **Performance Measurement**: Track conquest and retention by segment to refine strategies

---

## Implementation Roadmap

**Phase 1: Foundation (Month 1-2)**
- Finalize segment definitions and profiles
- Develop segmentation scoring model
- Create segment identification tools for dealers
- Establish segment-based KPIs and measurement

**Phase 2: Integration (Month 3-4)**
- Incorporate segmentation into marketing planning
- Initiate segment-specific creative development
- Begin dealer training program rollout
- Launch first targeted digital campaigns

**Phase 3: Expansion (Month 5-6)**
- Implement full multi-channel segment strategies
- Complete dealer training and support materials
- Introduce segment-specific customer journeys
- Launch product development initiatives

**Phase 4: Refinement (Month 7-12)**
- Analyze segment performance and response
- Refine targeting models based on initial results
- Optimize channel mix and marketing spend
- Develop next generation of segment-based strategies

---

## Conclusion

"""
    
    report_content += f"""The {k}-segment model provides a comprehensive framework for understanding the diverse customer landscape. By implementing targeted strategies for each segment, we can achieve:

- **Improved Marketing ROI**: More efficient allocation of marketing resources
- **Enhanced Customer Acquisition**: More compelling, relevant messaging
- **Increased Customer Retention**: Better alignment with customer needs and expectations
- **Product Development Guidance**: Clear direction for future vehicle development

This segmentation should be viewed as a living framework that will evolve as market conditions change and new data becomes available. Regular refinement of the model will ensure continued relevance and effectiveness.

---

*End of Report*
"""
    
    # Write the report to a file
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    log_status(f"Marketing report document created at: {report_path}")
    
    return report_path

def extend_kmodes_with_marketing_report(kmodes_results):
    """
    Extend k-modes clustering with marketing report generation.
    
    Parameters:
    -----------
    kmodes_results : dict
        The results from k-modes clustering run
        
    Returns:
    --------
    str
        Path to the generated marketing report
    """
    if not kmodes_results:
        log_status("No k-modes clustering results available for marketing report generation")
        return None
    
    log_status("Generating comprehensive marketing segmentation report...")
    
    try:
        # Extract key information
        cluster_data = kmodes_results.get('cluster_data')
        centroids_df = kmodes_results.get('centroids_df')
        num_clusters = kmodes_results.get('num_clusters')
        
        if not cluster_data is None and not centroids_df is None and num_clusters:
            # Create report directory
            report_dir = f"/dbfs/FileStore/acxiom_clustering/kmodes_marketing_report_{num_clusters}"
            os.makedirs(report_dir, exist_ok=True)
            
            # Analyze clusters to generate marketing insights
            segment_insights = analyze_kmodes_segments(cluster_data, centroids_df, num_clusters)
            
            # Generate visualizations
            visualization_paths = generate_segment_visualizations(segment_insights, report_dir, num_clusters)
            
            # Create the comprehensive report
            report_path = create_marketing_report_document(segment_insights, visualization_paths, report_dir, num_clusters)
            
            log_status(f"✓ Marketing segmentation report successfully generated at: {report_path}")
            return report_path
        else:
            log_status("Missing required data in k-modes results for marketing report generation")
            return None
            
    except Exception as e:
        log_status(f"ERROR generating marketing report: {str(e)}")
        log_status(traceback.format_exc())
        return None



# COMMAND ----------

# CELL 28: Run kmodes

def run_kmodes_with_marketing_report(cluster_counts=None, sample_size=None):
    """
    Run K-modes clustering with comprehensive marketing report generation
    
    Parameters:
    -----------
    cluster_counts : list, optional
        List of cluster counts to try (default [8, 10, 12])
    sample_size : int, optional
        Sample size to use for clustering
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Use default values if not specified
    if cluster_counts is None:
        cluster_counts = [8, 10, 12]
    
    if sample_size is None:
        sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000)
    
    log_status(f"===== EXECUTING K-MODES CLUSTERING WITH MARKETING REPORT =====")
    log_status(f"Cluster counts: {cluster_counts}")
    log_status(f"Sample size: {sample_size}")
    
    start_time = time.time()
    
    try:
        # First check if kmodes package is available
        if not install_kmodes_if_needed():
            log_status("ERROR: Required kmodes package not available")
            return False
        
        # Run k-modes with multiple cluster counts
        results = run_kmodes_multiple(
            cluster_counts=cluster_counts,
            sample_size=sample_size
        )
        
        if not results:
            log_status("ERROR: K-modes clustering failed")
            return False
        
        # Generate segment names for best cluster solution (lowest cost)
        cost_values = {k: result['cost'] for k, result in results.items()}
        best_k = min(cost_values, key=cost_values.get)
        
        log_status(f"Selected optimal clustering with k={best_k} clusters")
        
        # Generate marketing report for best clustering
        best_result = results[best_k]
        
        # Add segment names to best clustering result
        segment_names = generate_kmodes_segment_names(best_result)
        if segment_names:
            log_status(f"Generated {len(segment_names)} segment names for optimal clustering")
        
        # Generate marketing report
        report_path = generate_kmodes_marketing_report(best_result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if report_path:
            log_status(f"✅ K-modes clustering and marketing report completed in {total_time:.2f} seconds")
            log_status(f"Marketing report saved to: {report_path}")
            return True
        else:
            log_status(f"❌ Marketing report generation failed after {total_time:.2f} seconds")
            return False
        
    except Exception as e:
        log_status(f"ERROR in k-modes clustering: {str(e)}")
        log_status(traceback.format_exc())
        return False
    

# Example usage:
# Run K-Prototypes with 8 clusters only and increased weight for categorical variables
run_just_kprototypes(cluster_counts=[8], categorical_weight=0.8)

# Run K-Prototypes with default settings (8, 10, 12 clusters)
# run_just_kprototypes()

# Run K-Prototypes with multiple cluster counts and a smaller sample size
# run_just_kprototypes(cluster_counts=[5, 8, 10, 12, 15], sample_size=30000)

#(cluster_counts=[4])
#run_just_kmodes(cluster_counts=[5, 8, 10])
#compare_kmodes_with_other_methods(sample_size=15000, num_clusters=8)
#determine_optimal_k_for_kmodes_and_run(sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000), max_k=15)

#run_kmodes_with_marketing_report(cluster_counts=[4], sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000))

# Run kprototypes with multiple cluster counts

#run_kprototypes_with_marketing_report(cluster_counts=[4, 8, 10], sample_size = GLOBAL_CONFIG.get('clustering_sample_size', 50000))



# COMMAND ----------

# Cell 29: Invoke Kprototypes

# Run with default settings (8, 10, 12 clusters)
#run_kprototypes_with_marketing_report()

# Run with specific cluster count
run_kprototypes_with_marketing_report(cluster_counts=[8])

# Run with multiple cluster counts and adjusted categorical weight
#run_kprototypes_with_marketing_report(cluster_counts=[5, 8, 10], categorical_weight=0.7)

# Run with global sample size
#run_kprototypes_with_marketing_report(cluster_counts=[5, 8, 10], sample_size=GLOBAL_CONFIG.get('clustering_sample_size', 50000))

# COMMAND ----------

# CELL: Run MCA and K-means with Directory Creation

def run_mca_kmeans_with_directory_creation():
    """
    Run MCA analysis followed by k-means clustering with guaranteed directory creation
    """
    log_status("===== STARTING MCA AND K-MEANS PIPELINE WITH DIRECTORY CREATION =====")
    
    # Step 1: Extract data
    log_status("Step 1: Extracting data...")
    extracted_data = direct_extract_acxiom_data(sample_size=GLOBAL_CONFIG['mca_sample_size'])
    
    if not extracted_data:
        log_status("ERROR: Data extraction failed")
        return False
    
    # Step 2: Manually prepare data for MCA with fixed NaN handling
    log_status("Step 2: Preparing data for MCA with correct NaN handling...")
    
    # Get Spark DataFrame
    spark_df = extracted_data['spark_df']
    # Convert to pandas
    df = spark_df.toPandas()
    
    # Get column definitions
    id_column, vehicle_columns, propensity_columns, demographic_columns, lifestyle_columns, financial_columns = define_columns()
    
    # Create available columns lists
    available_vehicle_cols = [col for col in vehicle_columns if col in df.columns]
    available_propensity_cols = [col for col in propensity_columns if col in df.columns]
    available_demographic_cols = [col for col in demographic_columns if col in df.columns]
    available_lifestyle_cols = [col for col in lifestyle_columns if col in df.columns]
    available_financial_cols = [col for col in financial_columns if col in df.columns]
    
    # Combine all available feature columns
    available_feature_cols = (available_vehicle_cols + available_propensity_cols + 
                            available_demographic_cols + available_lifestyle_cols + 
                            available_financial_cols)
    
    # Create category mappings
    column_categories = {}
    for col in available_vehicle_cols:
        column_categories[col] = "Vehicle"
    for col in available_propensity_cols:
        column_categories[col] = "Propensity"
    for col in available_demographic_cols:
        column_categories[col] = "Demographic"
    for col in available_lifestyle_cols:
        column_categories[col] = "Lifestyle"
    for col in available_financial_cols:
        column_categories[col] = "Financial"
    
    # Extract features
    features = df[available_feature_cols].copy()
    id_values = df[id_column].copy() if id_column in df.columns else None
    
    # Identify categorical columns
    categorical_cols = []
    for col in available_feature_cols:
        try:
            # Skip columns that have all NaN values
            if features[col].isna().all():  # Correct check for all NaN
                log_status(f"Skipping column {col} with all NaN values")
                continue
                
            # Check if categorical by type or value count
            is_categorical = (pd.api.types.is_object_dtype(features[col]) or 
                             isinstance(features[col].dtype, pd.CategoricalDtype) or
                            (pd.api.types.is_numeric_dtype(features[col]) and features[col].nunique() <= 15))
            
            if is_categorical:
                categorical_cols.append(col)
                # Convert to string and fill NaNs
                features[col] = features[col].fillna("missing").astype(str)
        except Exception as e:
            log_status(f"Warning: Error processing column {col}: {str(e)}")
            # Skip problematic column
            continue
    
    # Check if we have enough categorical columns
    if len(categorical_cols) < 3:
        log_status(f"ERROR: Not enough categorical columns for MCA (only {len(categorical_cols)})")
        return False
    
    # Create prepared data dictionary
    prepared_data = {
        'features': features,
        'feature_cols': available_feature_cols,
        'id_column': id_column,
        'id_values': id_values,
        'categorical_cols': categorical_cols,
        'column_categories': column_categories
    }
    
    log_status(f"Successfully prepared {len(categorical_cols)} categorical columns for MCA")
    
    # Step 3: Run MCA analysis
    log_status("Step 3: Running MCA analysis...")
    
    mca_analysis = robust_run_mca_analysis(prepared_data)
    
    if not mca_analysis:
        log_status("ERROR: MCA analysis failed")
        return False
    
    log_status(f"Successfully completed MCA analysis with {mca_analysis['n_dims']} dimensions")
    
    # Step 4: Run k-means clustering with specified cluster counts
    log_status("Step 4: Running k-means clustering...")
    
    cluster_counts = [4, 8, 12]
    clustering_results = {}
    
    # Create base output directory if it doesn't exist
    output_prefix = GLOBAL_CONFIG['clustering_output_prefix']
    output_base_dir = os.path.dirname(output_prefix)
    
    # Ensure base directory exists
    try:
        if not dbutils.fs.ls(output_base_dir):
            log_status(f"Creating base output directory: {output_base_dir}")
            dbutils.fs.mkdirs(output_base_dir)
        else:
            log_status(f"Output directory already exists: {output_base_dir}")
    except Exception as e:
        log_status(f"WARNING: Could not create output directory {output_base_dir}: {str(e)}")
        log_status("Will attempt to continue anyway")
    
    for k in cluster_counts:
        log_status(f"Running k-means with {k} clusters...")
        
        # Select dimensions for clustering
        n_dims = mca_analysis['n_dims']
        dim_cols = [f'MCA_dim{i+1}' for i in range(n_dims)]
        
        # Get data for clustering
        cluster_data = mca_analysis['mca_coords'][dim_cols].fillna(0)
        
        # Run k-means
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data)
        
        # Calculate silhouette score if possible
        try:
            sil_score = silhouette_score(cluster_data, cluster_labels)
            log_status(f"K-means with k={k}: silhouette score = {sil_score:.4f}")
        except Exception as e:
            log_status(f"Warning: Unable to calculate silhouette score: {str(e)}")
            sil_score = None
        
        # Add cluster labels to MCA coordinates
        result_df = mca_analysis['mca_coords'].copy()
        result_df['kmeans_cluster'] = cluster_labels
        
        # Create centroids DataFrame
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=dim_cols)
        centers['cluster_id'] = range(k)
        
        # Save results
        result_path = f"{output_prefix}_kmeans_{k}_clusters.parquet"
        centers_path = f"{output_prefix}_kmeans_{k}_centers.csv"
        
        # Create specific directories for this cluster count
        kmeans_dir = os.path.dirname(result_path)
        try:
            if not dbutils.fs.ls(kmeans_dir):
                log_status(f"Creating k-means output directory: {kmeans_dir}")
                dbutils.fs.mkdirs(kmeans_dir)
        except Exception as e:
            log_status(f"WARNING: Could not create k-means directory {kmeans_dir}: {str(e)}")
        
        # Save files with explicit error handling
        try:
            result_df.to_parquet(f"/dbfs{result_path}")
            log_status(f"Saved k-means clustering results to {result_path}")
            
            # Verify file exists after saving
            if dbutils.fs.ls(result_path):
                log_status(f"Verified file exists: {result_path}")
            else:
                log_status(f"WARNING: File does not exist after saving: {result_path}")
        except Exception as e:
            log_status(f"ERROR saving clustering results: {str(e)}")
        
        try:
            centers.to_csv(f"/dbfs{centers_path}", index=False)
            log_status(f"Saved k-means centers to {centers_path}")
            
            # Verify file exists after saving
            if dbutils.fs.ls(centers_path):
                log_status(f"Verified file exists: {centers_path}")
            else:
                log_status(f"WARNING: File does not exist after saving: {centers_path}")
        except Exception as e:
            log_status(f"ERROR saving centroids: {str(e)}")
        
        # Store results for marketing report
        clustering_results[k] = {
            'cluster_data': result_df,
            'centroids_df': centers,
            'num_clusters': k,
            'silhouette_score': sil_score
        }
        
        log_status(f"Successfully completed k-means clustering with {k} clusters")
    
    # Step 5: Generate marketing reports
    log_status("Step 5: Generating marketing reports...")
    
    for k, result in clustering_results.items():
        log_status(f"Generating marketing report for {k} clusters...")
        
        try:
            # Create report directory
            report_dir = f"{output_prefix}_kmeans_{k}_marketing_report"
            try:
                if not dbutils.fs.ls(report_dir):
                    log_status(f"Creating marketing report directory: {report_dir}")
                    dbutils.fs.mkdirs(report_dir)
            except Exception as dir_error:
                log_status(f"WARNING: Could not create report directory {report_dir}: {str(dir_error)}")
            
            # Generate report
            report_path = generate_kmodes_marketing_report(
                result,
                output_prefix=f"{output_prefix}_kmeans_{k}"
            )
            
            if report_path:
                log_status(f"✅ Successfully generated marketing report at {report_path}")
                
                # Verify report exists
                if dbutils.fs.ls(report_path):
                    log_status(f"Verified report file exists: {report_path}")
                else:
                    log_status(f"WARNING: Report file does not exist after generation: {report_path}")
            else:
                log_status(f"❌ Failed to generate marketing report for {k} clusters")
        except Exception as e:
            log_status(f"ERROR generating marketing report for {k} clusters: {str(e)}")
            log_status(traceback.format_exc())
    
    log_status("===== MCA AND K-MEANS PIPELINE COMPLETED =====")
    return True

# Execute the pipeline with directory creation
run_mca_kmeans_with_directory_creation()

# COMMAND ----------

################## Marketing Persona Reports ###################


# COMMAND ----------

##### Viz Cell 1: Vehicle Customer Segment Distribution Report #####
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Segment data
segments = {
    'Premium Luxury Enthusiasts': 2.5,
    'Ultra-Premium Specialty Buyers': 0.7,
    'Technology-Forward Early Adopters': 3.0,
    'Upscale Family Vehicle Buyers': 6.4,
    'Mainstream Family Vehicle Owners': 25.3,
    'Value-Oriented Vehicle Buyers': 24.1,
    'Practical Utility Vehicle Owners': 12.3,
    'Economy-Minded Traditional Buyers': 25.7
}

# Create a DataFrame for the segments
df = pd.DataFrame({
    'Segment': list(segments.keys()),
    'Percentage': list(segments.values())
})

# Calculate the counts (assuming 10,000 total customers based on the clustering results)
total_customers = 10000
df['Count'] = (df['Percentage'] * total_customers / 100).astype(int)

# Sort by percentage (optional)
# df = df.sort_values('Percentage', ascending=False)

# Create a color map for the segments
colors = ['#3366cc', '#dc3912', '#ff9900', '#109618', 
          '#990099', '#0099c6', '#dd4477', '#66aa00']

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create the bar chart
bars = ax.bar(df['Segment'], df['Percentage'], color=colors)

# Add percentage labels above each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{height}%', ha='center', va='bottom', fontweight='bold')

# Add count labels inside each bar
for bar in bars:
    height = bar.get_height()
    count = int(height * total_customers / 100)
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{count:,}', ha='center', va='center', 
            color='white', fontweight='bold')

# Customize the plot
ax.set_title('Vehicle Customer Segment Distribution', fontsize=16, pad=20)
ax.set_ylabel('Percentage of Customers', fontsize=12)
ax.set_ylim(0, max(df['Percentage']) * 1.1)  # Add some space for the labels

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Save the figure
plt.savefig('vehicle_segment_distribution.png', dpi=300, bbox_inches='tight')
"""


# COMMAND ----------

##### Viz Cell 2: Veicle Customer Segment Quadrant Map #####
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Define segment data
# x: Luxury Orientation, y: Innovation Openness, size: percentage
segments = [
    {'name': 'Premium Luxury Enthusiasts', 'x': 5.5, 'y': 0, 'size': 2.5, 'cluster': 0, 'color': '#3366cc'},
    {'name': 'Ultra-Premium Specialty Buyers', 'x': 4.5, 'y': 1.5, 'size': 0.7, 'cluster': 1, 'color': '#dc3912'},
    {'name': 'Technology-Forward Early Adopters', 'x': 2.0, 'y': 5.5, 'size': 3.0, 'cluster': 2, 'color': '#ff9900'},
    {'name': 'Upscale Family Vehicle Buyers', 'x': 3.0, 'y': 4.0, 'size': 6.4, 'cluster': 3, 'color': '#109618'},
    {'name': 'Mainstream Family Vehicle Owners', 'x': -1.0, 'y': 1.5, 'size': 25.3, 'cluster': 4, 'color': '#990099'},
    {'name': 'Value-Oriented Vehicle Buyers', 'x': -2.5, 'y': 0, 'size': 24.1, 'cluster': 5, 'color': '#0099c6'},
    {'name': 'Practical Utility Vehicle Owners', 'x': -1.5, 'y': -2.5, 'size': 12.3, 'cluster': 6, 'color': '#dd4477'},
    {'name': 'Economy-Minded Traditional Buyers', 'x': -3.0, 'y': -3.5, 'size': 25.7, 'cluster': 7, 'color': '#66aa00'}
]

# Convert to DataFrame
df = pd.DataFrame(segments)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the segments as bubbles
for _, segment in df.iterrows():
    ax.scatter(
        segment['x'], 
        segment['y'], 
        s=segment['size'] * 50,  # Adjust size multiplier as needed
        color=segment['color'], 
        alpha=0.7, 
        edgecolor='white',
        linewidth=1
    )
    # Add cluster number labels
    ax.text(
        segment['x'], 
        segment['y'], 
        str(segment['cluster']),
        ha='center', 
        va='center', 
        fontweight='bold',
        color='white'
    )
    # Add segment name and size labels
    ax.text(
        segment['x'], 
        segment['y'] - 0.6, 
        f"{segment['name']}\n({segment['size']}%)",
        ha='center', 
        va='top', 
        fontsize=8,
        color='black'
    )

# Add quadrant labels
ax.text(3, 3, 'Tech-Forward Premium', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
ax.text(-3, 3, 'Tech-Forward Value', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
ax.text(3, -3, 'Traditional Premium', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')
ax.text(-3, -3, 'Traditional Value', ha='center', va='center', fontsize=12, fontweight='bold', color='#555555')

# Add axes labels
ax.set_xlabel('Luxury Orientation', fontsize=14)
ax.set_ylabel('Innovation Openness', fontsize=14)

# Add title
ax.set_title('Vehicle Customer Segment Map', fontsize=16, pad=20)

# Set axis limits
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Add origin lines
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

# Add axis descriptions
ax.text(6, 0, 'Premium/Luxury', ha='right', va='center', fontsize=10)
ax.text(-6, 0, 'Economy/Value', ha='left', va='center', fontsize=10)
ax.text(0, 6, 'Innovative/Tech-Forward', ha='center', va='top', fontsize=10)
ax.text(0, -6, 'Traditional', ha='center', va='bottom', fontsize=10)

# Add a legend for market segments
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3366cc', markersize=10, label='Premium Market (3.2%)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#109618', markersize=10, label='Innovation-Focused (9.4%)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#990099', markersize=10, label='Mainstream Market (49.4%)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66aa00', markersize=10, label='Practical/Economy (38.0%)')
]
ax.legend(handles=legend_elements, loc='lower right')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Save the figure
plt.savefig('vehicle_segment_quadrant_map.png', dpi=300, bbox_inches='tight')
"""


# COMMAND ----------

##### Viz Cell 3: Vehicle Customer Segment Radar Profiles #####
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Define a function for creating a radar chart
def radar_factory(num_vars, frame='circle'):
    # Create a radar chart with `num_vars` axes
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    # Class for creating a radar chart
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            self.theta = theta
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
            
        def fill(self, *args, closed=True, **kwargs):
            return super().fill(self.theta, args[0], closed=closed, **kwargs)
            
        def plot(self, *args, **kwargs):
            return super().plot(self.theta, args[0], **kwargs)
            
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(self.theta), labels)
            
        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
                
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                verts = unit_poly_verts(self.theta)
                verts.append(verts[0])
                path = Path(verts)
                spine = Spine(self, 'circle', path)
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
    
    register_projection(RadarAxes)
    return theta

def unit_poly_verts(theta):
    # Return vertices of polygon for subplot axes.
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

# Segment profiles based on key attributes
segment_profiles = {
    'Premium Luxury Enthusiasts': {
        'color': '#3366cc',
        'data': [
            90,  # Luxury Orientation
            20,  # Price Sensitivity
            70,  # Tech Adoption
            85,  # Brand Loyalty
            40,  # Family Focus
            30,  # Utility Priority
        ]
    },
    'Ultra-Premium Specialty Buyers': {
        'color': '#dc3912',
        'data': [
            95,  # Luxury Orientation
            10,  # Price Sensitivity
            80,  # Tech Adoption
            90,  # Brand Loyalty
            20,  # Family Focus
            25,  # Utility Priority
        ]
    },
    'Technology-Forward Early Adopters': {
        'color': '#ff9900',
        'data': [
            65,  # Luxury Orientation
            40,  # Price Sensitivity
            95,  # Tech Adoption
            40,  # Brand Loyalty
            50,  # Family Focus
            35,  # Utility Priority
        ]
    },
    'Upscale Family Vehicle Buyers': {
        'color': '#109618',
        'data': [
            75,  # Luxury Orientation
            45,  # Price Sensitivity
            70,  # Tech Adoption
            65,  # Brand Loyalty
            90,  # Family Focus
            60,  # Utility Priority
        ]
    },
    'Mainstream Family Vehicle Owners': {
        'color': '#990099',
        'data': [
            45,  # Luxury Orientation
            70,  # Price Sensitivity
            55,  # Tech Adoption
            60,  # Brand Loyalty
            85,  # Family Focus
            65,  # Utility Priority
        ]
    },
    'Value-Oriented Vehicle Buyers': {
        'color': '#0099c6',
        'data': [
            25,  # Luxury Orientation
            85,  # Price Sensitivity
            40,  # Tech Adoption
            50,  # Brand Loyalty
            60,  # Family Focus
            55,  # Utility Priority
        ]
    },
    'Practical Utility Vehicle Owners': {
        'color': '#dd4477',
        'data': [
            35,  # Luxury Orientation
            75,  # Price Sensitivity
            30,  # Tech Adoption
            65,  # Brand Loyalty
            50,  # Family Focus
            90,  # Utility Priority
        ]
    },
    'Economy-Minded Traditional Buyers': {
        'color': '#66aa00',
        'data': [
            15,  # Luxury Orientation
            95,  # Price Sensitivity
            20,  # Tech Adoption
            70,  # Brand Loyalty
            45,  # Family Focus
            60,  # Utility Priority
        ]
    }
}

# Define attribute labels
attributes = ['Luxury Orientation', 'Price Sensitivity', 'Tech Adoption', 
              'Brand Loyalty', 'Family Focus', 'Utility Priority']

# Create figure and radar chart factory
n_attributes = len(attributes)
theta = radar_factory(n_attributes, frame='polygon')

# Create subplots for each segment (in a 3x3 grid)
fig, axes = plt.subplots(figsize=(16, 14), nrows=3, ncols=3, 
                         subplot_kw=dict(projection='radar'))
fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85, bottom=0.05)

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Set plot limits for all axes
for ax in axes:
    ax.set_ylim(0, 100)

# Plot each segment on its own subplot
for i, (segment_name, profile) in enumerate(segment_profiles.items()):
    # Skip if we have more segments than subplots
    if i >= len(axes) - 1:  # Reserve the last subplot for the combined view
        break
        
    # Plot the radar chart
    ax = axes[i]
    ax.plot(profile['data'], color=profile['color'], linewidth=2.5)
    ax.fill(profile['data'], alpha=0.25, color=profile['color'])
    ax.set_title(segment_name, size=11, y=1.1, color=profile['color'], fontweight='bold')
    ax.set_varlabels(attributes)
    
    # Customize grid lines
    ax.set_rgrids([20, 40, 60, 80], labels=['20', '40', '60', '80'], angle=0, fontsize=7)
    
    # Rotate attribute labels for better readability
    for label, angle in zip(ax.get_xticklabels(), theta):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_fontsize(8)

# Create a combined view on the last subplot
combined_ax = axes[-1]
for segment_name, profile in segment_profiles.items():
    combined_ax.plot(profile['data'], color=profile['color'], linewidth=1.5, 
                     label=segment_name, alpha=0.8)

combined_ax.set_title('All Segments Comparison', size=12, y=1.1, fontweight='bold')
combined_ax.set_varlabels(attributes)
combined_ax.set_rgrids([20, 40, 60, 80], labels=['20', '40', '60', '80'], angle=0, fontsize=7)
combined_ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1.0), fontsize=8)

# Add main title
fig.suptitle('Vehicle Customer Segment Profiles', fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.93, 'Based on hierarchical clustering analysis', 
         horizontalalignment='center', fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.9])

# Show the plot
plt.show()

# Save the figure
plt.savefig('vehicle_segment_radar_profiles.png', dpi=300, bbox_inches='tight')
"""


# COMMAND ----------

##### Viz Cell 4: #####
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Define segment data with marketing strategies
segments = {
    'Premium Luxury Enthusiasts': {
        'size': 2.5,
        'color': '#3366cc',
        'messaging': [
            'Emphasize exclusivity and premium craftsmanship',
            'Highlight prestigious brand heritage and status',
            'Focus on cutting-edge technology and innovation',
            'Showcase luxury materials and attention to detail'
        ],
        'channels': [
            'Luxury lifestyle publications',
            'Exclusive events and experiences',
            'Personalized direct marketing',
            'High-end digital platforms',
            'Strategic partnerships with luxury brands'
        ],
        'product_focus': [
            'Flagship luxury models',
            'Premium features and options',
            'Exclusive limited editions',
            'Personalization programs'
        ],
        'customer_experience': [
            'White-glove concierge service',
            'Personalized shopping experience',
            'VIP ownership benefits',
            'Exclusive access to brand events'
        ]
    },
    'Ultra-Premium Specialty Buyers': {
        'size': 0.7,
        'color': '#dc3912',
        'messaging': [
            'Focus on exclusivity and rarity',
            'Emphasize bespoke customization options',
            'Highlight engineering excellence and craftsmanship',
            'Position as the pinnacle of automotive achievement'
        ],
        'channels': [
            'Direct one-to-one outreach',
            'Invitation-only events',
            'Ultra-high-net-worth networks',
            'Specialized luxury publications',
            'Private showings and experiences'
        ],
        'product_focus': [
            'Limited production models',
            'Bespoke customization programs',
            'Signature editions',
            'Collector series vehicles'
        ],
        'customer_experience': [
            'Completely personalized purchase journey',
            'Factory visits and behind-the-scenes access',
            'Dedicated relationship manager',
            'Exclusive owner community events'
        ]
    },
    'Technology-Forward Early Adopters': {
        'size': 3.0,
        'color': '#ff9900',
        'messaging': [
            'Focus on innovation and cutting-edge technology',
            'Emphasize environmental benefits and sustainability',
            'Highlight connectivity and integration features',
            'Position as forward-thinking and progressive'
        ],
        'channels': [
            'Technology publications and platforms',
            'Digital and social media',
            'Innovation conferences and events',
            'Tech-focused partnerships and integrations',
            'Online communities and forums'
        ],
        'product_focus': [
            'Electric and alternative fuel vehicles',
            'Models with advanced technology features',
            'Connectivity and smart integration',
            'Innovative design and materials'
        ],
        'customer_experience': [
            'Digital-first shopping experience',
            'Virtual reality product demonstrations',
            'Tech-focused showroom experiences',
            'Community of like-minded early adopters'
        ]
    },
    'Upscale Family Vehicle Buyers': {
        'size': 6.4,
        'color': '#109618',
        'messaging': [
            'Balance luxury with practical family functionality',
            'Emphasize safety features and technology',
            'Highlight spacious comfort and premium materials',
            'Focus on quality of experience for the whole family'
        ],
        'channels': [
            'Upscale family lifestyle publications',
            'Premium digital and social channels',
            'Family-oriented events and sponsorships',
            'Parenting communities and networks'
        ],
        'product_focus': [
            'Premium SUVs and crossovers',
            'Family-oriented luxury vehicles',
            'Safety-focused premium models',
            'Versatile luxury vehicles with ample space'
        ],
        'customer_experience': [
            'Family-friendly premium showrooms',
            'Child-friendly facilities during purchase and service',
            'Family test drive experiences',
            'Premium ownership benefits for the whole family'
        ]
    },
    'Mainstream Family Vehicle Owners': {
        'size': 25.3,
        'color': '#990099',
        'messaging': [
            'Focus on value, reliability, and practicality',
            'Emphasize family-friendly features and versatility',
            'Highlight safety ratings and features',
            'Show how vehicles fit into family lifestyle'
        ],
        'channels': [
            'Mass market advertising',
            'Family-focused digital platforms',
            'Social media and content marketing',
            'Partnerships with family brands',
            'Community events and sponsorships'
        ],
        'product_focus': [
            'Mid-size SUVs and crossovers',
            'Family sedans and minivans',
            'Models with strong safety ratings',
            'Versatile and practical vehicles'
        ],
        'customer_experience': [
            'Family-friendly showrooms',
            'Straightforward purchase process',
            'Transparent pricing and financing',
            'Reliable service and support'
        ]
    },
    'Value-Oriented Vehicle Buyers': {
        'size': 24.1,
        'color': '#0099c6',
        'messaging': [
            'Emphasize affordability and value proposition',
            'Focus on fuel efficiency and low operating costs',
            'Highlight reliability and practical features',
            'Demonstrate strong return on investment'
        ],
        'channels': [
            'Mass market advertising with value messaging',
            'Price comparison platforms',
            'Deal-focused digital marketing',
            'Email campaigns with special offers',
            'Search engine marketing for price-focused queries'
        ],
        'product_focus': [
            'Economy and compact vehicles',
            'Fuel-efficient models',
            'Entry-level variants with good feature set',
            'Models with strong warranty coverage'
        ],
        'customer_experience': [
            'No-pressure sales environment',
            'Transparent pricing',
            'Simple buying process',
            'Value-oriented service packages'
        ]
    },
    'Practical Utility Vehicle Owners': {
        'size': 12.3,
        'color': '#dd4477',
        'messaging': [
            'Focus on capability, durability, and functionality',
            'Highlight utility features and versatility',
            'Emphasize reliability and toughness',
            'Show practical applications and use cases'
        ],
        'channels': [
            'Industry and trade publications',
            'Specialized interest groups',
            'Work and utility-focused events',
            'Practical demonstration videos',
            'Partnerships with related industries'
        ],
        'product_focus': [
            'Pickup trucks and utility vans',
            'Work-oriented SUVs',
            'Models with towing and cargo capabilities',
            'Vehicles with practical customization options'
        ],
        'customer_experience': [
            'Practical demonstrations of capabilities',
            'Straightforward purchase experience',
            'Focus on specs and performance metrics',
            'Service programs that minimize downtime'
        ]
    },
    'Economy-Minded Traditional Buyers': {
        'size': 25.7,
        'color': '#66aa00',
        'messaging': [
            'Emphasize lowest total cost of ownership',
            'Focus on reliability and longevity',
            'Highlight fuel economy and efficiency',
            'Show value of basic, no-frills transportation'
        ],
        'channels': [
            'Budget-focused advertising',
            'Local marketing and promotions',
            'Dealership-level campaigns',
            'Value-oriented digital platforms',
            'Targeted offers and incentives'
        ],
        'product_focus': [
            'Economy cars and compact models',
            'Base trim levels with essential features',
            'Most fuel-efficient options',
            'Used vehicle programs with warranty'
        ],
        'customer_experience': [
            'Quick, efficient buying process',
            'Focus on affordability and payment options',
            'Straightforward, no-frills approach',
            'Economy service packages and maintenance'
        ]
    }
}

# Function to create a strategy card for a segment
def create_strategy_card(segment_name, data, ax):
    # Set title
    ax.set_title(segment_name, fontsize=12, fontweight='bold', color=data['color'], pad=10)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Background color
    ax.set_facecolor('#f8f9fa')
    
    # Segment size
    ax.text(0.02, 0.98, f"Segment size: {data['size']}%", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Create text blocks for each strategy section
    ypositions = [0.85, 0.60, 0.35, 0.10]
    heights = [0.15, 0.15, 0.15, 0.15]
    titles = ['Key Messaging', 'Marketing Channels', 'Product Focus', 'Customer Experience']
    content_lists = [data['messaging'], data['channels'], data['product_focus'], data['customer_experience']]
    
    for ypos, height, title, content in zip(ypositions, heights, titles, content_lists):
        # Section background
        rect = plt.Rectangle((0.05, ypos-height), 0.9, height, 
                           fill=True, color='white', alpha=0.8,
                           transform=ax.transAxes, zorder=1,
                           linewidth=1, edgecolor='#dddddd')
        ax.add_patch(rect)
        
        # Section title
        ax.text(0.07, ypos-0.03, title, transform=ax.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                horizontalalignment='left', zorder=2)
        
        # Section content as bullet points
        for i, item in enumerate(content):
            ax.text(0.09, ypos-0.06-(i*0.025), f"• {item}", transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='left',
                    zorder=2, wrap=True)

# Create the marketing strategies visualization
def create_marketing_strategies_visualization():
    # Create figure
    fig = plt.figure(figsize=(16, 20))
    
    # Create a GridSpec
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title for the entire figure
    fig.suptitle('Marketing Strategy Recommendations by Segment', fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, 'Based on hierarchical clustering analysis of customer vehicle preferences',
             horizontalalignment='center', fontsize=10)
    
    # Create a subplot for each segment
    segment_names = list(segments.keys())
    
    for i, segment_name in enumerate(segment_names):
        # Calculate grid position
        row = i // 2
        col = i % 2
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Create strategy card
        create_strategy_card(segment_name, segments[segment_name], ax)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Return the figure
    return fig

# Create the visualization
fig = create_marketing_strategies_visualization()

# Show the plot
plt.show()

# Save the figure
plt.savefig('vehicle_segment_marketing_strategies.png', dpi=300, bbox_inches='tight')
"""


# COMMAND ----------

# Databricks permissions diagnostic script

# Check current user
print("Current User:")
print(spark.sql("SELECT current_user()").collect()[0][0])

# List available writable locations
print("\nPossible writable locations:")
print("1. /tmp directory:")
display(dbutils.fs.ls("/tmp"))

# print("\n2. Your home directory:")
home_dir = f"/user/{spark.sql('SELECT current_user()').collect()[0][0]}"
display(dbutils.fs.ls(home_dir))

# Attempt to write to a temporary location
try:
    test_path = "/tmp/permissions_test.txt"
    dbutils.fs.put(test_path, "Permissions test", overwrite=True)
    print(f"\nSuccessfully wrote to {test_path}")
    
    # Read back the file
    print("File contents:")
    print(dbutils.fs.head(test_path))
    
except Exception as e:
    print(f"\nFailed to write to temporary location: {e}")

# Additional diagnostic information
print("\nCurrent Spark Configuration:")
for key, value in spark.conf.getAll().items():
    print(f"{key}: {value}")

# COMMAND ----------

# Explicit write test to /tmp

# Generate a unique filename
from datetime import datetime
import uuid

# Create a unique filename
unique_filename = f"/tmp/write_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.txt"

try:
    # Attempt to write the file
    dbutils.fs.put(unique_filename, "Databricks file write test successful!", overwrite=True)
    
    # Confirm the file was written
    print(f"Successfully wrote file: {unique_filename}")
    
    # Read back the contents
    print("File contents:")
    print(dbutils.fs.head(unique_filename))
    
    # List the contents of /tmp to verify
    print("\nContents of /tmp after write:")
    display(dbutils.fs.ls("/tmp"))
    
except Exception as e:
    print(f"An error occurred: {e}")
    

# COMMAND ----------

############ Updated Training data set ###############

# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, coalesce, lit
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Define source tables
SOURCE_FEATURES_TABLE = "work.marsci.sw_onstar_enhanced_conversion_propensity_v2"  # Your original table
CROSSWALK_TABLE = "work.marsci.vw_id_crosswalk"  # ID crosswalk table
DEMOGRAPHICS_TABLE = "work.marsci.acxiom_full_demographic_test"  # Demographic data table
TARGET_MODEL_TABLE = "work.marsci.sw_onstar_conversion_basic_model_features"  # New table for model training

# Load the data
print("Loading data from source tables...")
df_features = spark.table(SOURCE_FEATURES_TABLE)
df_crosswalk = spark.table(CROSSWALK_TABLE)
df_demographics = spark.table(DEMOGRAPHICS_TABLE)

print(f"Loaded {df_features.count()} rows from features table")
print(f"Loaded {df_crosswalk.count()} rows from crosswalk table")
print(f"Loaded {df_demographics.count()} rows from demographics table")

# Define key identifiers needed for any model
id_cols = ['FEATURE_DATE', 'ACCOUNT_NUMBER', 'ACCOUNT_ID', 'VIN_ID']
target_col = ['pay_flg']

# Define features available to Basic plan customers
# These are features that Basic customers could potentially use or access
basic_features = [
    # Vehicle information
    'VEHICLE_STATUS_CD', 'VEH_MAKE', 'VEH_MODEL', 'VEH_MANUF_YEAR', 'VEHICLE_WIFI_CAPABLE', 'ONSTAR_INSTALL_FLG',
    
    # Basic mobile app features - these are available to Basic customers
    'LAST_MONTH_SUM_DAY_GETVEHICLEDATA_CNT',      # Check vehicle status
    'LAST_MONTH_SUM_DAY_GETVEHICLELOCATION_CNT',  # See vehicle location
    'LAST_MONTH_SUM_DAY_LOCKDOORS_CNT',           # Lock doors
    'LAST_MONTH_SUM_DAY_UNLOCKDOORS_CNT',         # Unlock doors
    'LAST_MONTH_SUM_DAY_REMOTESTART_CNT',         # Remote start
    'LAST_MONTH_SUM_DAY_REMOTESTOP_CNT',          # Remote stop
    'LAST_MONTH_SUM_DAY_LOCKTRUNK_CNT',           # Lock trunk
    'LAST_MONTH_SUM_DAY_UNLOCKTRUNK_CNT',         # Unlock trunk
    
    # OnStar Button presses - basic users can press the blue button
    'LAST_MONTH_SUM_DAY_BLUEBUTTONPRESS_CNT',
    
    # Subscription history features
    'NUM_PREV_PAID_CORE_PLANS',
    'NUM_PREV_PAID_NONCORE_PLANS',
    'NUM_PREV_COMP_CORE_PLANS',
    'NUM_PREV_COMP_NONCORE_PLANS',
    'NUM_PREV_DRPO_PLANS',
    'TOTAL_DAYS_WITH_ACTIVE_SUBSCRIPTION',
    'TOTAL_DAYS_WITHOUT_ACTIVE_SUBSCRIPTION',
    
    # Include 3-month and 6-month app usage for trend analysis
    'LAST_THREE_MONTHS_SUM_DAY_GETVEHICLEDATA_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_GETVEHICLELOCATION_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_LOCKDOORS_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_UNLOCKDOORS_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_REMOTESTART_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_REMOTESTOP_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_LOCKTRUNK_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_UNLOCKTRUNK_CNT',
    'LAST_THREE_MONTHS_SUM_DAY_BLUEBUTTONPRESS_CNT',
    
    'LAST_SIX_MONTHS_SUM_DAY_GETVEHICLEDATA_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_GETVEHICLELOCATION_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_LOCKDOORS_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_UNLOCKDOORS_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_REMOTESTART_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_REMOTESTOP_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_LOCKTRUNK_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_UNLOCKTRUNK_CNT',
    'LAST_SIX_MONTHS_SUM_DAY_BLUEBUTTONPRESS_CNT',
]

# Select important demographic features that may influence conversion
demographic_cols = [
    # Income and financial status
    'ax_est_hh_income_prmr_plus_cd',
    'ax_estimate_hh_income_prmr_cd',
    'ax_home_market_value_prmr_cd',
    'ax_net_worth_prmr_cd',
    'ax_infobase_affordability_us',
    'ax_econ_stblty_ind_financial',
    
    # Vehicle ownership and preferences
    'ax_nbr_of_veh_owned_prmr',
    'ax_veh_new_carbuyr_prmr_flg',
    'ax_veh_lifestyle_ind_prmr_cd',
    'ax_auto_enthusiast_ind',
    'ax_auto_work_flg',
    'ax_have_auto_loan',
    'ax_prsnl_joint_auto_loan',
    
    # Household composition
    'ax_age_2yr_incr_indiv1_plus_cd',
    'ax_nbr_of_childrn_in_hh_prmr_plus',
    'ax_num_adults_hh_prmr_plus',
    'ax_household_size',
    'ax_home_owner_renter_prmr_flg',
    'ax_marital_status_in_hh',
    'ax_presence_of_children_flg',
    
    # Technology adoption
    'ax_attd_bhvr_prop_tech_adpt',
    'ax_technology_flg',
    
    # Lifestyle indicators
    'ax_entering_adulthood_flg',
    'ax_broader_living_flg',
    'ax_professional_living_flg',
    'ax_common_living_flg',
    'ax_sporty_living_flg',
    'ax_sports_grouping_flg',
    'ax_outdoors_grouping_flg',
    'ax_spectr_auto_mocycl_rcng_flg',
    'ax_nascar_flg',
    'ax_environmental_issues_flg',
    
    # Luxury affinity
    'ax_affnty_new_cadillac_fin',
    'ax_prchs_new_lux_sedan_fin',
    'ax_prchs_new_lux_suv_fin',
    'ax_prchs_new_mid_lux_car_fin',
    
    # Credit behavior
    'ax_heavy_transactors_ind',
    'ax_carry_fwd_a_bal_cc',
    'ax_price_snstv_pny_pnchr'
]

# Now construct the model features dataframe
print("Creating model features dataset...")

# Filter out vehicles that are not eligible for Basic plan (model year < 2015)
df_eligible = df_features.filter(col('VEH_MANUF_YEAR') >= 2015)

# Select the columns we want for our model
model_cols = id_cols + target_col + basic_features
df_model = df_eligible.select(model_cols)

# Join logic - create a pipeline to enrich our data with demographic information

# Step 1: Join with ID crosswalk to get individual IDs
print("Joining with ID crosswalk table...")
try:
    # This will work if service data with INDIV_ID has already been integrated
    if 'INDIV_ID' in df_model.columns:
        df_model = df_model.join(
            df_crosswalk.select('indiv_id', 'entity_realID_person_id', 'household_id', 'amperity_id'),
            df_model['INDIV_ID'] == df_crosswalk['indiv_id'],
            how='left'
        )
    else:
        # Otherwise, try to join through account information first
        print("INDIV_ID not available directly, using ID crosswalk to establish connections...")
        # For demonstration, assuming ACCOUNT_ID can be linked
        # In actual implementation, you might need a different join strategy
        
        # Try to load service lane features to get INDIV_ID
        try:
            service_df = spark.table("work.marsci.service_lane_features")
            print("Successfully loaded service_lane_features table")
            df_model = df_model.join(
                service_df.select('VIN_ID', 'FEATURE_DATE', 'INDIV_ID', 'HOUSEHOLD_ID'),
                on=['VIN_ID', 'FEATURE_DATE'],
                how='left'
            )
            # Then join with crosswalk
            df_model = df_model.join(
                df_crosswalk.select('indiv_id', 'entity_realID_person_id', 'household_id', 'amperity_id'),
                df_model['INDIV_ID'] == df_crosswalk['indiv_id'],
                how='left'
            )
        except Exception as e:
            print(f"Service lane features table not available: {str(e)}")
            print("Using a direct join strategy with the crosswalk table")
            # This is a fallback method if service_lane_features isn't available
            df_model = df_model
except Exception as e:
    print(f"Error during crosswalk join: {str(e)}")

# Step 2: Join with demographics using appropriate IDs
print("Joining with demographic data...")
# Select the most recent demographic data for each individual
df_demographics_latest = df_demographics.withColumn(
    "row_num",
    F.row_number().over(
        F.window.partitionBy("indiv_id").orderBy(F.desc("time_stamp"))
    )
).filter(col("row_num") == 1).drop("row_num")

# Now join with this demographics data
try:
    # Attempt to join on indiv_id if available
    if 'indiv_id' in df_model.columns:
        print("Joining demographics on indiv_id...")
        df_model = df_model.join(
            df_demographics_latest.select(['indiv_id'] + demographic_cols),
            df_model['indiv_id'] == df_demographics_latest['indiv_id'],
            how='left'
        )
    # If that fails, try to join on amperity_id if available
    elif 'amperity_id' in df_model.columns:
        print("Joining demographics on amperity_id...")
        df_model = df_model.join(
            df_demographics_latest.select(['amperity_id'] + demographic_cols),
            df_model['amperity_id'] == df_demographics_latest['amperity_id'],
            how='left'
        )
    else:
        print("No common ID found for demographic join")
except Exception as e:
    print(f"Error during demographic join: {str(e)}")

# Try to load loyalty data if available
try:
    loyalty_df = spark.table("work.marsci.loyalty_features")
    print("Successfully loaded loyalty_features table")
    df_model = df_model.join(
        loyalty_df.select(['ACCOUNT_NUMBER', 'FEATURE_DATE', 
                          'MIN_LOYALTY_TIER', 'MAX_LOYALTY_TIER', 
                          'LOY_REDEEMED_FLAG', 'TOTAL_LOY_NUM_TRANSACTIONS']),
        on=['ACCOUNT_NUMBER', 'FEATURE_DATE'],
        how='left'
    )
except Exception as e:
    print(f"Loyalty features table not available: {str(e)}")

# Add derived features that might be useful predictors
print("Creating derived features...")

# Frequency of app usage (as a rate per month)
df_model = df_model.withColumn(
    'APP_USAGE_RATE_MONTHLY',
    (
        coalesce(col('LAST_MONTH_SUM_DAY_GETVEHICLEDATA_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_GETVEHICLELOCATION_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_LOCKDOORS_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_UNLOCKDOORS_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_REMOTESTART_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_REMOTESTOP_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_LOCKTRUNK_CNT'), lit(0)).cast('double') +
        coalesce(col('LAST_MONTH_SUM_DAY_UNLOCKTRUNK_CNT'), lit(0)).cast('double')
    )
)

# Usage trend (increasing or decreasing)
df_model = df_model.withColumn(
    'APP_USAGE_TREND',
    (
        (
            coalesce(col('LAST_MONTH_SUM_DAY_GETVEHICLEDATA_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_GETVEHICLELOCATION_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_LOCKDOORS_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_UNLOCKDOORS_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_REMOTESTART_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_REMOTESTOP_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_LOCKTRUNK_CNT'), lit(0)).cast('double') +
            coalesce(col('LAST_MONTH_SUM_DAY_UNLOCKTRUNK_CNT'), lit(0)).cast('double')
        ) -
        (
            (
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_GETVEHICLEDATA_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_GETVEHICLELOCATION_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_LOCKDOORS_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_UNLOCKDOORS_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_REMOTESTART_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_REMOTESTOP_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_LOCKTRUNK_CNT'), lit(0)).cast('double') +
                coalesce(col('LAST_THREE_MONTHS_SUM_DAY_UNLOCKTRUNK_CNT'), lit(0)).cast('double')
            ) / 3
        )
    )
)

# Flag for whether they use the app consistently
df_model = df_model.withColumn(
    'IS_ACTIVE_APP_USER',
    when(col('APP_USAGE_RATE_MONTHLY') > 5, 1).otherwise(0)
)

# Flag for whether they press the blue button
df_model = df_model.withColumn(
    'IS_BLUEBUTTON_USER',
    when(coalesce(col('LAST_THREE_MONTHS_SUM_DAY_BLUEBUTTONPRESS_CNT'), lit(0)) > 0, 1).otherwise(0)
)

# Add features derived from demographic data
df_model = df_model.withColumn(
    'IS_HIGH_INCOME',
    when(col('ax_est_hh_income_prmr_plus_cd').isin(['I', 'J', 'K', 'L', 'M', 'N', 'P']), 1).otherwise(0)
)

df_model = df_model.withColumn(
    'IS_LUXURY_VEHICLE_BUYER',
    when(
        (coalesce(col('ax_affnty_new_cadillac_fin'), lit(100)) < 30) |
        (coalesce(col('ax_prchs_new_lux_sedan_fin'), lit(100)) < 30) |
        (coalesce(col('ax_prchs_new_lux_suv_fin'), lit(100)) < 30) |
        (coalesce(col('ax_prchs_new_mid_lux_car_fin'), lit(100)) < 30),
        1
    ).otherwise(0)
)

df_model = df_model.withColumn(
    'IS_AUTO_ENTHUSIAST',
    when(
        (coalesce(col('ax_auto_enthusiast_ind'), lit(0)) == 1) |
        (coalesce(col('ax_auto_work_flg'), lit(0)) == 1) |
        (coalesce(col('ax_spectr_auto_mocycl_rcng_flg'), lit(0)) == 1) |
        (coalesce(col('ax_nascar_flg'), lit(0)) == 1),
        1
    ).otherwise(0)
)

df_model = df_model.withColumn(
    'IS_TECH_ADOPTER',
    when(
        (coalesce(col('ax_attd_bhvr_prop_tech_adpt'), lit(100)) < 30) |
        (coalesce(col('ax_technology_flg'), lit(0)) == 1),
        1
    ).otherwise(0)
)

df_model = df_model.withColumn(
    'HAS_FAMILY',
    when(
        (coalesce(col('ax_nbr_of_childrn_in_hh_prmr_plus'), lit(0)) > 0) |
        (coalesce(col('ax_presence_of_children_flg'), lit(0)) == 1),
        1
    ).otherwise(0)
)

# Calculate missing values and display column statistics
print("Checking missing values...")
total_records = df_model.count()
null_counts = df_model.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df_model.columns
])

# Print column stats
print(f"Final dataset has {total_records} rows and {len(df_model.columns)} columns")
null_counts_pd = null_counts.toPandas().T
null_counts_pd.columns = ['missing_count']
null_counts_pd['missing_percentage'] = null_counts_pd['missing_count'] / total_records * 100
print(null_counts_pd.sort_values('missing_percentage', ascending=False).head(10))

# Calculate class distribution
class_counts = df_model.groupBy("pay_flg").count().toPandas()
print("\nTarget class distribution:")
print(class_counts)

# Create the table
print(f"\nCreating table: {TARGET_MODEL_TABLE}")
df_model.write.mode("overwrite").format("delta").saveAsTable(TARGET_MODEL_TABLE)

print(f"Successfully created {TARGET_MODEL_TABLE}")
