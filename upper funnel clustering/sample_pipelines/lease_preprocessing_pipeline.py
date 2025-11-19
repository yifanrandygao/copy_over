# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import BooleanType, DoubleType, IntegerType, LongType, StringType
from pyspark.ml.linalg import VectorUDT

## Imports from edai_mlops_stacks in artifactory
from edai_mlops_stacks.config import get_config
from edai_mlops_stacks.etl import load_and_select_cols
from edai_mlops_stacks.io import build_output_table

## Local class installed from utils
from edai_aai_mime_marsci.utils.savana import DqCheck
from edai_aai_mime_marsci.utils.ml.mlutils import drop_invalid_feature_cols


# COMMAND ----------

# MAGIC %md
# MAGIC Set up notebook widgets. This is important for deployment.

# COMMAND ----------

dbutils.widgets.dropdown(
    name="run_type", defaultValue="interactive", choices=["interactive", "batch"])
dbutils.widgets.dropdown(
    name="environment", defaultValue="dev", choices=["dev", "test", "prod"])
dbutils.widgets.dropdown(
    name="model_type", defaultValue="inference", choices=["train", "inference"])

# COMMAND ----------

# MAGIC %md
# MAGIC The next cell saves the widget data as variables. 

# COMMAND ----------

## Extract widget data

run_type = dbutils.widgets.get("run_type")
environment = dbutils.widgets.get("environment")
model_type = dbutils.widgets.get("model_type")

## Flag for running in interactive mode
interactive = run_type == "interactive"

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration
# MAGIC
# MAGIC The next cell gets the config file according to the selected environment (dev|prod)
# MAGIC This defines everything that we might want to change between runs.
# MAGIC
# MAGIC
# MAGIC
# MAGIC The `if interactive` check runs the commands inside if the notebook is set to `interactive` mode, but not when deployed in `batch` mode.
# MAGIC

# COMMAND ----------

config = get_config(env = environment, 
                    name="lease", 
                    repo="edai_aai_mime_marsci")


gold_table_configs = config.gold
acxiom_config = gold_table_configs.acxiom_input_data
polk_config = gold_table_configs.polk_input_data
id_config = gold_table_configs.id_xref_data
output_config = gold_table_configs.lease_model_data

if interactive:
    print(polk_config)
    print(output_config)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data processing function
# MAGIC
# MAGIC These functions can be left in the notebook or can be moved into .py files in `utils` for reusabilty and testability.

# COMMAND ----------

# TODO: refactor to be based on real ID rather than indiv_id once the nulls have been fixed
def process_lease_data(
    acxiom_config, polk_config, id_config, output_config, model_type
):
    """
    Processes the raw input data for the lease model.

    Args:
        input_config: config dict for input table
        output_config: config dict for output table
    Returns:
        A spark dataframe
    """
    ## configs
    acxiom_timestamp = output_config.acxiom_timestamp
    target_start_dt = output_config.target_start_dt
    target_end_dt = output_config.target_end_dt
    lease_cols = output_config.lease_cols
    recode_bool_cols = output_config.recode_bool_cols
    string_cols = output_config.string_cols
    flag_cols = output_config.flag_cols

    # filters
    acxiom_filter = (F.col("time_stamp") == acxiom_timestamp) & (
        ~F.col("indiv_id").isNull()
    ) & (~F.col("gm_person_real_id").isNull())
    acquisition_filter = ((F.col("acquisition") == 1) & (~F.col("gm_person_real_id").isNull()))  # vehicle sale in current month and id not null

    window_spec = Window.partitionBy("gm_person_real_id").orderBy(F.col("time_stamp").desc())

    # datasets
    acxiom_df = (
        load_and_select_cols(acxiom_config)
        .filter(acxiom_filter)
        .withColumn("row_num",F.row_number().over(window_spec))
        .filter(F.col("row_num")==1)
        .drop("indiv_id","row_num")
    )
    polk_df = (
        load_and_select_cols(polk_config)
        .filter(acxiom_filter)
        .withColumn("row_num",F.row_number().over(window_spec))
        .filter(F.col("row_num")==1)
        .drop("time_stamp", "indiv_id", "amperity_id", "pronghorn_id","row_num")
    )  # uses same filter condition as acxiom
    for col in recode_bool_cols:
        polk_df = polk_df.withColumn(col,F.when(F.col(col)==True,F.lit(1)).otherwise(F.lit(0)))

    id_xref = load_and_select_cols(id_config)
    hh_df = id_xref.groupBy("gm_person_real_id").agg(F.max("household_id").alias("household_id"))

    # create lease data
    # counts all leases prior to cut-off and creates flag for leases in six months following cut-off
    lease_df = (
        load_and_select_cols(polk_config)
        .select(*lease_cols)
        .filter(acquisition_filter)
        .withColumn(
            "prior_acquisition",
            F.when((F.col("time_stamp") <= acxiom_timestamp), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "prior_lease",
            F.when(((F.col("time_stamp") <= acxiom_timestamp) & (F.col("latest_vehicle_delivery_is_lease") == "true")), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "prior_sale",
            F.when(((F.col("time_stamp") <= acxiom_timestamp) & (F.col("latest_vehicle_delivery_is_lease") == "false")), F.lit(1)).otherwise(F.lit(0)),
        )
        
    )
    if model_type == "train":
        lease_df = (
            lease_df.withColumn(
                "target",
                F.when(((F.col("time_stamp").between(target_start_dt,target_end_dt))&(F.col("latest_vehicle_delivery_is_lease") == "true")), F.lit(1)).otherwise(F.lit(0)),
            )
        )
    elif model_type == "inference":
        lease_df = lease_df.withColumn("target", F.lit(0)) # placeholder for aggregation
    lease_df = (
        lease_df.groupBy("gm_person_real_id")
            .agg(
                F.max("indiv_id").alias("indiv_id"),
                F.max("amperity_id").alias("amperity_id"),
                F.max("pronghorn_id").alias("pronghorn_id"),
                F.sum("prior_acquisition").alias("prior_acquisition_count"),
                F.sum("prior_lease").alias("prior_lease_count"),
                F.sum("prior_sale").alias("prior_sale_count"),
                F.max("target").alias("target"),
            )
    )

    # combine data
    output_df = (
        lease_df.join(hh_df, ["gm_person_real_id"], "left")
        .join(acxiom_df, ["gm_person_real_id"], "left")
        .join(polk_df, ["gm_person_real_id"], "left")
        .filter(F.col("prior_acquisition_count") > 0) # removes future customers (after training data timestamp)
    )
    output_df = output_df.fillna(0, flag_cols)

    # update data types for modeling pipeline
    for col in string_cols:
        output_df = output_df.withColumn(col, F.col(col).cast(StringType()))
    for col in flag_cols:
        output_df = output_df.withColumn(col, F.col(col).cast(BooleanType()))

    if model_type == "inference":
        output_cols = output_config.id_cols + output_config.feature_cols + ["time_stamp"]
        output_df = output_df.select(*output_cols)

    return output_df

# COMMAND ----------

# MAGIC %md
# MAGIC Processing the data and returning a dataframe

# COMMAND ----------

output_df = process_lease_data(acxiom_config, polk_config, id_config, output_config, model_type).cache()
# if interactive:
#     display(output_df.limit(20)) 

# COMMAND ----------

output_df.columns

# COMMAND ----------

# display(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Quality checking
# MAGIC
# MAGIC The DQ contract for this file is stored in `configs/data_quality/gold_affinity_model_data_if_dq.yaml`
# MAGIC
# MAGIC

# COMMAND ----------

dq = DqCheck(sdf = output_df, 
             source = output_config.table.table)
output_df = dq.contract(commit = True)

# COMMAND ----------

# MAGIC %md
# MAGIC Writes the final delta table to Unity Catalog 

# COMMAND ----------

# help(build_output_table)

# COMMAND ----------

build_output_table(output_df, output_config)

# COMMAND ----------


