# Databricks notebook source
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from aztek.dbx.tools import get_user
from edai_mlops_stacks.config import set_config, import_configs
from pathlib import Path

# COMMAND ----------

# set default values - most common use case is monthly inferencing
default_filter_month = (date.today().replace(day=1) - relativedelta(months=1)).strftime("%Y-%m-%d")
default_output_table = "lease_propensity_inference"

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("filter_month",default_filter_month)
dbutils.widgets.text("output_table",default_output_table)

# COMMAND ----------

output_table = dbutils.widgets.get("output_table")
filter_month = dbutils.widgets.get("filter_month") # month for inferencing or training
target_start_month = (datetime.strptime(filter_month,"%Y-%m-%d").date() + relativedelta(months=1)).strftime("%Y-%m-%d")
target_end_month = (datetime.strptime(filter_month,"%Y-%m-%d").date() + relativedelta(months=6)).strftime("%Y-%m-%d")

# COMMAND ----------

# MAGIC %md
# MAGIC This Pipeline defines how you can construct a repo-level config file that is consumed by the other pipeline notebooks.
# MAGIC
# MAGIC There are several config segments that get combined at the end of the notebook:
# MAGIC
# MAGIC - source_tables_config: location of base assets for the project (assumed readonly)
# MAGIC - bronze_config: Config for any bronze layer tables
# MAGIC - silver_config: Config for any silver layer tables
# MAGIC - gold_config: Config for any gold layer tables
# MAGIC - models_config: Config for any ML models (In development)
# MAGIC - validation_config: unspecified dict for any validation parameters
# MAGIC
# MAGIC
# MAGIC The table configs all expect parameters that match Savana configs, and parameters that can be used by the `build_output_table()` function in `utils.io`. These are:
# MAGIC
# MAGIC - catalog: the catalog in UC
# MAGIC - schema: the schema in UC
# MAGIC - table: the table name in UC
# MAGIC - columns: list of columns to select
# MAGIC - rename: dict of {<column name> : <new column names>, ...} To enable consistent renaming of columns
# MAGIC - evolution: Is schema evolution enabled?
# MAGIC - cluster: list of column names to liquid cluster on
# MAGIC - mode: append|overwrite etc.
# MAGIC - merge_schema: True / False

# COMMAND ----------

# MAGIC %md #Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source tables

# COMMAND ----------

source_tables_config = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze layer

# COMMAND ----------

bronze_config = {}

# COMMAND ----------

# MAGIC %md ## Silver layer

# COMMAND ----------

silver_config = {}

# COMMAND ----------

# MAGIC %md ## Gold layer

# COMMAND ----------

gold_config = {
    "acxiom_input_data": {
        "table": {
            "catalog": "work",
            "schema": "marsci",
            "table": "output_develop_fs_acxiom_demo_uspop",
            "columns": [
                "indiv_id",
                "time_stamp",
                "ax_age_2yr_incrt_cd",
                "ax_credit_card_ind_prmr_cd",
                "ax_est_hh_income_prmr_plus_cd",
                "ax_first_veh_make_cd",
                "ax_first_veh_model_cd",
                "ax_gm_possessor_cd",
                "ax_home_market_value_prmr_cd",
                "ax_home_owner_renter_prmr_flg",
                "ax_input_indiv_gender_prmr_cd",
                "ax_marital_status_in_hh_prmr_plus_cd",
                "ax_nbr_of_childrn_in_hh_prmr_plus",
                "ax_nbr_of_veh_owned_prmr",
                "ax_net_worth_prmr_cd",
                "ax_nielsen_dma_cd",
                "ax_num_adults_hh_prmr_plus",
                "ax_veh_truck_rv_prmr_cd",
                "ax_acqrd_prsnl_joint_auto_loan",
                "ax_acqrd_prsnl_joint_mortgage",
                "ax_affnty_new_audi",
                "ax_affnty_new_bmw_fin",
                "ax_affnty_new_cadillac_fin",
                "ax_affnty_new_jaguar_fin",
                "ax_affnty_new_land_rover_fin",
                "ax_affnty_new_lexus_fin",
                "ax_affnty_new_lincoln_fin",
                "ax_affnty_new_mercedes_benz_fin",
                "ax_affnty_new_porsche_fin",
                "ax_attd_bhvr_prop_tech_adpt",
                "ax_auto_work_flg",
                "ax_car_dlr_mfg_dlr_wrnty_club",
                "ax_carry_fwd_a_bal_cc",
                "ax_curr_drive_rec_veh",
                "ax_curr_drive_suv",
                "ax_econ_stblty_ind_financial",
                "ax_edu_details_input_indiv_cd",
                "ax_employment_status_cd",
                "ax_have_auto_loan",
                "ax_heavy_transactors_ind",
                "ax_hh_estimated_income_100_pct_cd",
                "ax_household_size",
                "ax_in_market_new_rglr_veh",
                "ax_infobase_affordability_us",
                "ax_likely_buy_first_house_nxt_yr",
                "ax_outdoors_grouping_flg",
                "ax_prchs_new_hybrid_lux_fin",
                "ax_prchs_new_lux_4wd_fin",
                "ax_prchs_new_lux_awd_fin",
                "ax_prchs_new_lux_convrtble_fin",
                "ax_prchs_new_lux_cuv_fin",
                "ax_prchs_new_lux_sedan_fin",
                "ax_prchs_new_lux_sports_car_fin",
                "ax_prchs_new_lux_suv_fin",
                "ax_prchs_new_mid_lux_car_fin",
                "ax_prchs_new_mid_suv_fin",
                "ax_prchs_new_prem_lux_car_fin",
                "ax_presence_of_children_flg",
                "ax_price_snstv_pny_pnchr",
                "ax_prsnl_joint_auto_loan",
                "ax_ptntl_potntl_civic_infln",
                "ax_sports_grouping_flg",
                "ax_state_abbreviation",
                "ax_stocks_bonds_investor_cd",
                "ax_veh1_make_cd",
                "ax_veh1_type_cd",
                "ax_population_density_cd",
                "gm_person_real_id",
            ],
            "evolution": False,
            "cluster": [],
            "mode": "overwrite",
            "rename": {},
        },
    },
    "polk_input_data": {
        "table": {
            "catalog": "work",
            "schema": "marsci",
            "table": "output_develop_fs_polk_gm_uspop",
            "columns": [
                "indiv_id",
                "time_stamp",
                "pronghorn_id",
                "amperity_id",
                "latest_purchased_vehicle_new",
                "latest_purchased_segment_pickup",
                "avg_msrp_all_current_vehicles",
                "latest_purchased_segment_van",
                "second_latest_vehicle_delivery_is_lease",
                "latest_make_cd",
                "latest_purchased_segment_truck",
                "latest_model_cd",
                "latest_vehicle_status_cd",
                "latest_vehicle_delivery_is_lease",
                "latest_purchased_segment_suv",
                "latest_purchased_segment_suv_ev",
                "latest_purchased_segment_suv_ice",
                "latest_purchased_vehicle_year",
                "latest_purchased_segment_luxury_car",
                "latest_purchased_segment_demandspaces_mainstream_suv",
                "latest_purchased_segment_ev",
                "latest_purchased_segment_luxury_suv",
                "latest_purchased_segment_demandspaces_pickup",
                "latest_purchased_vehicle_msrp",
                "latest_purchased_segment_car",
                "latest_purchased_segment_demandspaces_luxury_suv",
                "latest_purchased_segment_sport",
                "acquisition",
                "latest_vehicle_age_years",
                "months_since_last_purchase",
                "years_since_last_purchase",
                "month_of_purchase",
                "ld_bool_lead_to_owner",
                "ld_is_active_lead",
                "gm_person_real_id",
            ],
            "evolution": False,
            "cluster": [],
            "mode": "overwrite",
            "rename": {},
        },
    },
    "id_xref_data": {
        "table": {
            "catalog": "dataproducts_dev",
            "schema": "bronze_acxiom",
            "table": "gm_owners",
            "columns": ["INDIV_BUSINESS_ID", "entity_realID_person_id", "HH_ID"],
            "evolution": False,
            "cluster": [],
            "mode": "overwrite",
            "rename": {
                "INDIV_BUSINESS_ID": "INDIV_ID",
                "entity_realID_person_id": "gm_person_real_id",
                "HH_ID": "household_id"
            },
        },
    },
    "lease_model_data": {
        "table": {
            "catalog": "work",
            "schema": "marsci",
            "table": output_table,
            "columns": [],
            "evolution": False,
            "cluster": [],
            "mode": "overwrite",
            "rename": {},
        },
        "acxiom_timestamp": filter_month,
        "target_start_dt": target_start_month,
        "target_end_dt": target_end_month,
        "id_cols": ["indiv_id", "gm_person_real_id", "household_id", "amperity_id", "pronghorn_id"],
        "ignore_cols": [],
        "target_col": "target",
        "lease_cols": [
            "gm_person_real_id", 
            "indiv_id",
            "amperity_id",
            "pronghorn_id",
            "time_stamp", 
            "acquisition", 
            "latest_vehicle_delivery_is_lease",
        ],
        "recode_bool_cols": [
            "latest_purchased_vehicle_new",
            "latest_vehicle_delivery_is_lease",
            "second_latest_vehicle_delivery_is_lease",
        ],
        "impute_cols": [
            "prior_sale_count",
            "prior_acquisition_count",
            "ax_est_hh_income_prmr_plus_cd",
            "ax_net_worth_prmr_cd",
            "months_since_last_purchase",
        ],
        "flag_cols": [       
            "latest_purchased_vehicle_new",
            "latest_purchased_segment_pickup",
            "second_latest_vehicle_delivery_is_lease",
            "latest_purchased_segment_truck",
            "latest_vehicle_delivery_is_lease",
            "latest_purchased_segment_suv",
            "latest_purchased_segment_suv_ice",
            "latest_purchased_segment_ev",
            "latest_purchased_segment_luxury_suv",
        ],
        "string_cols" : [
            "ax_population_density_cd",
            "latest_vehicle_status_cd",
            "ax_marital_status_in_hh_prmr_plus_cd",
            "ax_nielsen_dma_cd",
        ],
        "feature_cols" : [
            "prior_sale_count",
            "prior_acquisition_count",
            "ax_est_hh_income_prmr_plus_cd",
            "ax_net_worth_prmr_cd",
            "months_since_last_purchase",
            "latest_purchased_vehicle_new",
            "latest_purchased_segment_pickup",
            "second_latest_vehicle_delivery_is_lease",
            "latest_purchased_segment_truck",
            "latest_vehicle_delivery_is_lease",
            "latest_purchased_segment_suv",
            "latest_purchased_segment_suv_ice",
            "latest_purchased_segment_ev",
            "latest_purchased_segment_luxury_suv",
            "ax_population_density_cd",
            "latest_vehicle_status_cd",
            "ax_marital_status_in_hh_prmr_plus_cd",
            "ax_nielsen_dma_cd",
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature table

# COMMAND ----------

feature_config = {}


# COMMAND ----------

# MAGIC %md # Models
# MAGIC
# MAGIC Models can be defined in this config as shown in the next cell.
# MAGIC
# MAGIC The following cell gives a pattern to import multiple model configs from another location

# COMMAND ----------

models_config = {
    "lease_modelling": {
        "mlflow": {
            "experiment_path": "/"
            + str(Path(*Path.cwd().parents[2].parts[2:]) / "lease_experiments")
            + "/",
            "experiment_name": "lease_experimentation",
            "run_name": "Models/lease_modelling",
            "registry": "UC",
            "catalog": "embeddings",
            "schema": "public",
            "model_name": "lease_model",
            "tags": {},
            "mlflow_evaluate": {
                "evaluate_over": ["test"],
                "evaluators": ["default"],
                "evaluator_config": {
                    "default": {
                        "log_model_explainability": False,
                    },
                },
            },
        },
        "model_settings": {
            "model_type": "classifier",
            "target_col": "target",
            "cols_excluded_from_modelling": [
                "gm_person_real_id",
                "time_stamp",
                "indiv_id",
                "household_id",
                "amperity_id",
                "pronghorn_id",
                "target",
            ],
            "dataset_types": ["train", "test", "val"],
            "test_size": 0.1,
            "val_size": 0.2,
            "cache": {
                "train": False,
                "val": True,
                "test": True,
            },
            "sampling": {
                "stratify": False,
                "stratify_col": "target",
                "sample_frac": 0.5,
                "stratify_col_levels": [0, 1],
            },
            "class_balancing": {
                "balance_factor": 1,  # new class balance, if set to 1 then you have a balanced dataset, if set to 2, the negative/positive class ratio would be 2. to retain as much samples from the negative class, the balance factor can be set to > 1 and re-weighting can be performed via model parameters if possible (for example, scale_pos_weight can be set to the new class balance in the xgboost model to give more weight to the positive class)
            },
            "pipeline_parameters": {
                "cross_validation": {
                    "numFolds": 3,
                },
            },
            "static_hyperparameters": {
                "eval_metric": "logloss",
            },
            "hyperopt_parameters": {
                "hyper_parameters": {
                    "max_depth": {
                        "stochastic_expression_type": "quniform",
                        "low": 5,
                        "high": 12,
                        "q": 1,
                    },
                    "colsample_bytree": {
                        "stochastic_expression_type": "uniform",
                        "low": 0.5,
                        "high": 1.0,
                    },
                    "learning_rate": {
                        "stochastic_expression_type": "uniform",
                        "low": 0.01,
                        "high": 0.1,
                    },
                    "n_estimators": {
                        "stochastic_expression_type": "quniform",
                        "low": 50,
                        "high": 200,
                        "q": 10,
                    },
                    "subsample": {
                        "stochastic_expression_type": "uniform",
                        "low": 0.6,
                        "high": 1.0,
                    },
                    "gamma": {
                        "stochastic_expression_type": "uniform",
                        "low": 0,
                        "high": 5,
                    },
                },
                "search_algorithm": "tpe",
                "max_evals": 5,
            },
        },
        "calibration_method": "isotonic",  # options include "sigmoid" and "isotonic"
    },
    "run_configs": {
        "labels": ["target"],
        "run": {
            "notebook": str(
                Path.cwd().parents[1]
                / "edai_aai_mime_marsci/models/lease/lease_model"
            ),
            "timeout": 80000,
            "max_retries": 3,
            "args": {
                "environment": "dev",
                "run_type": "batch",
                "target_col": "target",
                "calibrate": "True",
            },
        },
    },
}


# COMMAND ----------

# Double check the mlflow experiment path does not include Workspace/
models_config["lease_modelling"]["mlflow"]["experiment_path"]

# COMMAND ----------

# MAGIC %md # Output Full config
# MAGIC
# MAGIC ## Ensure you change the name and repo arguments in cell 18 to match your repo

# COMMAND ----------

config = {
    "source_tables": source_tables_config,
    "bronze": bronze_config,
    "silver": silver_config,
    "gold": gold_config,
    "feature": feature_config,
    "models": models_config,
}

# COMMAND ----------

set_config(config, env="dev", name="lease", repo="edai_aai_mime_marsci")

# COMMAND ----------


