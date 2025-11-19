# Databricks notebook source
# MAGIC %md
# MAGIC # Training Pipeline
# MAGIC
# MAGIC Description:
# MAGIC
# MAGIC This notebook includes steps to build and register a churn classification model to MLflow and evaluates the model with MLflow Evaluate. 
# MAGIC
# MAGIC ## Steps
# MAGIC
# MAGIC 1. Import required libraries (Python Packages and internal utility functions)
# MAGIC 2. Assign widget parameters
# MAGIC 3. Load configuration file
# MAGIC 4. Load dataset
# MAGIC 5. Define model architecture
# MAGIC 6. Define hyperopt loss function
# MAGIC 7. Train model
# MAGIC 8. Save best run to mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Auto Reload
# MAGIC This next cell allows you to make changes to the library functions that are visible when the libraries are re-imported

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Import libraries

# COMMAND ----------

import os

os.environ["DISABLE_MLFLOWDBFS"] = "True"
from pprint import pprint
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    FeatureHasher,
    SQLTransformer,
    Imputer,
    RobustScaler,
    StringIndexer,
    OneHotEncoder,
)
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from edai_aai_mime_marsci.utils.ml.mlutils import (
    prepare_inference_results,
    gen_uspop_histogram,
    plot_lift_chart,
    calculate_lift_metrics
)

from edai_mlops_stacks.config import get_config, param_from_dab_or_widget, config_to_dict
from edai_mlops_stacks.etl import load_and_select_cols
# from edai_mlops_stacks.ml.spark_ml_wrappers import (
#     SparkMLPropensityWrapper,
#     SparkMLWrapper,
# )

from edai_aai_mime_marsci.utils.ml.ml_wrappers import PronghornPropensityWrapper

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Assign Widget Parameters
# MAGIC
# MAGIC The next cell saves the widget data as variables. This is important for deployment.
# MAGIC
# MAGIC The `param_from_dab_or_widget()` function can overwrite these values in production if necessary.

# COMMAND ----------

dbutils.widgets.dropdown(
    name="run_type", defaultValue="interactive", choices=["interactive", "batch"]
)
dbutils.widgets.dropdown(
    name="environment", defaultValue="dev", choices=["dev", "test", "prod"]
)

dbutils.widgets.text(name="target_col", defaultValue="target")

dbutils.widgets.dropdown(
    name="calibrate", defaultValue="False", choices=["True", "False"]
)

# COMMAND ----------

## Extract widget data
run_type = param_from_dab_or_widget("run_type")
environment = param_from_dab_or_widget("environment")
target_col = param_from_dab_or_widget("target_col")
calibrate = param_from_dab_or_widget("calibrate")

## Flag for running in interactive mode
interactive = run_type == "interactive"
calibrate = calibrate == "True"


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Configuration
# MAGIC The next cell gets the config file according to the selected environment (dev|prod)
# MAGIC This defines everything that we might want to change between runs.
# MAGIC Then we extract the configs for:
# MAGIC - feature table
# MAGIC
# MAGIC The `if interactive` check runs the commands inside if the notebook is set to `interactive` mode, but not when deployed in `batch` mode.

# COMMAND ----------

# load config as namedtuple from config json
config = get_config(
    env=environment,
    name="lease",
    repo="edai_aai_mime_marsci",

)


gold_config = config.gold.lease_model_data
feature_config = config.feature
model_config = config.models.lease_modelling

impute_cols = gold_config.impute_cols
flag_cols = gold_config.flag_cols
string_cols = gold_config.string_cols
feature_cols = gold_config.feature_cols

# if interactive:
#     print(model_config._asdict())


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Load gold table
# MAGIC

# COMMAND ----------

data = load_and_select_cols(gold_config)

# COMMAND ----------

# MAGIC %md
# MAGIC drop any label column that is not equal to the target_col specified by the widget

# COMMAND ----------

col_list = feature_cols + [target_col]
data = data.select(*col_list)

# COMMAND ----------

# if interactive:
#     display(data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Feature selection 

# COMMAND ----------

# imputer = Imputer(inputCols=impute_cols, outputCols=impute_cols, strategy="mean")

# numerical_assembler = VectorAssembler(
#     inputCols=impute_cols, outputCol="numerical_assembled"
# )
# numerical_scaler = RobustScaler(
#     inputCol="numerical_assembled", outputCol="numerical_scaled"
# )

# # String/Cat Indexer (will encode missing/null as separate index)
# string_cols_indexed = [c + "_index" for c in string_cols]
# string_indexer = StringIndexer(
#     inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep"
# )

# # OHE categoricals
# ohe_cols = [column + "_ohe" for column in string_cols]
# one_hot_encoder = OneHotEncoder(
#     inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep"
# )

# # Assemble vector
# feature_cols = ["numerical_scaled"] + flag_cols + ohe_cols
# feature_cols2 = ["numerical_scaled"] + flag_cols
# vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# # vector_assembler2 = VectorAssembler(inputCols=flag_cols, outputCol="test2")

# # Instantiate the pipeline
# stages_list = [
#     imputer,
#     numerical_assembler,
#     numerical_scaler,
#     string_indexer,
#     one_hot_encoder,
#     vector_assembler
# ]

# pipeline = Pipeline(stages=stages_list)

# COMMAND ----------

# xgb = SparkXGBClassifier(features_col="features", label_col=target_col)
# auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
# precision = MulticlassClassificationEvaluator(
#     labelCol=target_col, predictionCol = "prediction", metricName="precisionByLabel", metricLabel=1
# )
# recall = MulticlassClassificationEvaluator(
#     labelCol=target_col, predictionCol = "prediction", metricName="recallByLabel", metricLabel=1
# )
# f1 = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol = "prediction", metricName="fMeasureByLabel", metricLabel=1)

# COMMAND ----------

# xgb_model = xgb.fit(train_transformed_df)
# prediction_df = xgb_model.transform(test_transformed_df).cache()
# prediction_df.count()
# # prediction_df = spark.read.table('work.wzn583.lease_xgb_prediction_df')
# # print(f'auc: {auc.evaluate(prediction_df)}')
# # print(f'precision: {precision.evaluate(prediction_df)}')
# print(f'recall: {recall.evaluate(prediction_df)}')
# # print(f'f1: {f1.evaluate(prediction_df)}')

# COMMAND ----------

# result_df = prepare_inference_results(prediction_df, resolution='percentiles', reduced=False, verbose=False)
# lift_metrics = calculate_lift_metrics(predictions=result_df, target_col=target_col)
# plot_lift_chart(
#     lift_metrics,
#     # save_path_counts=lift_metrics_counts_save_path,
#     # save_path_lift=lift_metrics_save_path
# )

# COMMAND ----------

# MAGIC %md
# MAGIC This section is intentionally left blank for the data scientist to implement their preferred feature selection technique. You are free to choose any method that suits your use case.
# MAGIC
# MAGIC Our recommended approach is to encapsulate the entire feature selection process in a function. This function should take the dataset and candidate feature columns as input, and return a list of selected features. Ensure that any hardcoded values are moved to the configuration file for better flexibility and maintainability.
# MAGIC
# MAGIC Once you have applied the feature selection function to the data, create a new dataset as a subset of your original data, based on the selected features. Additionally, ensure that this new dataset includes columns that you want to retain but do not use for modeling. These non-modelling columns should be listed in the configuration notebook under `cols_excluded_from_modelling`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5. Training a model

# COMMAND ----------

# MAGIC %md
# MAGIC To effectively train a model, there are a few important steps to follow:
# MAGIC
# MAGIC - **Define the Model Pipeline**:
# MAGIC Start by defining the model pipeline, which should include all the necessary feature engineering transformations. These transformations might include steps such as scaling numeric features, one-hot encoding categorical variables, and any other preprocessing tasks needed for your data. The pipeline also includes the model it The entire Spark ML model pipeline  should be wrapped within a function that returns the pipeline object. This function can then be passed directly to the `SparkMLPropensityWrapper', which manages the training, evaluation, and logging processes.
# MAGIC
# MAGIC - **Handle Hyperparameter Tuning (Optional)**:
# MAGIC If you plan to perform hyperparameter tuning, you need to define the appropriate hyperopt loss functions. These functions are critical for guiding the hyperparameter optimization process and should be passed into the wrapper. 
# MAGIC
# MAGIC - **Integrate Custom Metrics:**
# MAGIC When working with custom evaluation metrics, ensure that these metrics are properly imported into your notebook. You should pass these custom metrics as a list into the `extra_metrics` argument of the `train_model()` method from the wrapper class `SparkMLPropensityWrapper`. Keep in mind that these custom metrics must be compatible with MLflow Evaluate. For more information on how to ensure compatibility, please refer to the relevant MLflow documentation at https://mlflow.org/docs/latest/models.html#performing-model-validation.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 Define model architecture

# COMMAND ----------

# def SparkXGB_pipeline(impute_cols: list, string_cols: list, flag_cols: list, target_col: str, model_params: dict = {}, pipeline_params: dict = {}) -> Pipeline:
#     """
#     Returns a Spark ML Pipeline for XGBoost 

#     Args:
#         impute_cols (list): List of numeric cols to impute nulls
#         string_cols (list): List of categorical columns for one hot encoding
#         flag_cols (list): List of flag columns to include in pipeline
#         target_col (str): Name of the target column
#         model_params (dict): Hyperparameters for XGBoost
#         pipeline_params (dict): Additional parameters for the pipeline

#     Returns:
#         Pipeline: Spark ML pipeline
#     """
#     imputer = Imputer(inputCols=impute_cols, outputCols=impute_cols, strategy="mean")

#     numerical_assembler = VectorAssembler(
#         inputCols=impute_cols, outputCol="numerical_assembled"
#     )
#     numerical_scaler = RobustScaler(
#         inputCol="numerical_assembled", outputCol="numerical_scaled"
#     )

#     # String/Cat Indexer (will encode missing/null as separate index)
#     string_cols_indexed = [c + "_index" for c in string_cols]
#     string_indexer = StringIndexer(
#         inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep"
#     )

#     # OHE categoricals
#     ohe_cols = [column + "_ohe" for column in string_cols]
#     one_hot_encoder = OneHotEncoder(
#         inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep"
#     )

#     # Assemble vector
#     feature_cols = ["numerical_scaled"] + flag_cols + ohe_cols
#     feature_cols2 = ["numerical_scaled"] + flag_cols
#     vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#     # Define model
#     xgb = SparkXGBClassifier(
#         features_col="features",
#         label_col=target_col,
#         **model_params
#     )

#     # Instantiate the pipeline
#     stages_list = [
#         imputer,
#         numerical_assembler,
#         numerical_scaler,
#         string_indexer,
#         one_hot_encoder,
#         vector_assembler,
#         xgb
#     ]

#     pipeline = Pipeline(stages=stages_list)

#     return(pipeline)

# COMMAND ----------

def SparkXGB_pipeline(data: DataFrame, feature_cols: list, target_col: str, model_params: dict = {}, pipeline_params: dict = {}) -> Pipeline:
    """
    Returns a Spark ML Pipeline for XGBoost with automatic feature engineering based on data types (numeric, categorical, and boolean).

    Args:
        data (DataFrame): Input data for detecting data types.
        feature_cols (list): List of feature columns.
        target_col (str): Name of the target column.
        model_params (dict): Hyperparameters for XGBoost.
        pipeline_params (dict): Additional parameters for the pipeline.

    Returns:
        Pipeline: Spark ML pipeline
    """
    

    # Initialize stages for the pipeline
    stages = []
    
    # Detect feature types based on the input data
    numeric_cols = []
    categorical_cols = []
    boolean_cols = []

    for col_name, dtype in data.dtypes:
        if col_name in feature_cols:
            if dtype in ['int', 'bigint', 'double', 'float']:
                numeric_cols.append(col_name)
            elif dtype == 'string':
                categorical_cols.append(col_name)
            elif dtype == 'boolean':
                boolean_cols.append(col_name)
    
    if numeric_cols:
        # Impute missing numeric values and scale
        imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols, strategy="mean")
        stages.append(imputer)
        numerical_assembler = VectorAssembler(
            inputCols=numeric_cols, outputCol="numerical_assembled"
        )
        stages.append(numerical_assembler)
        numerical_scaler = RobustScaler(
            inputCol="numerical_assembled", outputCol="numerical_scaled"
        )
        stages.append(numerical_scaler)

    if categorical_cols:
        # String/Cat Indexer (will encode missing/null as separate index)
        string_cols_indexed = [c + "_index" for c in string_cols]
        string_indexer = StringIndexer(
            inputCols=string_cols, outputCols=string_cols_indexed, handleInvalid="keep"
        )
        stages.append(string_indexer)

        # OHE categoricals
        ohe_cols = [column + "_ohe" for column in string_cols]
        one_hot_encoder = OneHotEncoder(
            inputCols=string_cols_indexed, outputCols=ohe_cols, handleInvalid="keep"
        )
        stages.append(one_hot_encoder)

    # Handle boolean columns by converting True/False to 1/0 using SQLTransformer
    if boolean_cols:
        for col_name in boolean_cols:
            sql_transformer = SQLTransformer(statement=f"SELECT *, int({col_name}) as {col_name}_int FROM __THIS__")
            stages.append(sql_transformer)

    # Assemble all feature columns
    assembled_feature_cols = ["numerical_scaled"] + ohe_cols + [f"{col_name}_int" for col_name in boolean_cols]

    # Final feature assembly
    if assembled_feature_cols:
        assembler = VectorAssembler(inputCols=assembled_feature_cols, outputCol="features", handleInvalid="keep")
        stages.append(assembler)

    # XGBoost Model
    model_xgb = SparkXGBClassifier(
        features_col="features",
        label_col=target_col,
        num_workers=data.sparkSession.sparkContext.defaultParallelism,
        **model_params
    )

    stages.append(model_xgb)

    # Create the pipeline with all stages
    pipeline = Pipeline(stages=stages)

    return pipeline


# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 Define hyperopt loss function (Required for hyperparameter tuning)
# MAGIC

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def log_loss_fn(prediction_df: DataFrame, target_col: str) -> float:
    """
    Returns log loss value. 

    Args:
        prediction_df (DataFrame): predictions dataframe from a model
        target_col (str): name of target column
    Returns:
        log_loss (float): log loss value

    """
    mcEvaluator = MulticlassClassificationEvaluator(metricName="logLoss",
                                                    labelCol=target_col,
                                                    predictionCol="prediction",
                                                    probabilityCol="probability")
    log_loss=mcEvaluator.evaluate(prediction_df)
        
    return log_loss

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.3 Initialize the classifier wrapper and train model (under construction)

# COMMAND ----------

# MAGIC %md
# MAGIC Example usage:
# MAGIC
# MAGIC - (1) instantiate the wrapper class with appropriate inputs
# MAGIC
# MAGIC `model_wrapper = SparkMLPropensityWrapper(config=model_config, data = dataset, pipeline_fn = SparkXGB_pipeline )`
# MAGIC
# MAGIC - (2) split the dataset into desired dataset types (e.g. [train, val, test] or [train, test]). For a custom split_dataset method, subclass SparkMLPropensityWrapper class and define the custom splitting logic in the new class.
# MAGIC
# MAGIC `model_wrapper.split_dataset()`  
# MAGIC         
# MAGIC
# MAGIC - (3) train the model (with hyperparameter tunning)
# MAGIC
# MAGIC `results = model_wrapper.train_model(hyperopt=True, hyperopt_loss_fn= log_loss_fn )`
# MAGIC
# MAGIC - (3) train the model (without hyperparameter tunning) 
# MAGIC
# MAGIC `results = model_wrapper.train_model()`  or
# MAGIC `results = model_wrapper.train_model(hyperopt=False)`
# MAGIC
# MAGIC - (3) train the model (without evaluation of val/test datasets)
# MAGIC
# MAGIC `results = model_wrapper.train_model(hyperopt=False , evaluation = False)`
# MAGIC
# MAGIC `results = model_wrapper.train_model(hyperopt=True, hyperopt_loss_fn= log_loss_fn , evaluation = False )`
# MAGIC
# MAGIC - (3) train the model (without hyperparameter tuninning and with extra/custom metrics)
# MAGIC
# MAGIC `from custom_metrics_module import (custom_metric1, custom_metric2)`
# MAGIC
# MAGIC `results = model_wrapper.train_model(hyperopt=False , extra_metrics = [custom_metric1 , custom_metric2])`
# MAGIC
# MAGIC - (4) calibrate the probabilities (make sure to enter the run_id of the model you wish to calibrate, if run_id is not specified, the last trained model will be calibrated)
# MAGIC
# MAGIC `model_wrapper.calibrate_propensity_model(run_id ='your-model-run-id')`
# MAGIC
# MAGIC please note that custom metrics should be compatible with mlflow.evaluate(), for further information
# MAGIC on defining such metrics please see https://mlflow.org/docs/latest/models.html#performing-model-validation

# COMMAND ----------

if interactive:
    # train_pipeline = SparkMLPropensityWrapper(model_config, data = data, pipeline_fn= SparkXGB_pipeline)
    train_pipeline = PronghornPropensityWrapper(model_config, data = data, pipeline_fn= SparkXGB_pipeline)
    # train_pipeline.experiment_name = train_pipeline.experiment_name
    # train_pipeline.experiment_fullname = train_pipeline.experiment_path + train_pipeline.experiment_name
    # train_pipeline.target_col = target_col
    # train_pipeline.stratify_col = target_col
    # train_pipeline.cols_excluded_from_modelling = train_pipeline.cols_excluded_from_modelling +[target_col]
    # train_pipeline.feature_cols = [ col  for col in train_pipeline.data.columns if col not in train_pipeline.cols_excluded_from_modelling]

# COMMAND ----------

if interactive:
    train_pipeline.split_dataset()

# COMMAND ----------

if interactive:
    # without hyperparameter tuning
    train_pipeline.train_model()
     
    # with hyperparameter tuning
    # train_pipeline.train_model(hyperopt=True, hyperopt_loss_fn = log_loss_fn) 

# COMMAND ----------

# if not interactive:
#     #train_pipeline = SparkMLPropensityWrapper(model_config, data = data, pipeline_fn= SparkXGB_pipeline)
#     train_pipeline = PronghornPropensityWrapper(model_config, data = data, pipeline_fn= SparkXGB_pipeline)
#     train_pipeline.experiment_name = target_col + '_'+ train_pipeline.experiment_name
#     train_pipeline.experiment_fullname = train_pipeline.experiment_path + train_pipeline.experiment_name
#     train_pipeline.target_col = target_col
#     train_pipeline.stratify_col = target_col
#     train_pipeline.cols_excluded_from_modelling = train_pipeline.cols_excluded_from_modelling +[target_col]
#     train_pipeline.feature_cols = [ col  for col in train_pipeline.data.columns if col not in train_pipeline.cols_excluded_from_modelling]
#     train_pipeline.split_dataset()
#     train_pipeline.train_model()
#     train_pipeline.calibrate_propensity_model()
#     dbutils.notebook.exit(True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 Probability Calibration

# COMMAND ----------

# MAGIC %md
# MAGIC Please enter the run ID of the model you wish to calibrate. If no run ID is provided, the most recently trained model will be calibrated by default. Keep in mind that this may differ from the model you intended to select, if you have recently performed hyperparameter tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC If the dataset is large enough, the isotonic regression method typically yields better-calibrated probabilities compared to the sigmoid method. However, in cases where the predicted probabilities show a sigmoid-like (S-shaped) distortion, sigmoid calibration may be more suitable. Therefore, we generally recommend isotonic calibration unless there is evidence of such distortion. Additionally, analytic methods are available to correct for biases in probability predictions, and these can be incorporated into the scoring pipeline. For more details, please refer to the following source: [https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf](url)

# COMMAND ----------

if calibrate:
    # train_pipeline.calibrate_propensity_model(run_id = "361cdba009264550b5974a9edcd55496")
    train_pipeline.calibrate_propensity_model()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 Tag best run

# COMMAND ----------

# MAGIC %md
# MAGIC Tag the best run in MLflow with a `best_model` tag. Input the `best_run_id` in the config file to identify the best model run. Runs tagged with `best_model` will automatically be registered and deployed via CI/CD upon a PR approval.

# COMMAND ----------

# if interactive:
#     # tag best run
#     train_pipeline.tag_best_run(
#         best_run_id="27a50267e92f46a8bd7ec806c41d7738"
#         )
