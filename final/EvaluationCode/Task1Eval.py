from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json
import pandas as pd
import os
def evaluate(json_file_path,ground_truth_file):
  y_truth = load_json(ground_truth_file)
  y_pred = load_json(json_file_path)
  if len(y_truth)!=len(y_pred):
    raise Error('num of pred result incorrect!')
  spearmanr_corr = stats.spearmanr(y_truth, y_pred)[0]
  mse=mean_squared_error(y_truth, y_pred)
  mae=mean_absolute_error(y_truth, y_pred)
  print("%s: SPEARMAN: %.4f, MSE: %.4f, MAE: %.4f"%(json_file_path,spearmanr_corr,mse,mae))



def load_json(json_file_path):
  json_str = open(json_file_path,'r').read()
  json_result = json.loads(json_str)['result']
  result = {"post_id":[],"popularity_score":[]}
  for row in json_result:
    result["post_id"].append(row["post_id"])
    result["popularity_score"].append(row["popularity_score"])
  dataframe = pd.DataFrame(result)
  return dataframe.sort('post_id')['popularity_score'].tolist()

# use this method as main method
# directory: a directory contains all candidates' json submission
# ground_truth_file: a json file contains the ground truth
def evaluate_jsons(directory,ground_truth_file):
  for filename in os.listdir(directory):
    if filename.endswith(".json"):
        json_file_path = (os.path.join(directory, filename))
        evaluate(json_file_path,ground_truth_file)


# EXAMPLE USAGE
evaluate_jsons('../EvaluationJsonExample/Task1','../EvaluationJsonExample/task1.json')