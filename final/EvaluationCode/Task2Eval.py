from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json
import pandas as pd
import os

def calc_hit_rate(pred_df,truth_df,hit_top):
  pred_pids = set(pred_df.sort('ranking_position')['post_id'].tolist()[:hit_top])
  truth_pids = set(truth_df.sort('ranking_position')['post_id'].tolist()[:hit_top])
  intersection = len(pred_pids&truth_pids)
  return float(intersection)/float(hit_top)

def calc_mae(pred_df,truth_df):
  truth_pops = truth_df.sort('post_id')['popularity_score'].tolist()
  pred_pops = pred_df.sort('post_id')['popularity_score'].tolist()
  mae=mean_absolute_error(truth_pops, pred_pops)
  return mae

def calc_spearman(pred_df,truth_df):
  truth_pops = truth_df.sort('post_id')['popularity_score'].tolist()
  pred_pops = pred_df.sort('post_id')['popularity_score'].tolist()
  spearmanr_corr = stats.spearmanr(truth_pops, pred_pops)[0]
  return spearmanr_corr

#MHR, MAE, Spearmanr's Rho
def evaluate(json_file_path,ground_truth_file,hit_top=5):
  pred_df = load_json(json_file_path)
  truth_df = load_json(ground_truth_file)
  hit_rate = calc_hit_rate(pred_df,truth_df,hit_top)
  mae = calc_mae(pred_df,truth_df)
  spearmanr = calc_spearman(pred_df,truth_df)
  print("%s: hit rate: %.4f, MAE: %.4f,Spearman: %.4f"%(json_file_path,hit_rate,mae,spearmanr))

def load_json(json_file_path):
  json_str = open(json_file_path,'r').read()
  json_result = json.loads(json_str)['result']
  result = {"post_id":[],"popularity_score":[],"ranking_position":[]}
  for row in json_result:
    result["post_id"].append(row["post_id"])
    result["popularity_score"].append(row["popularity_score"])
    result["ranking_position"].append(row["ranking_position"])
  dataframe = pd.DataFrame(result)
  return dataframe

# use this method as main method
# directory: a directory contains all candidates' json submission
# ground_truth_file: a json file contains the ground truth
def evaluate_jsons(directory,ground_truth_file):
  for filename in os.listdir(directory):
    if filename.endswith(".json"):
        json_file_path = (os.path.join(directory, filename))
        evaluate(json_file_path,ground_truth_file)

# EXAMPLE USAGE
evaluate_jsons('../EvaluationJsonExample/Task2','../EvaluationJsonExample/task2.json')