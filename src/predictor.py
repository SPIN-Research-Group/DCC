import os
import re
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

TRAIN_SPLIT = 0.2
workload_type = "3D"

def apply_get_num(data, i, j):
    if isinstance(data, list):
        return data[i][j]
    else:
        ls = json.loads(data)
        return ls[i][j]

def get_running_time_ms(cycles):
    return 0.769 * cycles / 1000 / 1000

def parse_output(output_str):
  results = {"cycles":-1, "LD": -1, "ST": -1, "MAC_AB": -1, "WR_GB": -1, "MV_SB": -1,}
  for line in output_str.splitlines():
    if "memory_system_cycles" in line:
      results["cycles"] = int(line.split(":")[1])
    elif "total_num_read_requests" in line:
      results["LD"] = int(line.split(":")[1])
    elif "total_num_write_requests" in line:
      results["ST"] = int(line.split(":")[1])
    elif "total_num_pim_write_to_gemv_buffer_requests" in line:
      results["WR_GB"] = int(line.split(":")[1])
    elif "total_num_pim_mac_all_bank_requests" in line:
      results["MAC_AB"] = int(line.split(":")[1])
    elif "total_num_pim_move_to_softmax_buffer_requests" in line:
      results["MV_SB"] = int(line.split(":")[1])
  return results


def read_data(path):
  global workload_type
  trace_list = []
  cnt = 0
  for f_name in os.listdir(path):
    if f_name.endswith('.log'):
      f_name_parts = f_name.split('_')
      numbers = re.findall("\d+", f_name_parts[0])
      tiling = f_name_parts[1].replace("-", ",")
      tiling = json.loads(tiling)
      f = open(os.path.join(path, f_name), "r")
      output_str = f.read()
      results = parse_output(output_str)
      if results["cycles"] != -1:
        data_dict = {
          "tiling": tiling,
          "f_name": f_name.replace(".log", ""),
          "results": results,
        }
        if workload_type == "2D":
          ba, vec_len = numbers
          data_dict["batch"] = int(ba)
          data_dict["vec_len"] = int(vec_len)
        elif workload_type == "3D":
          ba, nh, dh, seqL = numbers
          data_dict["batch"] = int(ba)
          data_dict["nhead"] = int(nh)
          data_dict["dhead"] = int(dh)
          data_dict["seq_len"] = int(seqL)
        trace_list.append(data_dict)
      else:
        cnt += 1

  if cnt > 0:
    print(f"{cnt} log files read failed in {path}")
  return trace_list

def save_to_df(data, path):
  def add_data(k, v, res):
    if k in res:
      res[k].append(v)
    else:
      res[k] = [v]

  pd_dict = {}
  for data_item in data:
    for k, v in data_item.items():
      if isinstance(v, dict):
        for k1, v1 in v.items():
          key = f"{k}_{k1}"
          add_data(key, v1, pd_dict)
      else:
        add_data(k, v, pd_dict)
  df = pd.DataFrame.from_dict(pd_dict)
  df.sort_values(by=["f_name"], inplace=True)
  return df

def parse_data(log_dir="./logs"):
  trace_info_list = read_data(log_dir)
  output_path = log_dir.split("/")[-2] + ".csv"
  return save_to_df(trace_info_list, output_path)

def data_processing(df, L_filter = 0):
    global workload_type
    if L_filter != 0:
        df = df[df["seq_len"] == L_filter]
    T_names = []
    if workload_type == "2D":
        range_i = 2
        group_cols = ["batch", "vec_len"]
    elif workload_type == "3D":
        range_i = 3
        group_cols = ["batch","nhead","dhead","seq_len"]
    for i in range(range_i):
        for j in range(3):
            col_name = f"T_{i}_{j}"
            df[col_name] = df["tiling"].apply(apply_get_num, args=(i, j))
            T_names.append(col_name)
    grouped_dfs = {}
    for group_keys, group_df in df.groupby(group_cols):
        grouped_dfs[group_keys] = group_df.copy().reset_index()
    return grouped_dfs, T_names

def test_performance(df, key = None):
    global workload_type
    pred_index = df["predicted_cycles"].idxmin()
    pred_results = df["results_cycles"][pred_index]

    if workload_type == "2D":
        baseline = df[(df["T_0_0"] == min(16, key[0])) & (df["T_1_1"] == 64)]
    elif workload_type == "3D":
        baseline = df[(df["T_0_0"] == min(16, key[1])) & (df["T_1_1"] == 4) & (df["T_2_1"] == 16)]
    if len(baseline) == 0:
        print(f"Error: No baseline found for key: {key}")
        return 0, 0,  get_running_time_ms(pred_results)

    baseline_idx = baseline["results_cycles"].idxmin()
    baseline_restlts = baseline["results_cycles"][baseline_idx]

    speedup = baseline_restlts / pred_results
    return speedup, get_running_time_ms(baseline_restlts), get_running_time_ms(pred_results)


def run_prediction(df):
    global workload_type
    if workload_type == "2D":
      range_i = 2
      X_names = ["batch", "vec_len"]
    else:
      range_i = 3
      X_names = ["batch", "nhead", "dhead", "seq_len"]
    Y_name = "results_cycles"
    X_df = df[X_names]
    y_series = df[Y_name]
    for i in range(range_i):
        for j in range(3):
            col_name = f"T_{i}_{j}"
            X_df[col_name] = df["tiling"].apply(apply_get_num, args=(i, j))
            X_names.append(col_name)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=TRAIN_SPLIT, random_state=1358, shuffle=True)

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=6198
    )

    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_df)
    df['predicted_cycles'] = np.nan
    df.loc[X_df.index, 'predicted_cycles'] = y_pred
    return df

def config_selection(df,ouput_path):
    df_groups, T_names = data_processing(df)

    result_list = []
    for k, v in df_groups.items():
        speedup_item, basline, our= test_performance(v, k)
        if workload_type == "2D":
            result_list.append((k[0], k[1], our, basline, speedup_item))
        elif workload_type == "3D":
            result_list.append((k[0], k[1], k[2], k[3], our, basline, speedup_item))

    if workload_type == "2D":
        col_names = ["batch_size", "vec_len", "our_time(ms)", "baseline(ms)", "Speedup"]
    elif workload_type == "3D":
        col_names = ["batch_size", "#head", "input_dim", "output_dim", "our_time(ms)", "baseline(ms)", "Speedup"]
    df_res = pd.DataFrame(result_list, columns=col_names)
    df_res.to_csv(ouput_path, index=False)
    return df_res


def build_predictor(log_dir, output_path, job_type:str ):
    global workload_type
    if job_type in ["2D", "3D"]:
        workload_type = job_type
    else:
       raise ValueError("Unsupported job type.")

    data_df = parse_data(log_dir)
    predict_df = run_prediction(data_df)
    config_selection(predict_df, output_path)
