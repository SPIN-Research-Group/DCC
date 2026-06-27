import os
import json
import subprocess
import argparse
import random
import multiprocessing
import pandas as pd
from tqdm import tqdm
from generate_configs import generate_configs_2D, generate_configs_3D
from test_configs import all_test_sets, set_mode

simulator_path = ""
config_dir = "./temp/configs/"
trace_dir = "./temp/traces/"
log_dir = "./temp/logs/"


def check_dirs():
  print(log_dir)
  for dir_path in [config_dir, trace_dir, log_dir]:
    if not os.path.exists(dir_path):
      os.makedirs(dir_path, exist_ok=True)


def gen_single_config(config):
  if len(config) == 4:
    return generate_configs_3D(config[0], config[1], config[2], config[3], config_dir)
  elif len(config) == 2:
    return generate_configs_2D(config[0], config[1], config_dir)
  else:
    raise ValueError("Invalid config length")


def generate_traces(configs):
  config_f_list = []
  pool = multiprocessing.Pool()
  results = pool.map(gen_single_config, configs)
  pool.close()
  pool.join()
  for idx in range(len(configs)):
    config_f_list.append(results[idx])

  return config_f_list


def run_trace_gen_and_sim(config):
  gen_trace_func = config["gen_trace_func"]
  try:
    trace_file = gen_trace_func(config, trace_dir)
    f_name = str(os.path.basename(trace_file)).replace(".trace", "")
    yaml_path = os.path.join(trace_dir, f_name + ".yaml")
    log_path = os.path.join(log_dir, f_name + ".log")
    err_path = os.path.join(log_dir, f_name + ".err")

    if not (os.path.exists(log_path) and os.path.exists(err_path) and os.path.getsize(err_path) == 0):
      cmd = f"{simulator_path} -f \"{yaml_path}\""
      cmd_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
      with open(log_path, "w") as f:
        f.write(cmd_result.stdout)
      with open(err_path, "w") as f:
        f.write(cmd_result.stderr)

    if os.path.exists(trace_file):
      os.remove(trace_file)
  except Exception as e:
    print(f"Run Failed {config}: {e}")
    f_name = "ERROR.log"

  return f_name


def get_args():
  parser = argparse.ArgumentParser(
    description="Model configuration",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  ## set system configuration
  parser.add_argument(
    "--backend",
    type=str,
    default="attacc",
    choices=["attacc", "hbmpim"])

  parser.add_argument(
    "--workload",
    type=str,
    default='attn_inference',
    choices=["attn_single", "attn_inference", "gemv_single", "gemv_inference",
             "red_single", "va_single", "relu_single"])
  args = parser.parse_args()
  return args


def main(args):
  global log_dir, simulator_path
  set_mode(args.workload)
  workload_set = all_test_sets[args.workload]
  inputs = workload_set[0]
  if args.backend == "attacc":
    simulator_path = "LD_LIBRARY_PATH=./Ramulator2/AttAcc/:$LD_LIBRARY_PATH ./Ramulator2/ramulator2-AttAcc"
    gen_trace_func = workload_set[1]
  else:
    simulator_path = "LD_LIBRARY_PATH=./Ramulator2/HBMPIM/:$LD_LIBRARY_PATH ./Ramulator2/ramulator2-HBMPIM"
    gen_trace_func = workload_set[2]

  log_dir = f"./temp/logs/{args.backend}-{args.workload}/"
  check_dirs()

  configs = generate_traces(inputs)
  experiments_list = []
  for cfg_f in configs:
    config_info = json.load(open(cfg_f, "r"))
    for i in range(len(config_info["tiling"])):
      new_cfg = config_info.copy()
      new_cfg["tiling"] = [config_info["tiling"][i]]
      new_cfg["gen_trace_func"] = gen_trace_func
      experiments_list.append(new_cfg)

  random.shuffle(experiments_list)
  run_list = list(experiments_list)
  # print("Running Experiment numbers:", len(run_list))

  cycle_results = []
  with multiprocessing.Pool(maxtasksperchild=128) as pool:
    for res in tqdm(
        pool.imap(run_trace_gen_and_sim, run_list),
        desc="profile_experiments",
        total=len(run_list),
        ncols=100,
        unit="task"
    ):
      cycle_results.append(res)


if __name__ == "__main__":
  args = get_args()
  main(args)