import math
import json

Groups = 16
Cores = 64
total_cores = Groups * Cores



def generate_search_space_2D(dims, current_state, result):
  if current_state[0] == 0:
    rest = f"{current_sol[0][2]}_{current_sol[1][2]}"
    if current_sol[0][0] == 16 and current_sol[1][1] == 64:
      all_sol.append(current_sol.copy())
      if not(rest in result):
        result[rest] = [current_sol.copy()]
    elif (current_state[1] == 1 and current_state[2] <= 4):
      if rest in result:
        result[rest].append(current_sol.copy())
      else:
        all_sol.append(current_sol.copy())
        result[rest] = [current_sol.copy()]
    return 1

  cand_a = []
  cand_b = []
  for i in range(1, current_state[1]+1):
    cand_a.append(i)
    if cand_a[-1] > dims[0]:
      break

  for i in range(1, current_state[2]+1):
    cand_b.append(i)
    if cand_b[-1] > dims[0]:
      break

  # set serach order: first search the configs that match the alignment, then search the rest configs.
  new_dims = dims[1:]
  for a in cand_a:
    for b in cand_b:
      if (a & (a - 1) == 0) and (b & (b - 1) == 0) and a * b <= dims[0]:
        new_state = [current_state[0] - 1, current_state[1] // a, current_state[2] // b]
        current_sol.append((a, b, math.ceil(dims[0] / (a * b))))
        generate_search_space_2D(new_dims, new_state, result)
        current_sol.pop()

  for a in cand_a:
    for b in cand_b:
      if not ((a & (a - 1) == 0) and (b & (b - 1) == 0)) and a * b <= dims[0]:
        new_state = [current_state[0] - 1, current_state[1] // a, current_state[2] // b]
        current_sol.append((a, b, math.ceil(dims[0] / (a * b))))
        generate_search_space_2D(new_dims, new_state, result)
        current_sol.pop()

  return


def generate_search_space_3D(dims, current_state, result):
  if current_state[0] == 0:

    rest = f"{current_sol[0][2]}_{current_sol[1][2]}_{current_sol[2][2]}"
    if current_sol[0][0] == 16 and current_sol[1][1] == 4 and current_sol[2][1] == 16:
      all_sol.append(current_sol.copy())
      if not(rest in result):
        result[rest] = [current_sol.copy()]
    elif (current_state[1] == 1 and current_state[2] == 1 and current_sol[1][2] >= 16):
      if rest in result:
        result[rest].append(current_sol.copy())
      else:
        all_sol.append(current_sol.copy())
        result[rest] = [current_sol.copy()]
    return


  cand_a = []
  cand_b = []

  for i in range(1, current_state[1]+1):
    cand_a.append(i)
    if cand_a[-1] > dims[0]:
      break

  for i in range(1, current_state[2]+1):
    cand_b.append(i)
    if cand_b[-1] > dims[0]:
      break

  new_dims = dims[1:]
  for a in cand_a:
    for b in cand_b:
      if a * b <= dims[0]:
        new_state = [current_state[0] - 1, current_state[1] // a, current_state[2] // b]
        current_sol.append((a, b, math.ceil(dims[0] / (a * b))))
        generate_search_space_3D(new_dims, new_state, result)
        current_sol.pop()
  return

def generate_configs_2D(batch, vec_len, output_dir = "./config/"):
  global all_sol, current_sol
  result = dict()
  dims = [batch, vec_len]
  current_sol = []
  all_sol = []
  generate_search_space_2D(dims, (len(dims), Groups, Cores), result)
  if batch < Groups:
    baseline_group1 = batch
    baseline_group2 = Groups//baseline_group1
    baseline_sol = [(baseline_group1, 1, math.ceil(batch / baseline_group1)), (baseline_group2, 64, math.ceil(vec_len / 64 / baseline_group2))]
    if baseline_sol not in all_sol:
      all_sol.append(baseline_sol)
  config_info = {}
  config_info["tiling"] = all_sol
  config_info['batch'] = batch
  config_info['vec_len'] = vec_len
  f_name = f"{output_dir}/config({batch}, {vec_len}).json"
  with open(f_name, "w") as f:
    json.dump(config_info, f, indent=2)
  return f_name

def generate_configs_3D(batch, nhead, dhead, seq_len, output_dir = "./config/"):
  global all_sol, current_sol
  result = dict()
  dims = [batch * nhead, dhead, seq_len]
  current_sol = []
  all_sol = []
  generate_search_space_3D(dims, (len(dims), Groups, Cores), result)
  if batch * nhead < Groups:
    baseline_group1 = min(batch * nhead, 16)
    baseline_group2 = Groups//baseline_group1
    baseline_sol = [(baseline_group1, 1, math.ceil(batch * nhead / baseline_group1)), (1, 4, math.ceil(dhead / 4)), (baseline_group2, 16, math.ceil(seq_len / 16 / baseline_group2))]
    if baseline_sol not in all_sol:
      all_sol.append(baseline_sol)
    baseline_sol = [(baseline_group1, 1, math.ceil(batch * nhead / baseline_group1)), (baseline_group2, 4, math.ceil(dhead / 4 / baseline_group2)), (1, 16, math.ceil(seq_len / 16))]
    if baseline_sol not in all_sol:
      all_sol.append(baseline_sol)
  config_info = {}
  config_info["tiling"] = all_sol
  config_info['batch'] = batch
  config_info['dhead'] = dhead
  config_info['nhead'] = nhead
  config_info['seq_len'] = seq_len
  f_name = f"{output_dir}/config({batch}, {nhead}, {dhead}, {seq_len}).json"
  with open(f_name, "w") as f:
    json.dump(config_info, f, indent=2)
  return f_name

def match_alignment(T):
  # not modify the baseline
  if len(T) == 3:
    if T[0][1] == 1 and T[1][0] == 1 and T[1][1] == 4 and T[2][0] == 1 and T[2][1] == 16:
      return T
  elif len(T) == 2:
    if T[0][1] == 1 and T[1][0] == 1 and T[1][1] == 64:
      return T

  for i in range(0, len(T)):
    for j in range(0, 2):
      v = T[i][j]
      T[i][j] = 1 << (v - 1).bit_length() if v > 1 else 1
  return T