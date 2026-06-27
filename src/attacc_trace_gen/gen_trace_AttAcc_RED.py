import math
import os
from generate_configs import *

model = "gpt-3-175B"

data_size = 2  # FP 16, 2 bytes

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # byte
n_mac = 16

# Granularity size
n_grf = 8
HBM_GS = {}
HBM_GS['col'] = prefetch_size
HBM_GS['row'] = n_col * HBM_GS['col']
HBM_GS['ba'] = n_row * HBM_GS['row']
HBM_GS['bg'] = n_bank * HBM_GS['ba']
HBM_GS['rank'] = n_bg * HBM_GS['bg']
HBM_GS['pch'] = n_rank * HBM_GS['rank']
HBM_GS['ch'] = n_pch * HBM_GS['pch']
HBM_GS['hbm'] = n_channel * HBM_GS['ch']
HBM_GS['attacc'] = max_n_hbm * HBM_GS['hbm']
PIM_GROUP_SIZE = (n_pch * n_rank * n_bg * n_bank)



cmd_score_wrgb = []
cmd_score_mac = []
cmd_score_mvsb = []
cmd_sfm = []
cmd_context_mvgb = []
cmd_context_mac = []
cmd_context_mvsb = []
cmd_context_ld = []
cmd_context_ret = []

valid_channels = []


def cmd_list_reset():
  global cmd_score_wrgb
  global cmd_score_mac
  global cmd_score_mvsb
  global cmd_sfm
  global cmd_context_mvgb
  global cmd_context_mac
  global cmd_context_mvsb
  global cmd_context_ld
  global cmd_context_ret
  global valid_channels

  cmd_score_wrgb = []
  cmd_score_mac = []
  cmd_score_mvsb = []
  cmd_sfm = []
  cmd_context_mvgb = []
  cmd_context_mac = []
  cmd_context_mvsb = []
  cmd_context_ld = []
  cmd_context_ret = []
  valid_channels = []


def gen_yaml(trace_path):
  raw = """Frontend:
  impl: PIMLoadStoreTrace
  path: PATH_TO_TRACE
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: 2147483648


MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps
      #preset: HBM3_5.2Gbps_NPC

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
      #impl: No
    plugins:
    - ControllerPlugin:
        impl: HBM3TraceRecorder
        path: ./temp/log/attacc_bank/cmd.log

  AddrMapper:
    impl: HBM3-PIM
  """
  new = raw.replace("PATH_TO_TRACE", os.path.abspath(trace_path))
  yaml_path = trace_path.replace(".trace", ".yaml")
  with open(yaml_path, 'w') as f:
    f.write(new)




def RED(T, Vec_addr, Ret_addr, itr):
  T = match_alignment(T)
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_context_ld.append([])
  cmd_context_ret.append([])


  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  for pos in range(math.ceil(T[0][2] * T[1][2] / n_mac)):
    for ba_idx in range(T[0][1] * T[1][1]):
      for lch_idx in range(T[0][0] * T[1][0]):
        addr = Vec_addr + lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + pos * prefetch_size
        hex_addr = hex(addr)[2:]
        cmd_context_ld[itr].append("ST 0x{0:0>8}".format(hex_addr))


  for pos in range(math.ceil(T[0][2] * T[1][2] / n_mac)):
    for ba_idx in range(T[0][1] * T[1][1]):
      for lch_idx in range(T[0][0] * T[1][0]):
        addr = Ret_addr + lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + pos * prefetch_size
        hex_addr = hex(addr)[2:]
        cmd_context_ret[itr].append("ST 0x{0:0>8}".format(hex_addr))

  for ba_idx in range(T[0][1] * T[1][1]):
    for lch_idx in range(T[0][0] * T[1][0] ):
      addr = lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba']
      hex_addr = hex(addr)[2:]
      cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

  for pos1 in range(math.ceil(T[1][2] / n_mac)):
    for lch_idx in range(T[0][0] * T[1][0]):
      addr = Vec_addr + lch_idx * HBM_GS['ch'] + pos1 * HBM_GS['col']
      hex_addr = hex(addr)[2:]
      cmd_score_mac[itr].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

  for bg_idx in range(math.ceil(T[0][1] * T[1][1]  / n_bg)):
    for lch_idx in range(T[0][0] * T[1][0]):
      addr = Ret_addr + lch_idx * HBM_GS['ch'] + bg_idx * HBM_GS['bg']
      hex_addr = hex(addr)[2:]
      cmd_score_mvsb[itr].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))


def total_size(tiling, num_itr):
  itr_size = math.ceil(tiling[0][2] / num_itr)
  Vec1_size = math.ceil(itr_size * tiling[1][2] * data_size / prefetch_size) * prefetch_size
  Ret_size = math.ceil(itr_size * tiling[1][2] * data_size / prefetch_size) * prefetch_size
  return itr_size, Vec1_size, Ret_size

def run_RED(tiling, output_path):
  num_itr = 1
  itr_size, Vec_size, Ret_size = total_size(tiling, num_itr)

  while num_itr <= tiling[0][2]:
    itr_size, Vec_size, Ret_size = total_size(tiling, num_itr)
    if Vec_size + Ret_size <= HBM_GS["ba"]:
      break
    else:
      num_itr += 1

  real_tiling = tiling.copy()
  real_tiling[0][2] = itr_size

  cmd_list_reset()
  ##-- Generate Commands --##
  Vec_addr = prefetch_size
  Ret_addr = Vec_addr + Vec_size
  for itr in range(num_itr):
    RED(real_tiling, Vec_addr, Ret_addr, itr)


  ##-- Ovelapping Commands --##
  total_cmd = []
  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  for itr in range(num_itr):
    total_cmd += cmd_context_ld[itr]
    total_cmd += barrier
    total_cmd += cmd_score_wrgb[itr]
    for j in range(real_tiling[0][2]):
      total_cmd += cmd_score_mac[itr]
      # if (j % n_mac == n_mac - 1 or j + 1 == real_tiling[0][2]):
      total_cmd += cmd_score_mvsb[itr]
    total_cmd += barrier
    total_cmd += cmd_context_ret[itr]


  all_cmd_str = "\n".join(total_cmd)
  with open(output_path, 'w') as trace_file:
    trace_file.write(all_cmd_str)
  trace_file.close()
  gen_yaml(output_path)


def f_name_gen(config_info, op_name, output_dir):
  tiling = config_info['tiling'][0]
  if op_name in ["VA", "RED"]:
    batch = config_info['batch']
    vec_len = config_info['vec_len']
    output_path = f"{output_dir}/{op_name}({batch}, {vec_len})_[{tiling[0]}-{tiling[1]}]_bank.trace"
    return output_path
  else:
    batch = config_info['batch']
    dhead = config_info['dhead']
    nhead = config_info['nhead']
    L = config_info['seq_len']
    output_path = f"{output_dir}/GEMV({batch},{nhead},{dhead},{L})_[{tiling[0]}-{tiling[1]}-{tiling[2]}]_bank.trace"
    return output_path

def gen_single_name(config_info, output_dir="./traces"):
  batch = config_info['batch']
  vec_len = config_info['vec_len']
  tiling = config_info['tiling'][0]
  output_path = f"{output_dir}/red({batch}, {vec_len})_[{tiling[0]}-{tiling[1]}]_bank.trace"
  return output_path

def gen_single_traces(config_info, output_dir="./traces"):
  assert len(config_info['tiling']) == 1
  output_path = gen_single_name(config_info, output_dir)
  tiling = config_info['tiling'][0]
  run_RED(tiling, output_path)
  return output_path