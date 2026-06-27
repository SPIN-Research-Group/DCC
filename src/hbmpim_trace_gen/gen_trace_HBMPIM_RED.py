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


def gen_half_bank_red_cmds(T, Vec_addr, Ret_addr, banks_bases, pos_i_st, grf_full):
  cmds = []
  for pos_i_grf_st in range(0, math.ceil(T[1][2] /  n_mac), n_grf):
    pos_i_grf_end = int(min(pos_i_grf_st + n_grf, math.ceil(T[1][2] /  n_mac)))
    for pos_i in range(pos_i_grf_st, pos_i_grf_end):
      for bank_base in banks_bases:
        addr = Vec_addr + bank_base + (pos_i_st +pos_i) * data_size
        hex_addr = hex(addr)[2:]
        cmds.append("PIM_MAC_OP1 0x{0:0>8}".format(hex_addr))
  for pos_i in range(4):
    for bank_base in banks_bases:
      addr = Vec_addr + bank_base + (pos_i) * data_size
      hex_addr = hex(addr)[2:]
      cmds.append("PIM_MAC_OP1 0x{0:0>8}".format(hex_addr))

  if grf_full:
    for bank_base in banks_bases:
      addr = Ret_addr + bank_base + pos_i_st * data_size
      hex_addr = hex(addr)[2:]
      cmds.append("PIM_WB_RES 0x{0:0>8}".format(hex_addr))
  return cmds


def RED(T, Vec1_addr, Ret_addr, barrier):
  T = match_alignment(T)
  cmd_ld = []
  for pos in range(math.ceil(T[0][2] * T[1][2] / n_mac)):
    for ba_idx in range(T[0][1] * T[1][1]):
      for lch_idx in range(T[0][0] * T[1][0]):
        addr = Vec1_addr + lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + pos * prefetch_size
        hex_addr = hex(addr)[2:]
        cmd_ld.append("ST 0x{0:0>8}".format(hex_addr))
  cmd_ld += barrier


  cmd_red = []
  # even_banks = [i for i in range(0, T[0][1] * T[1][1] * T[2][1], 2)]
  # old_banks = [i for i in range(1, T[0][1] * T[1][1] * T[2][1], 2)]
  even_bank_base = []
  old_banks_base = []
  for ba_idx in range(T[0][1] * T[1][1]):
    for lch_idx in range(T[0][0] * T[1][0]):
      if ba_idx % n_bank == 0:
        even_bank_base.append(lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'])
      elif ba_idx % n_bank == 1:
        old_banks_base.append(lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'])

  for itr in range(T[0][2]):
    pos_i_st = itr * T[1][2]
    grf_full = (itr % n_grf == n_grf - 1) or (itr == T[0][2] - 1)
    cmd_red += gen_half_bank_red_cmds(T, Vec1_addr, Ret_addr, even_bank_base, pos_i_st, grf_full)
    cmd_red += gen_half_bank_red_cmds(T, Vec1_addr, Ret_addr, old_banks_base, pos_i_st, grf_full)

  cmd_red += barrier

  cmd_ret = []
  for pos in range(math.ceil(T[0][2] * T[1][2] / n_mac)):
    for ba_idx in range(T[0][1] * T[1][1]):
      for lch_idx in range(T[0][0] * T[1][0]):
        addr = Ret_addr + lch_idx * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + pos * prefetch_size
        hex_addr = hex(addr)[2:]
        cmd_ret.append("ST 0x{0:0>8}".format(hex_addr))

  All_cmd = cmd_ld + cmd_red  + cmd_ret
  # print(len(cmd_ld), len(cmd_gemv), len(cmd_ret))
  return All_cmd


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

  Vec_addr = 0
  Ret_addr = Vec_addr + Vec_size


  total_cmd = []
  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  for itr in range(num_itr):
    cmds = RED(real_tiling, Vec_addr, Ret_addr, barrier)
    total_cmd += cmds


  all_cmd_str = "\n".join(total_cmd)
  with open(output_path, 'w') as trace_file:
    trace_file.write(all_cmd_str)
  trace_file.close()
  gen_yaml(output_path)



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