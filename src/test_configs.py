import math

from attacc_trace_gen.gen_trace_AttAcc_RED import gen_single_traces as gen_single_traces_red
from attacc_trace_gen.gen_trace_AttAcc_GEMV import gen_single_traces as gen_single_traces_gemv, set_mode_gemv
from attacc_trace_gen.gen_trace_AttAcc_ATTN import gen_single_traces as gen_single_traces_attn, set_mode_attn

from hbmpim_trace_gen.gen_trace_HBMPIM_GEMV import gen_single_traces as gen_single_traces_gemv_hbmpim
from hbmpim_trace_gen.gen_trace_HBMPIM_RED import gen_single_traces as gen_single_traces_red_hbmpim
from hbmpim_trace_gen.gen_trace_HBMPIM_VA import gen_single_traces as gen_single_traces_va_hbmpim
from hbmpim_trace_gen.gen_trace_HBMPIM_RELU import gen_single_traces as gen_single_traces_relu_hbmpim


def generate_test_set(candidates):
  configs = []
  if len(candidates) == 4:
    for ba in candidates[0]:
      for nh in candidates[1]:
        for dh in candidates[2]:
          for seqL in candidates[3]:
              configs.append([ba, nh, dh, seqL])
  elif len(candidates) == 2:
    for ba in candidates[0]:
      for seqL in candidates[1]:
        configs.append([ba, seqL])
  else:
    raise ValueError("Invalid number of candidates")
  return configs

attn_single = generate_test_set([[1, 4], [64, 32], [128], [512, 4096]])

attn_inference = generate_test_set([[1], [8, 11, 32, 42], [128], [i for i in range(128, 4096+1, 16)]])

gemv_single = generate_test_set([[1, 4], [64, 32], [128], [512, 4096]])

gemv_inference = generate_test_set([[1], [8, 32], [128], [5120]])
gemv_inference += generate_test_set([[1], [11, 42], [128], [6656]])
gemv_inference += generate_test_set([[1], [8, 32], [5120], [384]])
gemv_inference += generate_test_set([[1], [11, 42], [6656], [384]])

red_single = generate_test_set([[1, 2, 4], [1024, 2048, 4096]])

va_single = generate_test_set([[1, 2, 4], [1024, 2048, 4096]])

relu_single = generate_test_set([[1, 2, 4], [1024, 2048, 4096]])

all_test_sets = {
  "attn_single": [attn_single, gen_single_traces_attn, None],
  "attn_inference": [attn_inference, gen_single_traces_attn, None],
  "gemv_single": [gemv_single, gen_single_traces_gemv, gen_single_traces_gemv_hbmpim],
  "gemv_inference": [gemv_inference, gen_single_traces_gemv, gen_single_traces_gemv_hbmpim],
  "red_single": [red_single, gen_single_traces_red, gen_single_traces_red_hbmpim],
  "va_single": [va_single, None, gen_single_traces_va_hbmpim],
  "relu_single": [relu_single, None, gen_single_traces_relu_hbmpim]
}

def set_mode(workload):
  if workload == "attn_single":
    set_mode_attn("kernel")
  elif workload == "attn_inference":
    set_mode_attn("inference")
  elif workload == "gemv_single":
    set_mode_gemv("kernel")
  elif workload == "gemv_inference":
    set_mode_gemv("inference")


def custom_kernel_sizes(name, batch, input, output=None, nhead=32):
  if name == "ATTN":
    all_test_sets["attn_single"][1] = generate_test_set([[batch], [nhead], [input], [output]])
  elif name == "GEMV":
    all_test_sets["gemv_single"][1] = generate_test_set([[batch], [nhead], [input], [output]])
  elif name == "RED":
    all_test_sets["red_single"][1] = generate_test_set([[batch], [input]])
  elif name == "VA":
    all_test_sets["va_single"][1] = generate_test_set([[batch], [input]])
  elif name == "RELU":
    all_test_sets["relu_single"][1] = generate_test_set([[batch], [input]])
  else:
    raise ValueError(f"Unknown kernel name: {name}")


def custom_inference_sizes(batch, input, output):
  gemv_sizes = generate_test_set([[1], [batch*8], [128], [5120]])
  gemv_sizes += generate_test_set([[1], [math.ceil(batch*10.4)], [128], [6656]])
  gemv_sizes += generate_test_set([[1], [batch*8], [5120], [384]])
  gemv_sizes += generate_test_set([[1], [math.ceil(batch*10.4)], [6656], [384]])
  attn_size = generate_test_set([[1], [batch*8, math.ceil(batch*10.4)], [128], [i for i in range(input, input+output+1, 16)]])
  all_test_sets["attn_inference"][1] = attn_size
  all_test_sets["gemv_inference"][1] = gemv_sizes

