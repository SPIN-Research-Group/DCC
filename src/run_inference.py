import argparse
import copy
import csv
import os
from inference.system import *
from inference.config import *
from predictor import build_predictor

def write_csv(logfile, perfs, names = None):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm',
            ]
            if names is not None:
              col_name.insert(0, "name")
            wrt.writerow(col_name)

        for i, perf in enumerate(perfs):
            tag, config, time = perf
            if names is not None:
              info = [names[i]] + tag + config + time
            else:
              info = tag + config + time
            wrt.writerow(info)
        f.close()


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        name="AttAcc"):
    print("---Run {:<16} Batch {:<2} Lin {:<4} Lout {:<4} ---".
          format(name, batch, lin, lout))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    return perfs

def get_args():
  parser = argparse.ArgumentParser(
    description="Model configuration",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  ## set system configuration
  parser.add_argument("--all",
                      action='store_true',
                      help="Run all experiments for plots")
  parser.add_argument("--system",
                      type=str,
                      default="dgx-attacc",
                      help="dgx (each GPU has 80GB HBM),dgx-attacc (dgx + attacc)")
  parser.add_argument("--gpu",
                      type=str,
                      default='A100a',
                      help="GPU type (A100a and H100), A100a is A100 with HBM3")
  parser.add_argument("--ngpu",
                      type=int,
                      default=1,
                      help="number of GPUs in DGX system. default=8")
  parser.add_argument("--gmemcap",
                      type=int,
                      default=80,
                      help="memory capacity per GPU (GB). default=80")
  ## set attacc configuration
  parser.add_argument("--pim",
                      type=str,
                      default='bank',
                      help="pim mode. list: bank, bg, buffer")
  parser.add_argument("--powerlimit",
                      action='store_true',
                      help="power constraint for PIM ")
  parser.add_argument("--ffopt",
                      action='store_true',
                      help="apply feedforward parallel optimization")
  parser.add_argument("--pipeopt",
                      action='store_true',
                      help="apply pipeline optimization ")
  parser.add_argument("--model",
                      type=str,
                      default='GPT-13B',
                      help="model list: GPT-13B, LLAMA-33B, MT-310B")
  parser.add_argument("--word",
                      type=int,
                      default='2',
                      help="word size (precision): 1(INT8), 2(FP16)")
  parser.add_argument("--lin",
                      type=int,
                      default=128,
                      help="input sequence length")
  parser.add_argument("--lout",
                      type=int,
                      default=2048,
                      help="number of generated tokens")
  parser.add_argument("--batch",
                      type=int,
                      default=1,
                      help="batch size, default = 1")
  parser.add_argument("--save_path",
                      type=str,
                      default="../results/@inference_results.csv",
                      help="output save path")

  args = parser.parse_args()
  return args


def run_single_inference(args):
    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        raise ValueError("GPU type not supported")

    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024


    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    modelinfos["DCC"] = (args.res_label == "our_time(ms)")
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        pim_config = make_pim_config(pim_type,
                                     InterfaceType.NVLINK3,
                                     num_attacc=args.ngpu,
                                     power_constraint=args.powerlimit)
        pim_config["RES_ATTACC_LABEL"] = args.res_label
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    res_perfs = run(system,
                    args.batch,
                    args.lin,
                    args.lout,
                    pipe=args.pipeopt,
                    parallel=args.ffopt,
                    power_constraint=args.powerlimit,
                    name=args.name)

    return res_perfs

def run_group_inference(args):
  args.pipeopt = False
  args.ffopt = False

  args.res_label = "our_time(ms)"
  args.name="Attacc_base+DCC"
  perf_base_dcc = run_single_inference(args).copy()

  args.res_label = "baseline(ms)"
  args.name = "Attacc_base"
  perf_base_baseline = run_single_inference(args).copy()

  args.pipeopt = True
  args.ffopt = True

  args.res_label = "our_time(ms)"
  args.name = "Attacc_full+DCC"
  perf_full_dcc = run_single_inference(args).copy()

  args.res_label = "baseline(ms)"
  args.name = "Attacc_full"
  perf_full_baseline = run_single_inference(args).copy()

  args.system = "dgx"
  args.name = "GPU"
  perf_GPU = run_single_inference(args).copy()
  perfs = perf_base_dcc + perf_base_baseline + perf_full_dcc + perf_full_baseline + perf_GPU
  names = ["Attacc_base+DCC", "Attacc_base", "Attacc_full+DCC", "Attacc_full", "GPU"]

  write_csv(args.save_path, perfs, names)

if __name__ == "__main__":
  args = get_args()

  if os.path.exists(args.save_path):
    os.remove(args.save_path)

  if not os.path.isdir("./temp/simulation"):
    os.mkdir("./temp/simulation")

  build_predictor("./temp/logs/attacc-gemv_inference/", "./temp/simulation/@attacc-gemv_inference.csv", "3D")
  build_predictor("./temp/logs/attacc-attn_inference/", "./temp/simulation/@attacc-attn_inference.csv", "3D")

  if not args.all:
    run_group_inference(args)
  else:
    sizes = [(128, 2048), (2048,128), (2048,2048)]
    batchs = [1, 4]
    ngpus = [1]
    models = ["GPT-13B", "LLAMA-33B"]

    for md in models:
      for ba in batchs:
        for lin,lout in sizes:
          local_args = copy.deepcopy(args)
          local_args.model = md
          local_args.batch = ba
          local_args.ngpu = 1
          local_args.lin = lin
          local_args.lout = lout
          print("\n=== run inference for model={}, batch={}, lin={}, lout={} ===".format(md, ba, lin, lout))
          run_group_inference(local_args)
    



