## Define models and layer.
## Generate models
import os.path

from inference.devices import *
from inference.config import *
from inference.type import *
from predictor import build_predictor


class Layer:

    def __init__(self, stage, name, type, has_weight, dtype, m, n, k, numOp):
        self.stage = stage
        self.name = name
        self.type = type
        self.has_weight = has_weight
        self.m = m
        self.n = n
        self.k = k
        self.numOp = numOp
        self.dtype = dtype
        self.dbyte = 2
        if dtype in [DataType.W16A16]:
            self.dbyte = 2
        elif dtype in [DataType.W8A8]:
            self.dbyte = 1
        else:
            assert 0, "Only support W16A16, W8A8"
        self.bound = 'compute'  # 'memory'
        self.exec_time = 0
        self.energy = 0

        assert isinstance(type, LayerType), "Not support layer type"
        assert isinstance(dtype, DataType), "Not support data type"

    def get_infos(self):
        return self.m, self.n, self.k, self.numOp, self.dbyte

    def get_flops(self):
        if self.type == LayerType.SOFTMAX:
            return 5 * self.m * self.n * self.numOp

        elif self.type == LayerType.ACT:
            if 'relu' in self.name:
                return 1 * self.m * self.n * self.numOp
            elif 'glu' in self.name:
                return (8 + 1) * self.m * self.n * self.numOp
            else:
                return 8 * self.m * self.n * self.numOp

        elif self.type == LayerType.NORM:
            return 5 * self.m * self.n * self.numOp
        elif self.type in [LayerType.FC, LayerType.MATMUL]:
            return 2 * self.m * self.n * self.k * self.numOp
        elif self.type in [LayerType.G2G, LayerType.X2G]:
            return 0
        elif self.type in [LayerType.ADD]:
            return 1 * self.m * self.n * self.numOp
        else:
            assert 0, "In Function \"get_flops\": Not support layer type"

    def get_size(self):
        in1 = self.numOp * self.m * self.k * self.dbyte
        in2 = self.numOp * self.n * self.k * self.dbyte
        out = self.numOp * self.m * self.n * self.dbyte

        if self.type in [
                LayerType.SOFTMAX,  LayerType.G2G, LayerType.X2G
        ]:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = 0
            out = in1

            # For SwiGLU and GeGLU
            if 'glu' in self.name:
                in2 = in1

        elif self.type == LayerType.NORM:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = in1
            out = in1
        elif self.type  == LayerType.MATMUL and ("reduce" in self.name):
            in2 = 0
            out = self.numOp *  self.dbyte
        elif self.type in [LayerType.ADD, LayerType.ACT,]:
            in2 = in1
            out = in1

        return in1, in2, out




def main(config_file, num_gpu, data_size = 2):
    if data_size == 2:
        dtype = DataType.W16A16
    else:
        dtype = DataType.W8A8
    mem_cap = 80 * 1024 * 1024 * 1024  # 80GB
    # gpu_device = GPUType.H100
    gpu_device = GPUType.A100a
    df = pd.read_csv(config_file)

    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=mem_cap, mem_bw=2627.847*1e9)
    devices = xPU(DeviceType.GPU, xpu_config['GPU'], SCALING_FACTOR)
    df['GPU_time(ms)'] = 0
    for index, row in df.iterrows():
        batch = row["batch_size"]
        seq_len = row["output_dim"]
        dhead = row["input_dim"]
        num_heads = row["#head"]
        gemv = Layer('GEMV', 'gemv', LayerType.MATMUL, False, dtype,
                     1, seq_len, dhead, int(num_heads / num_gpu) * batch)
        exec_time, energy = devices.get_time_and_energy(gemv)
        df.loc[index, 'GPU_time(ms)'] = exec_time * 1000
        # print(f"GEMV{(batch, num_heads, dhead, seq_len)}:", gemv.bound, exec_time * 1000)
    save_name = config_file[:-4] + "_with_GPU.csv"
    df.to_csv(save_name, index=False)


def main_kernels(config_file, num_gpu, data_size = 2, op = "GEMV", output_path = "../results/"):
    if data_size == 2:
        dtype = DataType.W16A16
    else:
        dtype = DataType.W8A8
    mem_cap = 80 * 1024 * 1024 * 1024  # 80GB
    # gpu_device = GPUType.H100
    gpu_device = GPUType.A100a
    df = pd.read_csv(config_file)

    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=mem_cap)
    devices = xPU(DeviceType.GPU, xpu_config['GPU'], SCALING_FACTOR)
    df['GPU_time(ms)'] = 0
    df['Speedup_GPU'] = 0
    for index, row in df.iterrows():
        dhead = 1
        batch = 1
        if op in ["GEMV", "ATTENTION", "FC"]:
            dhead = row["input_dim"]
            num_heads = row["#head"]
            batch = row["batch_size"]
            seq_len = row["output_dim"]
        else:
            num_heads = row["batch_size"]
            seq_len = row["vec_len"]
        batch *= 5 # for 5 hbm cubes
        if "papi" in config_file and "[1,10]x[1]" in config_file:
            batch *= 2
        if op == "GEMV":
            layers = [Layer('GEMV', 'gemv', LayerType.MATMUL, False, dtype,
                         1, seq_len, dhead, int(num_heads / num_gpu) * batch)]
        elif op == "FC":
            layers = [Layer('FC', 'fc', LayerType.FC, False, dtype,
                            1, seq_len, dhead * 16, int(num_heads / 16 / num_gpu) * batch)]
        elif op == "RED":
            layers = [Layer('RED', 'reduce', LayerType.MATMUL, False, dtype,
                         1, 1, seq_len, int(num_heads / num_gpu) * batch)]
        elif op == "VA":
            layers = [Layer('VA', 'addition', LayerType.ADD, False, dtype,
                          1, 1, seq_len, int(num_heads / num_gpu) * batch)]
        elif op == "RELU":
            layers = [Layer('RELU', 'relu', LayerType.ACT, False, dtype,
                          1, 1, seq_len, int(num_heads / num_gpu) * batch)]
        elif op == "ATTENTION":
            layers = []
            layers.append(
                Layer('ATTENTION', 'score', LayerType.MATMUL, False, dtype,
                      1, seq_len, dhead, int(num_heads / num_gpu) * batch))
            layers.append(
                Layer('ATTENTION', 'softmax_att', LayerType.SOFTMAX, False, dtype,
                      1, seq_len, 1, int(num_heads / num_gpu) * batch))
            layers.append(
                Layer('ATTENTION', 'context', LayerType.MATMUL, False, dtype,
                      1, dhead, seq_len, int(num_heads / num_gpu) * batch))
        else:
            raise Exception("Not support op type")

        exec_time = 0
        for l in layers:
            l_time, l_energy = devices.get_time_and_energy(l)
            exec_time += l_time
            # print(f"Layer_{l.name}{(batch, num_heads, dhead, seq_len)}:", l.bound, exec_time * 1000)
        df.loc[index, 'GPU_time(ms)'] = exec_time * 1000
        df.loc[index, 'Speedup_GPU'] = exec_time * 1000 / df.loc[index, 'our_time(ms)']


    save_path = os.path.join(output_path, os.path.basename(config_file)[:-4] + "_with_GPU.csv")
    df.to_csv(save_path, index=False)



if __name__ == '__main__':
    if not os.path.isdir("./temp/simulation"):
        os.mkdir("./temp/simulation")

    build_predictor("./temp/logs/attacc-attn_single/", "./temp/simulation/@attacc-attn_single.csv", "3D")
    build_predictor("./temp/logs/attacc-gemv_single/", "./temp/simulation/@attacc-gemv_single.csv", "3D")
    build_predictor("./temp/logs/hbmpim-gemv_single/", "./temp/simulation/@hbmpim-gemv_single.csv", "3D")
    build_predictor("./temp/logs/attacc-red_single/", "./temp/simulation/@attacc-red_single.csv", "2D")
    build_predictor("./temp/logs/hbmpim-red_single/", "./temp/simulation/@hbmpim-red_single.csv", "2D")
    build_predictor("./temp/logs/hbmpim-relu_single/", "./temp/simulation/@hbmpim-relu_single.csv", "2D")
    build_predictor("./temp/logs/hbmpim-va_single/", "./temp/simulation/@hbmpim-va_single.csv", "2D")

    main_kernels(os.path.join("./temp/simulation/", "@attacc-attn_single.csv"), 1, 2, "ATTENTION")
    main_kernels(os.path.join("./temp/simulation/", "@attacc-gemv_single.csv"), 1, 2,"GEMV")
    main_kernels(os.path.join("./temp/simulation/", "@attacc-red_single.csv"), 1, 2, "RED")
    main_kernels(os.path.join("./temp/simulation/", "@hbmpim-gemv_single.csv"), 1,2, "GEMV")
    main_kernels(os.path.join("./temp/simulation/", "@hbmpim-red_single.csv"), 1, 2, "RED")
    main_kernels(os.path.join("./temp/simulation/", "@hbmpim-va_single.csv"), 1, 2, "VA")
    main_kernels(os.path.join("./temp/simulation/", "@hbmpim-relu_single.csv"), 1, 2, "RELU")

