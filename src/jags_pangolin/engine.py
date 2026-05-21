import subprocess
import platform
import re
import tempfile
from pathlib import Path
from pangolin.ir import *
from .Backend.scalar_ops import Scalar_ops
from .Backend.Multi_funcs import Multi_funcs
from .Backend.flow import flow
from .Backend.index import index

SYSTEM = platform.system()
if SYSTEM != "Windows":
    import pyjags


def ensure_size(arr, sizes, depth=0):
    while len(arr) < sizes[depth]:
        if depth == len(sizes) - 1:
            arr.append(None)
        else:
            arr.append([])
    return arr


class Sample_prob:
    class RunDFS:
        def __init__(self):
            self.visited = {}

        def dfs(self, node):
            name = "v" + str(node._n)
            if name in self.visited:
                return
            self.visited[name] = node
            for parent in node.parents:
                self.dfs(parent)

        def run_dfs(self, nodes):
            for node in nodes:
                self.dfs(nodes[node])
            return self.visited

    def _build_model_code(self, res):
        model_code = "model {\n"
        check = {}
        for node in sorted(res):
            if node in check:
                continue
            check[node] = True
            parents = [f"v{res[node].parents[i]._n}" for i in range(len(res[node].parents))]
            if flow.__dict__.get(res[node].op.name) is not None:
                tmp_p = [res[node].parents[i].shape for i in range(len(res[node].parents))]
                code = flow.__dict__[res[node].op.name](node, res[node].op, parents, 0, tmp_p)
                model_code += code + "\n"
            elif index.__dict__.get(res[node].op.name) is not None:
                tmp_p = index()
                tmpp = [res[node].parents[i] for i in range(len(res[node].parents))]
                code = tmp_p.Index(node, parents, tmpp)
                model_code += code + "\n"
            elif res[node].op.name != "Constant" and Scalar_ops.__dict__.get(res[node].op.name) is not None:
                code = Scalar_ops.__dict__[res[node].op.name](node, parents)
                model_code += code + "\n"
            elif Multi_funcs.__dict__.get(res[node].op.name) is not None:
                tmp_p = [res[node].parents[i].shape for i in range(len(res[node].parents))]
                code = Multi_funcs.__dict__[res[node].op.name](node, res[node].op, parents, tmp_p)
                model_code += code + "\n"
        model_code += "}"
        return model_code

    def _repack(self, arr, var):
        """Convert pyjags output (dim..., niter, chains) to (niter, dim...)
        and apply any per-op corrections."""
        arr = arr[..., 0]             
        arr = np.moveaxis(arr, -1, 0) 

        while arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if var.op.name == "Categorical":
            arr = arr - 1
        return arr

    def sample(self, sample_vars: list[RV], kwargs=[], values=[], niter=1000, debug=False):
        dic = {}
        for var in kwargs:
            dic["v" + str(var._n)] = var
        for sample_var in sample_vars:
            dic["v" + str(sample_var._n)] = sample_var
        app = self.RunDFS()
        res = app.run_dfs(dic)

        model_code = self._build_model_code(res)

        if debug:
            print("=== model ===")
            print(model_code)

        if SYSTEM == "Windows":
            return self._sample_windows(sample_vars, kwargs, values, niter, debug, res, model_code)
        else:
            return self._sample_pyjags(sample_vars, kwargs, values, niter, debug, res, model_code)

    def _sample_pyjags(self, sample_vars, kwargs, values, niter, debug, res, model_code):
        data = {}
        for node in res:
            if res[node].op.name == "Constant":
                data[node] = np.array(res[node].op.value, dtype=np.float64)
        for i in range(len(kwargs)):
            data[f"v{kwargs[i]._n}"] = np.array(values[i], dtype=np.float64)

        if debug:
            print("=== data ===")
            print(data)

        monitor_vars = [f"v{v._n}" for v in sample_vars]
        model = pyjags.Model(code=model_code, data=data, chains=1, adapt=1000)

 
        all_model_vars = set(model.variables)
        vars_to_monitor = [v for v in monitor_vars if v in all_model_vars]

        if debug:
            missing = [v for v in monitor_vars if v not in all_model_vars]
            if missing:
                print(f"Warning: variables not directly monitorable: {missing}")

        samples = model.sample(niter, vars=vars_to_monitor)

        final = []
        for var in sample_vars:
            arr = self._repack(samples[f"v{var._n}"], var)
            final.append(arr)
        return final

    def _sample_windows(self, sample_vars, kwargs, values, niter, debug, res, model_code):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "data.R"
            model_path = tmp / "model.bug"
            script_path = tmp / "script.txt"
            coda_chain = tmp / "CODAChain1.txt"
            coda_index = tmp / "CODAIndex.txt"

            with open(data_path, "w") as f:
                for node in res:
                    if res[node].op.name == "Constant":
                        f.write(Scalar_ops.__dict__["Constant_before"](node, res[node]))
                for i in range(len(kwargs)):
                    f.write(Scalar_ops.__dict__["Constant_before"](f"v{kwargs[i]._n}", RV(Constant(values[i]))))

            if debug:
                with open(data_path, "r") as f:
                    print("=== data ===")
                    print(f.read())

            with open(model_path, "w") as f:
                f.write(model_code + "\n")

            with open(script_path, "w") as f:
                script = f'model in "{str(model_path.absolute()).replace(chr(92), "/")}"\n'
                script += f'data in "{str(data_path.absolute()).replace(chr(92), "/")}"\n'
                script += "compile, nchains(1)\n"
                script += "initialize\n"
                script += "update 1000\n"
                for sample_var in sample_vars:
                    script += f"monitor {'v'+str(sample_var._n)}\n"
                script += f"update {niter}\n"
                script += "coda *\n"
                f.write(script)
                if debug:
                    print("=== script ===")
                    print(script)

            jags_path = "C:/Program Files/JAGS/JAGS-4.3.2/x64/bin/jags.bat"
            cmd = f'cd /d "{tmp}" && "{jags_path}" script.txt'
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode()

            return self._read_coda(sample_vars, coda_chain, coda_index)

    def _read_coda(self, sample_vars, coda_chain, coda_index):
        result = {}

        with open(coda_chain, "r") as f:
            res_lines = [float(line.strip().split()[1]) for line in f]

        pattern = re.compile(r'(v\d+)(?:\[(.*?)\])?')

        with open(coda_index, "r") as f:
            for line in f:
                v, start, end = line.strip().split()

                match = pattern.fullmatch(v)
                name = match.group(1)
                index_str = match.group(2)

                values = res_lines[int(start) - 1: int(end)]

                if index_str is None:
                    result[name] = values
                    continue

                indices = list(map(int, index_str.split(",")))

                if name not in result:
                    result[name] = []

                arr = result[name]
                arr = ensure_size(arr, indices)

                ref = arr
                for d in range(len(indices) - 1):
                    idx = indices[d] - 1
                    ref[idx] = ensure_size(ref[idx], indices, depth=d + 1)
                    ref = ref[idx]

                ref[indices[-1] - 1] = values
                result[name] = arr

        final = []
        for var in sample_vars:
            arr = np.moveaxis(np.array(result[f"v{var._n}"]), -1, 0)
            if var.op.name == "Categorical":
                arr = arr - 1
            final.append(arr)
        return final