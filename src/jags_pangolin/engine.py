import pyjags
from pangolin.ir import *
from .Backend.scalar_ops import Scalar_ops
from .Backend.Multi_funcs import Multi_funcs
from .Backend.flow import flow
from .Backend.index import index
import platform
import re
import tempfile
from pathlib import Path


def ensure_size(arr, sizes, depth=0):
def ensure_size(arr, sizes, depth=0):
    while len(arr) < sizes[depth]:
        if depth == len(sizes) - 1:
            arr.append(None)
            arr.append(None)
        else:
            arr.append([])
    return arr


def write_constant(f, name, val):
    """Write a constant to a JAGS data.R file with full float64 precision (:.17g)."""
    arr = np.array(val, dtype=np.float64)
    if arr.ndim == 0:
        f.write(f"{name} <- {float(arr):.17g}\n")
    elif arr.ndim == 1:
        vals_str = ", ".join(f"{v:.17g}" for v in arr)
        f.write(f"{name} <- c({vals_str})\n")
    else:
        # JAGS is column-major, so flatten with order='F'
        flat = arr.flatten(order='F')
        vals_str = ", ".join(f"{v:.17g}" for v in flat)
        dims = ", ".join(str(d) for d in arr.shape)
        f.write(f"{name} <- structure(c({vals_str}), .Dim = c({dims}))\n")


class Sample_prob:
    class RunDFS:
        def __init__(self):
            self.visited = {}


        def dfs(self, node):
            name = "v" + str(node._n)
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
            return self.visited

    def sample(self, sample_vars: list[RV], kwargs=[], values=[], niter=1000, debug=False):
        dic = {}
        for var in kwargs:
            dic["v" + str(var._n)] = var
        for sample_var in sample_vars:
            dic["v" + str(sample_var._n)] = sample_var
        app = self.RunDFS()
        res = app.run_dfs(dic)

        # --- Build data dict with full float64 precision ---
        # pyjags passes this directly to JAGS via C API — no text file, no truncation
        data = {}
        for node in res:
            if res[node].op.name == "Constant":
                data[node] = np.array(res[node].op.value, dtype=np.float64)
        for i in range(len(kwargs)):
            data[f"v{kwargs[i]._n}"] = np.array(values[i], dtype=np.float64)

        # --- Build model string (identical logic to before) ---
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

        if debug:
            print("=== model ===")
            print(model_code)
            print("=== data ===")
            print(data)

        # --- Run with pyjags ---
        # adapt=1000 matches the original "update 1000" burn-in before monitoring
        monitor_vars = [f"v{v._n}" for v in sample_vars]
        model = pyjags.Model(code=model_code, data=data, chains=1, adapt=1000)
        # pyjags returns shape: (dim_1, ..., dim_n, iterations, chains)
        samples = model.sample(niter, vars=monitor_vars)

        # --- Repack to match original return format: (niter, dim_1, ..., dim_n) ---
        final = []
        for var in sample_vars:
            key = f"v{var._n}"
            arr = samples[key]           # shape: (..., niter, 1)
            arr = arr[..., 0]            # drop the single chains axis → (..., niter)
            arr = np.moveaxis(arr, -1, 0)  # → (niter, ...)
            final.append(arr)

        return final