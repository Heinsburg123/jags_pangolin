from .scalar_ops import Scalar_ops
from .Multi_funcs import Multi_funcs
from pangolin.ir import RV

class flow:
    def _resolve_parent(parent):
        if isinstance(parent, tuple):
            base, idx_list, orig_shape = parent
            parts = [
                idx if idx is not None else f"1:{orig_shape[d]}"
                for d, idx in enumerate(idx_list)
            ]

            if base.endswith(']'):
                return f"{base[:-1]}, {', '.join(parts)}]"
            return f"{base}[{', '.join(parts)}]"
        return parent  

    def _add_index_to_parent(parent, idx):

        if isinstance(parent, tuple):
            base, idx_list, orig_shape = parent
            new_idx = list(idx_list)
            for d, v in enumerate(new_idx):
                if v is None:
                    new_idx[d] = str(idx)
                    break
            return (base, new_idx, orig_shape)
        else:
            if parent.endswith(']'):
                return f"{parent[:-1]}, {idx}]"
            else:
                return f"{parent}[{idx}]"

    def VMap(n, op, parents, ite, shapes):
        in_axes = op.in_axes
        axis_size = op.axis_size
        op = op.base_op
        if axis_size is None:
            for i in range(len(parents)):
                if in_axes[i] is not None:
                    axis_size = shapes[i][in_axes[i]]

        ans = f"for(i{ite} in 1:{axis_size})" + "{\n"
        code = ""
        pars = []
        pars_str = []
        new_shapes = []

        for j in range(len(parents)):
            if in_axes[j] is not None:
                axis = in_axes[j]
                if isinstance(parents[j], tuple):
                    base, idx_list, orig_shape = parents[j]
                else:
                    base = parents[j]
                    orig_shape = shapes[j]         
                    idx_list = [None] * len(orig_shape)

                none_positions = [i for i, v in enumerate(idx_list) if v is None]
                orig_dim = none_positions[axis]

                new_idx = list(idx_list)
                new_idx[orig_dim] = f"i{ite}"

                pars.append((base, new_idx, orig_shape))

                parts = [
                    new_idx[d] if new_idx[d] is not None else f"1:{orig_shape[d]}"
                    for d in range(len(orig_shape))
                ]

                if base.endswith(']'):
                    pars_str.append(f"{base[:-1]}, {', '.join(parts)}]")
                else:
                    pars_str.append(f"{base}[{', '.join(parts)}]")

                new_shapes.append(shapes[j][:axis] + shapes[j][axis + 1:])
            else:
                pars.append(parents[j])
                pars_str.append(flow._resolve_parent(parents[j]))
                new_shapes.append(shapes[j])

        name = ""
        if n[-1] == ']':
            name = n[:-1] + f",i{ite}]"
        else:
            name = n + f"[i{ite}]"

        if op.name == "Constant":
            code += Scalar_ops.__dict__["Constant_after"](name, op)
        elif flow.__dict__.get(op.name) is not None:
            code += flow.__dict__[op.name](name, op, pars, ite + 1, new_shapes)
        elif Multi_funcs.__dict__.get(op.name) is not None:

            code += Multi_funcs.__dict__[op.name](name, op, pars_str, new_shapes)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars_str)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Scan(n, op, parents, ite, shapes):
        length = op.length
        in_axes = op.in_axes
        where_self = op.where_self
        op = op.base_op

        pars = []
        new_shapes = []
        offset = 0
        for j in range(len(parents)):
            if j == where_self:
                pars.append(parents[j])
                offset += 1
                new_shapes.append(shapes[j])
                continue
            if in_axes[j - offset] is not None:
                # FIX: use _add_index_to_parent so tuple parents from an outer
                # VMap are handled correctly instead of raw string slicing.
                tmp = flow._add_index_to_parent(parents[j], 1)
                new_shapes.append(shapes[j][1:])   # Scan always strips leading dim
            else:
                tmp = parents[j]
                new_shapes.append(shapes[j])
            pars.append(tmp)

        # Resolve to plain JAGS strings for leaf ops
        pars_str = [flow._resolve_parent(p) for p in pars]

        name = n[:-1] + ',1]' if n[-1] == ']' else n + '[1]'

        ans = ""
        if op.name == "Constant":
            ans += Scalar_ops.__dict__["Constant_after"](name, op)
        elif flow.__dict__.get(op.name) is not None:
            # Pass structured pars so inner VMap/Scan/Composite levels work
            ans += flow.__dict__[op.name](name, op, pars, ite + 1, new_shapes)
        elif Multi_funcs.__dict__.get(op.name) is not None:
            ans += Multi_funcs.__dict__[op.name](name, op, pars_str, new_shapes)
        else:
            ans += Scalar_ops.__dict__[op.name](name, pars_str)
        ans += "\n"

        # ── Loop body (time index = i{ite}, from 2 to length) ─────────────────
        pars = []
        offset = 0
        ans += f"for(i{ite} in 2:{length})" + "{\n"
        for j in range(len(parents)):
            if j == where_self:
                # Self-reference: index into the *output* array at previous step.
                # `n` is always a plain string (it's the name being assigned here),
                # so string arithmetic is safe.
                if n[-1] == ']':
                    tmp = f"{n[:-1]}, i{ite}-1]"
                else:
                    tmp = f"{n}[i{ite}-1]"
                offset += 1
            else:
                if in_axes[j - offset] is not None:
                    # FIX: same as the seed step – use the helper, not raw slicing
                    tmp = flow._add_index_to_parent(parents[j], f"i{ite}")
                else:
                    tmp = parents[j]
            pars.append(tmp)

        # Resolve to plain JAGS strings for leaf ops
        pars_str = [flow._resolve_parent(p) for p in pars]

        name = n[:-1] + f",i{ite}]" if n[-1] == ']' else n + f"[i{ite}]"

        code = ""
        if op.name == "Constant":
            code += Scalar_ops.__dict__["Constant_after"](name, op)
        elif flow.__dict__.get(op.name) is not None:
            # FIX: pass structured pars (not pars_str) so nested flow ops work.
            # Previously the loop body passed raw strings, breaking any
            # VMap/Composite that Scan wrapped around.
            code += flow.__dict__[op.name](name, op, pars, ite + 1, new_shapes)
        elif Multi_funcs.__dict__.get(op.name) is not None:
            # FIX: was `ans +=` (missing the indented "  " prefix); use `code`.
            code += Multi_funcs.__dict__[op.name](name, op, pars_str, new_shapes)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars_str)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Composite(n, op, parents, ite, shapes):
        num = op.num_inputs
        ops = op.ops
        par_nums = op.par_nums
        if len(par_nums) != len(ops):
            raise ValueError("number of ops should match the number of par_nums")
        if num != len(parents):
            raise ValueError("The number of parents should match num_inputs")
        new_list = []
        new_shapes = []
        ans = ""
        for i in range(len(par_nums)):
            pars = []
            shapes_tmp = []
            for j in range(len(par_nums[i])):
                if par_nums[i][j] < num:
                    pars.append(parents[par_nums[i][j]])
                    shapes_tmp.append(shapes[par_nums[i][j]])
                else:
                    if par_nums[i][j] - num >= len(new_list):
                        raise ValueError("Can't take parent that hasn't been created")
                    pars.append(new_list[par_nums[i][j] - num])
                    shapes_tmp.append(new_shapes[par_nums[i][j] - num])

            # FIX: resolve structured tuple parents to plain JAGS strings before
            # passing to leaf ops (Scalar_ops / Multi_funcs).  Flow ops receive
            # the original `pars` so they can continue filling index slots.
            pars_str = [flow._resolve_parent(p) for p in pars]

            idd = n.find("[")
            if idd == -1:
                name = f"{n}_{ite}_{i+1}"
            else:
                name = n[:idd] + f"_{ite}_{i+1}" + n[idd:]

            if i < len(par_nums) - 1:
                if ops[i].name == "Constant":
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i])
                elif Scalar_ops.__dict__.get(ops[i].name) is not None:
                    code = Scalar_ops.__dict__[ops[i].name](name, pars_str)
                elif Multi_funcs.__dict__.get(ops[i].name) is not None:
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars_str, shapes_tmp)
                elif flow.__dict__.get(ops[i].name) is not None:
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite + 1, shapes_tmp)
                new_list.append(name)
                new_shapes.append(ops[i].get_shape(*[shapes_tmp[k] for k in range(len(shapes_tmp))]))
                ans += code + "\n"
            else:
                name = n
                if ops[i].name == "Constant":
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i])
                elif Scalar_ops.__dict__.get(ops[i].name) is not None:
                    code = Scalar_ops.__dict__[ops[i].name](name, pars_str)
                elif Multi_funcs.__dict__.get(ops[i].name) is not None:
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars_str, shapes_tmp)
                elif flow.__dict__.get(ops[i].name) is not None:
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite, shapes_tmp)
                ans += code + "\n"
        return ans