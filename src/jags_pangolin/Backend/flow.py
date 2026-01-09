from .scalar_ops import Scalar_ops
from .Multi_funcs import Multi_funcs
from pangolin.ir import RV
class flow:
    def VMap(n, op, parents, ite, shapes):
        in_axes = op.in_axes
        axis_size = op.axis_size
        op = op.base_op
        if(axis_size == None):
            for i in range(len(parents)):
                if(in_axes[i] != None):
                    axis_size = shapes[i][ite]
        ans = f"for(i{ite} in 1:{axis_size})" + "{\n"
        code = ""
        pars = []
        new_shapes = []
        for j in range(len(parents)):
            
            if(in_axes[j] is not None):
                if(parents[j][-1]==']'):
                    tmp = f"{parents[j][:-1]},i{ite}]"
                else:
                    tmp = f"{parents[j]}[i{ite}]"
                new_shapes.append(shapes[j][1:])
            else:
                tmp = parents[j]
                new_shapes.append(shapes[j])
            pars.append(tmp)
        name = ""
        if(n[-1] == ']'):
            name = n[:-1] +f",i{ite}]"
        else:
            name = n+f"[i{ite}]"
        if(op.name == "Constant"):
            code += Scalar_ops.__dict__["Constant_after"](name, op)
        elif (flow.__dict__.get(op.name) is not None):
            code += flow.__dict__[op.name](name, op, pars, ite+1, new_shapes)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, new_shapes)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Autoregressive(n, op, parents, ite, shapes):
        length = op.length
        in_axes = op.in_axes
        where_self = op.where_self
        op = op.base_op
        pars = []
        new_shapes = []
        offset = 0
        for j in range(len(parents)):
            if(j == where_self):
                pars.append(parents[j])
                offset+=1
                new_shapes.append(shapes[j])
                continue
            if(in_axes[j-offset] is not None):
                if(parents[j][-1] == ']'):
                    tmp = f"{parents[j][:-1]}, 1]"
                else:
                    tmp = f"{parents[j]}[1]"
                new_shapes.append(shapes[j][1:])
            else:
                tmp = parents[j]
                new_shapes.append(shapes[j])
            pars.append(tmp)
        name = ""
        if(n[-1] == ']'):
            name = n[:-1] +',1]'
        else:
            name = n+'[1]'
        ans = ""
        if(op.name == "Constant"):
            ans += Scalar_ops.__dict__["Constant_after"](name, op)
        elif (flow.__dict__.get(op.name) is not None):
            ans += flow.__dict__[op.name](name, op, pars, ite+1, new_shapes)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, new_shapes)
        else:
            ans += Scalar_ops.__dict__[op.name](name, pars)
        ans += "\n"
        pars = []
        offset = 0
        ans += f"for(i{ite} in 2:{length})" + "{\n"
        for j in range(len(parents)):
            if(j == where_self):
                if(n[-1] == ']'):
                    tmp = f"{n[:-1]}, i{ite}-1]"
                else:
                    tmp = f"{n}[i{ite}-1]"
                offset+=1
            else:
                if(in_axes[j-offset] is not None):
                    if(parents[j][-1] == ']'):
                        tmp = f"{parents[j][:-1]}, i{ite}]"
                    else:
                        tmp = f"{parents[j]}[i{ite}]"
                else:
                    tmp = parents[j]
            pars.append(tmp)
        name = ""
        if(n[-1] == ']'):
            name = n[:-1] +f",i{ite}]"
        else:
            name = n +f"[i{ite}]"
        code = ""
        if(op.name == "Constant"):
            code += Scalar_ops.__dict__["Constant_after"](name, op)
        elif (flow.__dict__.get(op.name) is not None):
            code += flow.__dict__[op.name](name, op, pars, ite+1, new_shapes)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, new_shapes)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Composite(n, op, parents, ite, shapes):
        num = op.num_inputs
        ops = op.ops
        par_nums = op.par_nums
        if(len(par_nums) != len(ops)):
            raise ValueError("number of ops should match the number of par_nums")
        if(num != len(parents)):
            raise ValueError("The number of parents should match num_inputs")
        new_list = []
        new_shapes = []
        ans = ""
        for i in range(len(par_nums)):
            pars=[]
            shapes_tmp = []
            for j in range(len(par_nums[i])):
                if(par_nums[i][j] < num):
                    # print(len(par_nums[i]), len(shapes))
                    pars.append(parents[par_nums[i][j]])
                    shapes_tmp.append(shapes[par_nums[i][j]])
                else:
                    if(par_nums[i][j]-num >= len(new_list)):
                        raise ValueError("Can't take parent that hasn't been created")
                    pars.append(new_list[par_nums[i][j]-num])
                    shapes_tmp.append(new_shapes[par_nums[i][j]-num])
            idd = n.find("[")
            if(idd == -1):
                name = f"{n}_{ite}_{i+1}"
            else:
                name = n[:idd]+f"_{ite}_{i+1}"+n[idd:]
            if(i<len(par_nums)-1):
                if(ops[i].name == "Constant"):
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i]) 
                elif(Scalar_ops.__dict__.get(ops[i].name) is not None):
                    code = Scalar_ops.__dict__[ops[i].name](name, pars) 
                elif(Multi_funcs.__dict__.get(ops[i].name) is not None):
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars, shapes_tmp) 
                elif(flow.__dict__.get(ops[i].name) is not None):
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite+1, shapes_tmp)
                new_list.append(name)
                new_shapes.append(ops[i].get_shape(*[shapes_tmp[k] for k in range(len(shapes_tmp))]))
                ans+=code + "\n"
            else:
                name = n
                if(ops[i].name == "Constant"):
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i]) 
                elif(Scalar_ops.__dict__.get(ops[i].name) is not None):
                    code = Scalar_ops.__dict__[ops[i].name](name, pars) 
                elif(Multi_funcs.__dict__.get(ops[i].name) is not None):
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars, shapes_tmp) 
                elif(flow.__dict__.get(ops[i].name) is not None):
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite, shapes_tmp)
                ans+=code + "\n"
        return ans
                