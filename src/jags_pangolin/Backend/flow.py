from .scalar_ops import Scalar_ops
from .Multi_funcs import Multi_funcs
from pangolin.ir import RV
class flow:
    def VMap(n, op, parents, ite, res):
        in_axes = op.in_axes
        axis_size = op.axis_size
        op = op.base_op
        # if(len(in_axes) != len(parents)):
        #     raise ValueError("Length of in_axes must be equal to number of parents")
        # for i in range(len(in_axes)):
        #     if in_axes[i] is not None and axis_size is None:
        #         axis_size = parents[i].shape[0]
        #         break
        # for i in range(len(in_axes)):
        #     if(in_axes[i] is not None and parents[i].ndim == 0):
        #         raise ValueError("Input should be a vector if in_axes is not None")
        #     elif(in_axes[i] is not None and parents[i].shape[0] != axis_size):
        #         raise ValueError("All inputs with in_axes must have the same leading dimension")
        #     elif(in_axes[i] is None and parents[i].ndim != 0):
        #         raise ValueError("All inputs without in_axes must have only 0 leading dimension")
        if(axis_size == None):
            for i in range(len(parents)):
                if(in_axes[i] != None):
                    axis_size = res[i].shape[ite]
        ans = f"for(i{ite} in 1:{axis_size})" + "{\n"
        code = ""
        pars = []
        for j in range(len(parents)):
            
            if(in_axes[j] is not None):
                if(parents[j][-1]==']'):
                    tmp = f"{parents[j][:-1]},i{ite}]"
                else:
                    tmp = f"{parents[j]}[i{ite}]"
            else:
                tmp = parents[j]
            pars.append(tmp)
        name = ""
        if(n[-1] == ']'):
            name = n[:-1] +f",i{ite}]"
        else:
            name = n+f"[i{ite}]"
        if(op.name == "Constant"):
            code += Scalar_ops.__dict__["Constant_after"](name, op)
        elif (flow.__dict__.get(op.name) is not None):
            code += flow.__dict__[op.name](name, op, pars, ite+1, res)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, res)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Autoregressive(n, op, parents, ite, res):
        length = op.length
        in_axes = op.in_axes
        where_self = op.where_self
        op = op.base_op
        # if(len(in_axes) != len(parents)-1):
        #     raise ValueError("Length of in_axes must be equal to number of parents")
        # if(where_self < 0 or where_self >= len(parents)):
        #     raise ValueError("The position of self should be in correct range")
        # if(parents[where_self].ndim != 0):
        #     raise ValueError("Only an initial constant for autoregressive parent")
        # offset = 0
        # for i in range(len(parents)):
        #     if i == where_self:
        #         offset+=1
        #         continue
        #     if(in_axes[i-offset] is not None and parents[i].ndim==0):
        #         raise ValueError("Should have in_axes as None if parent is a single value constant")
        #     elif(in_axes[i-offset] is None and parents[i].ndim != 0):
        #         raise ValueError("Should have in_axes as 0 if parent is a constant vector")
        #     elif(in_axes[i-offset] is not None and parents[i].shape[0] != length):
        #         raise ValueError("Length of vector should match length of autoregressive")
            
        pars = []
        offset = 0
        for j in range(len(parents)):
            if(j == where_self):
                pars.append(parents[j])
                offset+=1
                continue
            if(in_axes[j-offset] is not None):
                if(parents[j][-1] == ']'):
                    tmp = f"{parents[j][:-1]}, 1]"
                else:
                    tmp = f"{parents[j]}[1]"
            else:
                tmp = parents[j]
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
            ans += flow.__dict__[op.name](name, op, pars, ite, res)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, res)
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
            code += flow.__dict__[op.name](name, op, pars, ite, res)
        elif(Multi_funcs.__dict__.get(op.name) is not None):
            ans += Multi_funcs.__dict__[op.name](name, op, pars, res)
        else:
            code += Scalar_ops.__dict__[op.name](name, pars)
        code += "\n"
        ans += "  " + code + "}\n"
        return ans

    def Composite(n, op, parents, ite, res):
        num = op.num_inputs
        ops = op.ops
        par_nums = op.par_nums
        if(len(par_nums) != len(ops)):
            raise ValueError("number of ops should match the number of par_nums")
        if(num != len(parents)):
            raise ValueError("The number of parents should match num_inputs")
        new_list = []
        ans = ""
        for i in range(len(par_nums)):
            pars=[]
            for j in range(len(par_nums[i])):
                if(par_nums[i][j] < num):
                    pars.append(parents[par_nums[i][j]])
                else:
                    if(par_nums[i][j]-num >= len(new_list)):
                        raise ValueError("Can't take parent that hasn't been created")
                    pars.append(new_list[par_nums[i][j]-num])
            if(i<len(par_nums)-1):
                idd = n.find("[")
                if(idd == -1):
                    name = f"{n}_{i+1}"
                else:
                    name = n[:idd]+f"_{i+1}"+n[idd:]
                if(ops[i].name == "Constant"):
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i]) 
                elif(Scalar_ops.__dict__.get(ops[i].name) is not None):
                    code = Scalar_ops.__dict__[ops[i].name](name, pars) 
                elif(Multi_funcs.__dict__.get(ops[i].name) is not None):
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars, res) 
                elif(flow.__dict__.get(ops[i].name) is not None):
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite, res)
                new_list.append(name)
                ans+=code + "\n"
            else:
                name = n
                if(ops[i].name == "Constant"):
                    code = Scalar_ops.__dict__["Constant_after"](name, ops[i]) 
                elif(Scalar_ops.__dict__.get(ops[i].name) is not None):
                    code = Scalar_ops.__dict__[ops[i].name](name, pars) 
                elif(Multi_funcs.__dict__.get(ops[i].name) is not None):
                    code = Multi_funcs.__dict__[ops[i].name](name, ops[i], pars, res) 
                elif(flow.__dict__.get(ops[i].name) is not None):
                    code = flow.__dict__[ops[i].name](name, ops[i], pars, ite, res)
                ans+=code + "\n"
        
        return ans
                