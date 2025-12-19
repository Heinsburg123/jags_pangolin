import numpy as np
from pangolin.ir import *

class Multi_funcs:
    def Matmul(n, op, parents, res):
        shape = op.get_shape(res[0].shape, res[1].shape)
        idd = n.find('[')
        idd2 = parents[0].find('[')
        idd3 = parents[1].find('[')
        if(idd != -1):
            if(len(shape)==1):
                name = n[:-1]+f",1:{shape[0]}]"
            else:
                name = n[:-1]+f",1:{shape[0]}, 1:{shape[1]}]"
        else:
            name = n

        if(idd2 != -1):
            if(len(res[0].shape)==1):
                name2 = parents[0][:-1]+f",1:{res[0].shape[0]}]"
            else:
                name2 = parents[0][:-1]+f",1:{res[0].shape[0]}, 1:{res[0].shape[1]}]"
        else:
            name2 = parents[0]

        if(idd3 != -1):
            if(len(shape)==1):
                name3 = parents[1][:-1]+f",1:{res[1].shape[0]}]"
            else:
                name3 = parents[1][:-1]+f",1:{res[1].shape[0]}, 1:{res[1].shape[1]}]"
        else:
            name3 = parents[1]

        return f"{name} <- {name2} %*% {name3}\n"

    def Sum(n, op, parents, res):
        code = ""
        offset = 0
        for i in range(res[0].ndim):
            if(i!=op.axis):
                code += f"for (j{i-offset} in 1:{res[0].shape[i]})"+"{\n"
            else:
                offset+=1
        code+=n
        for i in range(res[0].ndim-1):
            if(code[-1]==']'):
                code = code[:-1]+f",j{i}]"
            else:
                code = code + f"[j{i}]"
        code+= f"<- sum({parents[0]}"
        offset = 0
        for i in range(res[0].ndim):
            if(code[-1]==']'):
                if(i == op.axis):
                    code = code[:-1]+",]"
                    offset+=1
                else:
                    code = code[:-1]+f",j{i-offset}]"
            else:
                if(i == op.axis):
                    code = code+f"[]"
                    offset+=1
                else:
                    code = code+f"[j{i-offset}]"
        if(code[-1]!=']'):
            code+="[]"
        code+=")\n"
        for i in range(res[0].ndim-1):
            code +="}"
        return code
    def Softmax(n, op, parents, res):
        k = res[0].shape[0]
        idd = n.find('[')
        if(idd == -1):
            name1 = f"{n}_1"+n[idd:]+"[j]"
            name2 = f"{n}_2"+n[idd:]
            name3 = f"{n}[j]"
            tmp = f"{n}_1"+n[idd:]+"[]"
        else:
            name1 = n[:idd]+f"_1"+n[idd:-1]+",j]"
            name2 = n[:idd]+f"_2"+n[idd:]
            name3 = n[:-1]+",j]"
            tmp = f"{n}_1"+n[idd:-1]+",]"
        idd2 = parents[0].find('[')
        if(idd2 == -1):
            par_name = f"{parents[0]}[j]"
        else:
            par_name = f"{parents[0][:-1]}[j]"
        code = ""
        code += f"for (j in 1:{k})"+"{\n"
        code += f"  {name1} <-exp({par_name})\n"+"}\n"
        code += f"{name2} <- sum({tmp})\n"
        code += f"for (j in 1:{k})" + "{\n"
        code += f"  {name3} <- {name1}/{name2}\n" + "}\n"
        return code
    
    def MultiNormal(n, op, parents, res):
        p = res[0].shape[0]
        idd1 = n.find('[')
        idd2 = parents[0].find('[')
        idd3 = parents[1].find('[')
        if(idd1==-1):
            name1 = f"{n}[1:{p}]"
        else:
            name1 = n[:-1]+f",1:{p}]"
        if(idd2==-1):
            name2 = f"{parents[0]}[1:{p}]"
        else:
            name2 = parents[0][:-1]+f",1:{p}]"
        if(idd3==-1):
            name3 = f"{parents[1]}[1:{p},1:{p}]"
        else:
            name3 = parents[1][:-1]+f",1:{p},1:{p}]"
        return f"{name1} ~ dmnorm({name2}, inverse({name3}))"
    
    def Multinomial(n, op, parents, res):
        p = res[1].shape[0]
        idd1 = n.find('[')
        idd3 = parents[1].find('[')
        if(idd1==-1):
            name1 = f"{n}[1:{p}]"
        else:
            name1 = n[:-1]+f",1:{p}]"
        if(idd3==-1):
            name3 = f"{parents[1]}[1:{p}]"
        else:
            name3 = parents[1][:-1]+f",1:{p}]"
        return f"{name1} ~ dmulti({name3}, {parents[0]})"

    def Dirichlet(n, op, parents, res):
        p = res[0].shape[0]
        idd1 = n.find('[')
        idd2 = parents[0].find('[')
        if(idd1==-1):
            name1 = f"{n}[1:{p}]"
        else:
            name1 = n[:-1]+f",1:{p}]"
        if(idd2==-1):
            name2 = f"{parents[0]}[1:{p}]"
        else:
            name2 = parents[0][:-1]+f",1:{p}]"
        return f"{name1} ~ ddirch({name2})"


    