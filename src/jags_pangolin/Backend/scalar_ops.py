import numpy as np 

class Scalar_ops:
    def Constant_before(n:str, node):
        code = ""
        if node.ndim == 0:
            code += f"{n} <- {node.op.value}\n"
        else:
            code += f"{n}<-structure(c("
            arr = np.array(node.op.value)
            flat = arr.flatten(order = 'F')
            # flat = np.round(flat, decimals=8)
            # print(flat)
            for i in range(len(flat)):
                if(i == len(flat)-1):
                    code += f"{flat[i]}"
                else:
                    code += f"{flat[i]},"
                # print(flat[i])
            code += f"),.Dim=c("
            for i in range(node.ndim):
                if(i == node.ndim-1):
                    code += f"{node.shape[i]}))\n"
                else:
                    code += f"{node.shape[i]},"
        return code
    
    def Constant_after(n:str, op):
        lines = []
        # print(n)
        def recurse(indices, subarr):
            # Check if the current subarr is a scalar
            if np.isscalar(subarr) or subarr.ndim == 0:
                # Only add indices if name does not already have brackets
                idx_str = ",".join(str(i+1) for i in indices)
                if(n[-1] != ']'):
                    full_name = f"{n}[{idx_str}]" if indices else n
                else:
                    full_name = f"{n[:-1]},{idx_str}]" if indices else n
                lines.append(f"{full_name} <- {subarr}")
            else:
                for i in range(subarr.shape[0]):
                    recurse(indices + [i], subarr[i])

        recurse([], np.array(op.value))
        return "\n".join(lines)

    def Add(n: str, parents):
        return f"{n} <- {parents[0]} + {parents[1]}"

    def Sub(n: str, parents):
        return f"{n} <- {parents[0]} - {parents[1]}"

    def Mul(n: str, parents):
        return f"{n} <- {parents[0]} * {parents[1]}"

    def Div(n: str, parents):
        return f"{n} <- {parents[0]} / {parents[1]}"

    def Pow(n: str, parents):
        return f"{n} <- {parents[0]} ^ {parents[1]}"

    def Normal(n: str, parents):
        return f"{n} ~ dnorm({parents[0]}, 1/({parents[1]}*{parents[1]}))"

    def Cauchy(n: str, parents):
        return f"{n} ~ dt({parents[0]}, 1/({parents[1]}*{parents[1]}), 1)"

    def NormalPrec(n: str, parents):
        return f"{n} ~ dnorm({parents[0]}, {parents[1]})"

    def Lognormal(n: str, parents):
        return f"{n} ~ dlnorm({parents[0]}, 1/({parents[1]}*{parents[1]}))"

    def Bernoulli(n: str, parents):
        return f"{n} ~ dbern({parents[0]})"

    def BetaBinomial(n: str, parents):
        idd = n.find('[')
        if(idd == -1):
            tmp = f"{n}_5"
        else:
            tmp = n[:idd]+"_5"+n[idd:]
        return (
            f"{n} ~ dbin({tmp}, {parents[0]})\n"
            f"{tmp} ~ dbeta({parents[1]}, {parents[2]})"
        )

    def BernoulliLogit(n: str, parents):
        idd = n.find('[')
        if(idd == -1):
            tmp = f"{n}_5"
        else:
            tmp = n[:idd]+"_5"+n[idd:]
        return (
            f"{n} ~ dbern({tmp})\n"
            f"logit({tmp}) <- {parents[0]}"
        )

    def Binomial(n: str, parents):
        return f"{n} ~ dbin({parents[1]}, {parents[0]})"

    def Uniform(n: str, parents):
        return f"{n} ~ dunif({parents[0]}, {parents[1]})"

    def Categorical(n: str, parents):
        return f"{n} ~ dcat({parents[0]})"

    def Beta(n: str, parents):
        return f"{n} ~ dbeta({parents[0]}, {parents[1]})"

    def Exponential(n: str, parents):
        return f"{n} ~ dexp({parents[0]})"

    def Gamma(n: str, parents):
        return f"{n} ~ dgamma({parents[0]}, {parents[1]})"

    def Poisson(n: str, parents):
        return f"{n} ~ dpois({parents[0]})"

    def StudentT(n: str, parents):
        return (
            f"{n} ~ dt({parents[1]}, 1/({parents[2]}*{parents[2]}), {parents[0]})"
        )

    def Abs(n: str, parents):
        return f"{n} <- abs({parents[0]})"

    def Arccos(n: str, parents):
        return f"{n} <- acos({parents[0]})"

    def Arcsin(n: str, parents):
        return f"{n} <- asin({parents[0]})"

    def Arccosh(n: str, parents):
        return f"{n} <- acosh({parents[0]})"

    def Arcsinh(n: str, parents):
        return f"{n} <- asinh({parents[0]})"

    def Arctan(n: str, parents):
        return f"{n} <- atan({parents[0]})"

    def Arctanh(n: str, parents):
        return f"{n} <- atanh({parents[0]})"

    def Cos(n: str, parents):
        return f"{n} <- cos({parents[0]})"

    def Sin(n: str, parents):
        return f"{n} <- sin({parents[0]})"

    def Tan(n: str, parents):
        return f"{n} <- tan({parents[0]})"

    def Cosh(n: str, parents):
        return f"{n} <- cosh({parents[0]})"

    def Sinh(n: str, parents):
        return f"{n} <- sinh({parents[0]})"

    def Tanh(n: str, parents):
        return f"{n} <- tanh({parents[0]})"

    def Exp(n: str, parents):
        return f"{n} <- exp({parents[0]})"

    def Log(n: str, parents):
        return f"{n} <- log({parents[0]})"

    def Loggamma(n: str, parents):
        return f"{n} <- loggam({parents[0]})"

    def InvLogit(n: str, parents):
        return f"{n} <- exp({parents[0]})/(1 + exp({parents[0]}))"

    def Logit(n: str, parents):
        return f"{n} <- log({parents[0]}/(1 - {parents[0]}))"

    def Step(n: str, parents):
        return f"{n} <- step({parents[0]})"
        
        