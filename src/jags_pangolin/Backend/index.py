class index:
    code = []
    def loop(self, name, cur, arr_name, id_names, index, num, each):
        if(len(cur) == num):
            tmp = name
            flag = 0
            for x in each:
                if(x.ndim!=0):
                    flag = 1
            if(flag == 1):
                for i in range(num):
                    if(tmp[-1]==']'):
                        tmp = tmp[:-1]+f",{index[i]}]"
                    else:
                        tmp = tmp + f"[{index[i]}]"
            tmp += f"<-{arr_name}"
            for i in range(num):
                if(tmp[-1] == ']'):
                    tmp = tmp[:-1] + f",{cur[i]}]"
                else:
                    tmp = tmp + f"[{cur[i]}]"
            self.code.append(tmp)
        else:
            if(each[len(cur)].ndim == 0):
                tmp = id_names[len(cur)]
                if(each[len(cur)].op.name == "Constant"):
                    tmp += "+ 1"
                self.loop(name, cur + [tmp], arr_name, id_names, index + [1], num, each)        
            else:
                for i in range(each[len(cur)].shape[0]):
                    tmp = f"{id_names[len(cur)]}[{i+1}]"
                    if(each[len(cur)].op.name == "Constant"):
                        tmp += "+ 1"
                    self.loop(name, cur + [tmp], arr_name, id_names, index + [i+1], num, each)                
    
    def SimpleIndex(self, n, parents, cur):
        ans = ""
        if(cur[0].ndim == 0):
            ans = f"{n} <- {parents[0]}[{parents[1]}]"
        elif cur[0].ndim == 1:
            for i in range(len(cur[1].shape[0])):
                ans += f"{n}[{i+1}] <- {parents[0]}[{parents[1]}[{i+1}]]"
        else:
            self.loop(n, [], parents[0], parents[1:], [], cur[0].ndim, cur[1:])
            ans = "\n".join(self.code)
        self.code = []
        return ans
