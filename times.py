import time
from collections import defaultdict
from globals import *


class Timings:
    def __init__(self):
        self.d = defaultdict(list)

    def get(self, key):
        self.d[key].append(time.perf_counter_ns())

    def info(self, last=True):
        for key, value in self.d.items():
            temp = [(value[i] - value[i-1])/1e9 for i in range(1, len(value), 2)]
            temp_ = np.array(temp)
            if last:
                res = {'key': key, 'mean': temp_.mean(), 'var': temp_.var(), 'last': temp[-1], 'time': temp}
            else:
                res = {'key': key, 'mean': temp_.mean(), 'var': temp_.var(), 'time': temp}
            print(res)
