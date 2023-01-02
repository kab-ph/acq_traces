import numpy as np
from tqdm import tqdm
import matplotlib.pyplot
import h5py
import os


class acq:
    def __init__(self, no_samples, chunk):
        self.chunk = chunk
        self.no_samples = no_samples
        self.buff_init()
        self.pack = 0
        self.count = 0
        if('leakages' not in os.listdir()):
            os.mkdir("leakages")
        else:
            self.files = os.listdir("leakages")
            self.count_files('h5')

    def count_files(self, xten):
        self.pack = 0
        for i in self.files:
            if(xten in i):
                self.pack += 1

    def buff_init(self):
        self.buff = np.zeros((self.chunk, self.chunk), dtype='int8')

    def save_trace(self, trace):
        self.buff[self.count] = trace
        self.count += 1
        if self.count == self.chunk:
            file = h5py.File(f"leakages//{self.pack}.h5", 'w')
            file.create_dataset('leakages', data=self.buff)
            self.count = 0
            self.buff_init()
            self.pack += 1



class Leakeges:
    def __init__(self, dest):
        self.dir = 'leakages//0.h5'
        self.f = h5py.File(self.dir, 'r')
        self.buff = self.f['leakages'][:]
        self.rows, self.col = self.buff.shape
        self.count = len(os.listdir('leakages'))
        self.traces = np.zeros(((self.count * self.rows), self.col))

    def read(self):
        for i in tqdm(range(self.count)):
            self.dir = 'leakages' + f"//{i}.h5"
            self.f = h5py.File(self.dir, 'r')
            self.buff = self.f['leakages'][:]
            self.traces[(i * self.rows):((i * self.rows) + self.rows)] = self.buff
        return self.traces






def write_traces():
    chunck = 500
    no_samples = 200
    a = acq(no_samples, chunck)
    for i in tqdm(range(2000)):
        trace = np.random.uniform(20, size=500)
        a.save_trace(trace)

def read_traces():
    dir = 'leakages'
    leaks = Leakeges(dir)
    return leaks.read()


if __name__ == "__main__":
    traces = read_traces()
    # write_traces()
