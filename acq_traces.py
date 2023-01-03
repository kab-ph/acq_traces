import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import os
from scipy import fft
from scipy import signal



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
        self.buff = np.zeros((self.chunk, self.no_samples), dtype='int8')

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



class alignment:
    def align_trace(self, ref, trace, norm, visulize):

        self.ref1 = np.concatenate((ref, np.zeros(len(trace) - len(ref))))
        if (np.std(trace) != 0):
            if norm:
                self.ref1 = (self.ref1 - np.mean(self.ref1)) / (np.std(self.ref1) * len(self.ref1))
                trace = (trace - np.mean(trace)) / (np.std(trace))
            self.corr = signal.correlate(trace, self.ref1, 'full')
            self.corr = self.corr[int(len(trace)):int(2 * len(trace))]
            # corr=corr[0:200]
            self.max_corr = np.max(self.corr)
            # print(max_corr)
            self.peak_position = np.where(self.corr == self.max_corr)[0][0]
        else:
            # input('error')
            self.peak_position = 100
        if visulize:
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.plot(trace)
            ax2.plot(self.corr)
            plt.show()
        # self.trace_out = trace[self.peak_position:self.peak_position + len(ref)]
        return self.peak_position



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


def align_traces():
    f = h5py.File('19.h5', 'r')
    f = f['leakages'][:]
    f = f[:, 10000:32000]
    # ref = f[0, 1000: 17000]
    ref = np.load('ref.npy')
    a = alignment()
    aligned_traces = np.zeros((f.shape[0], 20600))
    # indexMax = a.align_trace(ref, f[1], norm=True, visulize=True)

    for x, y in enumerate(f):
        indexMax = a.align_trace(ref, y, norm=True, visulize=False)
        # print('index', indexMax)
        aligned_traces[x] = y[indexMax : indexMax + 20600]

    for i in range(50):
        plt.plot(aligned_traces[i])
    plt.show()






if __name__ == "__main__":
    # traces = read_traces()
    # write_traces()
    align_traces()
