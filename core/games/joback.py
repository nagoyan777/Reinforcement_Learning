import thermo
import numpy as np


class Joback():
    def __init__(self):
        self.nodes = ['C', 'O', 'N', 'c1ccc(cc1)', 'C(=O)']
        self.counts = [0] * len(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def reset(self):
        pass

    def step(self, action):
        reward = None
        done = None
        j = thermo.joback.Joback(smiles)
        data = j.estimate()
        reward = data['Tb']
        return reward, done
