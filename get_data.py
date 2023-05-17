import numpy as np
import torch


def getData(file_path):
    label = []
    feature = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('w'):
            label.append([0])
        else:
            label.append([1])
        line = line[1: -1].strip()
        line = line.split(' ')
        line = [float(i) for i in line]
        feature_ = [line[i: i + 2] for i in range(0, len(line), 2)]
        feature_.extend([[0.0, 0.0]] * (120 - len(feature_)))
        feature.append(feature_)


    feature = torch.Tensor(np.array(feature))
    label = torch.Tensor(np.array(label))
    print(feature.shape)
    print(label.shape)
    return feature, label

if __name__ == "__main__":
    label = getData('./eva_data.txt')
