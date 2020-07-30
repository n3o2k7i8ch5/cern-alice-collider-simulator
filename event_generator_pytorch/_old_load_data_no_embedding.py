from os.path import exists
from typing import List, Tuple
import torch

import pickle

from torch import Tensor

_FILE_NAME = 'data.data'


def load_data(device) -> Tuple[List[Tensor], int]:
    if exists(_FILE_NAME):
        with open(_FILE_NAME, 'rb') as input_file:
            (data, max_length) = pickle.load(input_file)
        return data, max_length

    file = open('raw_data', 'r')
    lines = file.readlines()

    event_list: List[Tensor] = []
    event = []
    max_length = 0

    for idx in range(len(lines)):

        if idx % 50_000 == 0:
            print(str(100 * idx / len(lines)) + '%')

        line = lines[idx].replace('\n', '')

        if len(line) == 0:
            event_list.append(torch.tensor(event))#.to(device=device))
            if max_length < len(event_list):
                max_length = len(event_list)
            event = []

            if idx > len(lines) / 10:
                break

            continue

        particle_params = []
        for param_idx, param in enumerate(line.split(' ')):

            if len(param) == 0:
                continue

            particle_params.append(float(param))

        event.append(particle_params)


    with open(_FILE_NAME, 'wb') as handle:
        pickle.dump((event_list, max_length), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return event_list, max_length
