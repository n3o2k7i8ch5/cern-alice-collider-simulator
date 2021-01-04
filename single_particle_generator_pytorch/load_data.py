from os.path import exists
from typing import List
import torch
import numpy as np

from torch import Tensor

from common.consts import *

_DATA_FILE_NAME = os.path.join(parent_path(), 'data/particle_data.data')
_RAW_DATA = os.path.join(parent_path(), 'data/raw_data')


def load_data() -> Tensor:
    if exists(_DATA_FILE_NAME):
        return torch.load(_DATA_FILE_NAME)

    file = open(_RAW_DATA, 'r')
    lines = file.readlines()

    event_list_cat: List[Tensor] = []
    event_list_cont: List[Tensor] = []

    event_cat = []
    event_cont = []

    max_length = 0

    for idx in range(len(lines)):

        if idx % 50_000 == 0:
            print(str(100 * idx / len(lines)) + '%')

        line = lines[idx].replace('\n', '')

        if len(line) == 0:
            event_list_cat.append(torch.tensor(event_cat))
            event_list_cont.append(torch.tensor(event_cont))

            if max_length < len(event_list_cat):
                max_length = len(event_list_cat)

            event_cat = []
            event_cont = []

            if idx > len(lines) / 20:
                break

            continue

        particle_params_cat = []
        particle_params_cont = []

        for param_idx, param in enumerate(line.split(' ')):

            if len(param) == 0:
                continue

            if param_idx == 0:  # PDG
                particle_params_cat.append(particles().index(int(param)))
            elif param_idx == 1:
                particle_params_cat.append(int(param))  # STATUS CODE - I assume it is a 1 or 0.
            #elif param_idx == 2:  # MOTHER 1
            #    particle_params_cat.append(int(param))
            #elif param_idx == 3:  # MOTHER 2
            #    particle_params_cat.append(int(param))
            #elif param_idx == 4:  # DAUGHTER 1
            #    particle_params_cat.append(int(param))
            #elif param_idx == 5:  # DAUGHTER 2
            #    particle_params_cat.append(int(param))
            elif param_idx == 6:  # WEIGHT
                particle_params_cont.append(sgnlog(float(param)))
            elif param_idx == 7:  # PX
                particle_params_cont.append(sgnlog(float(param)))
            elif param_idx == 8:  # PY
                particle_params_cont.append(sgnlog(float(param)))
            elif param_idx == 9:  # PZ
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 10:  # Energy
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 11:  # Vx
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 12:  # Vy
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 13:  # Vz
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 14:  # Polar Theta
                particle_params_cont.append(sgnlog(float(param))/10)
            elif param_idx == 15:  # Polar Phi
                particle_params_cont.append(sgnlog(float(param))/10)

        event_cont.append(particle_params_cont)
        event_cat.append(particle_params_cat)

    data_cat = event_list_cat
    data_cont = event_list_cont

    data = []

    for idx_e in range(len(data_cat)):
        for idx_p in range(len(data_cat[idx_e])):
            prtc_cat = data_cat[idx_e][idx_p].float()
            prtc_cont = data_cont[idx_e][idx_p].float()
            prtc = torch.cat([prtc_cat, prtc_cont], dim=0)
            data.append(prtc.unsqueeze(dim=0))

    data = torch.cat(data)

    torch.save(data, _DATA_FILE_NAME)

    return data


def sgnlog(val: float) -> float:
    sgn = 1 if val > 0 else -1
    logval = np.log((val * sgn) + 1)
    return float(logval * sgn)