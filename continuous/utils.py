import csv
# import pandas as pd
import numpy as np
import json
from os import path
from os.path import join, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

Benchmarking_train = [54, 94, 156, 68, 52, 101, 40, 135, 51, 42, 75, 67, 18, 53, 87, 36, 28, 61, 233, 25, 35, 20, 34, 79, 108, 46, 65, 90, 6, 73, 70, 10, 29, 167, 15, 31, 77, 116, 241, 155, 194, 99, 56, 149, 38, 261, 239, 234, 60, 173, 247, 178, 291, 16, 9, 21, 169, 257, 148, 296, 151, 259, 102, 145, 130, 205, 121, 105, 43, 242, 213, 171, 62, 202, 293, 224, 225, 152, 111, 55, 125, 200, 161, 1, 136, 106, 286, 139, 244, 230, 222, 238, 170, 267, 26, 132, 124, 23, 59, 3, 97, 119, 89, 12, 164, 39, 236, 263, 81, 188, 84, 11, 268, 192, 122, 22, 253, 219, 216, 137, 85, 195, 206, 212, 4, 274, 91, 248, 44, 131, 203, 63, 80, 37, 110, 50, 74, 120, 128, 249, 30, 14, 103, 49, 154, 82, 2, 143, 158, 147, 235, 83, 157, 142, 187, 185, 288, 45, 140, 271, 160, 146, 109, 223, 126, 98, 252, 134, 272, 115, 71, 117, 255, 141, 174, 33, 245, 92, 295, 281, 186, 260, 7, 166, 196, 66, 113, 153, 227, 107, 199, 298, 278, 114, 72, 165, 228, 176, 24, 162, 198, 180, 285, 232, 243, 207, 190, 262, 275, 172, 179, 269, 127, 86, 183, 273, 287, 215, 266, 95, 5, 299, 279, 13, 250, 96, 197, 177, 58, 289, 211, 220, 182, 282, 210, 280, 251, 283, 217, 276, 292, 221, 204, 191, 181, 209, 297, 264, 231, 254]
Benchmarking_test = [w for w in range(300) if w not in Benchmarking_train]

# train_worlds = [220, 181, 270, 246, 292, 195, 213, 254, 209, 221] # APPLD best 10 worlds
# train_worlds = list(range(300))
# train_worlds = [283, 293, 105, 153, 292, 254, 209, 221, 194, 245] # APPLD best 10 worlds new
# train_worlds = [74, 271, 213, 283, 265, 273, 137, 209, 245, 194] # APPLI best 10 worlds new
# train_worlds = [105, 284,  70,  17,  20, 196,  23,  50, 269, 254] # APPLI-APPLD best 10 worlds
# train_worlds = [74, 271, 213, 283, 265, 273, 137, 209, 194, 293, 105, 153, 292, 254, 221, 245]
train_worlds = Benchmarking_train + [213, 275, 289,  14,  98, 110,  63, 261, 241,  46]*5

def path_to_world(w):
    if w in Benchmarking_train:
        return 'Benchmarking/train/world_%d.world' %(w)
    elif w in Benchmarking_test:
        return 'Benchmarking/test/world_%d.world' %(w)
    else:
        raise ValueError("wrong world index!")
        return None

range_dict = {    
    'max_vel_x': [0.2, 2],    
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 20],    
    'vtheta_samples': [4, 60],    
    'occdist_scale': [0.05, 1],    
    'path_distance_bias': [0.1, 1.5],    
    'goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.01, 0.5]
}

APPLI_PARAMS = {
    "A1": [0.26, 2.00, 13, 44, 0.57, 0.76, 0.94, 0.02],
    "A2": [0.22, 0.87, 13, 31, 0.30, 0.36, 0.71, 0.30],
    "B1": [1.91, 1.70, 10, 47, 0.08, 0.71, 0.35, 0.23],
    "B2": [0.72, 0.73, 19, 59, 0.62, 1.00, 0.32, 0.24],
    "n1": [0.37, 1.33, 9 , 6 , 0.95, 0.83, 0.93, 0.01],
    "n2": [0.31, 1.05, 17, 20, 0.45, 0.61, 0.22, 0.23]
}

APPLD_PARAMS = {
    "curve"   : [0.80, 0.73, 6 , 42, 0.04, 0.98, 0.94, 0.19],
    "obstacle": [0.71, 0.91, 16, 53, 0.55, 0.54, 0.91, 0.39],
    "gap"     : [0.25, 1.34, 8 , 59, 0.43, 0.65, 0.98, 0.40],
    "open"    : [1.59, 0.89, 18, 18, 0.40, 0.46, 0.27, 0.42]
}

def load_appli_data():
    scan = []
    param = []
    for k in APPLI_PARAMS.keys():
        sc_pd = pd.read_csv("/home/users/zifanxu/APPLR-1/data/APPL-DI/APPLI/%s/_slash_front_slash_scan.csv" %(k))
        lg_pd = pd.read_csv("/home/users/zifanxu/APPLR-1/data/APPL-DI/APPLI/%s/_slash_local_goal.csv" %(k))
        lg_idx = 0
        for idx in sc_pd.index:
            ts = sc_pd.loc[idx]["rosbagTimestamp"]
            lg, lg_idx = find_local_goal(ts, lg_idx, lg_pd)
            sc_list = sc_pd.loc[idx]["ranges"][1:-1].split(",")
            scan.append([float(sc) for sc in sc_list] + [lg])
            param.append(APPLI_PARAMS[k])
    x = np.array(scan)
    y = np.array(param)
    return x, y

def load_appld_data():
    scan = []
    param = []
    for k in APPLD_PARAMS.keys():
        sc_pd = pd.read_csv("/home/users/zifanxu/APPLR-1/data/APPLD_Bags/%s/timed_front_scan/_slash_front_slash_scan.csv" %(k))
        lg_pd = pd.read_csv("/home/users/zifanxu/APPLR-1/data/APPL-DI/APPLD/%s_local_goal.csv" %(k))
        lg_pd.columns = ["rosbagTimestamp", "x", "y", "z", "ox", "oy", "oz", "w"]
        lg_idx = 0
        for idx in sc_pd.index:
            ts = sc_pd.loc[idx]["rosbagTimestamp"]
            lg, lg_idx = find_local_goal(ts, lg_idx, lg_pd)
            sc_list = sc_pd.loc[idx]["ranges"][1:-1].split(",")
            sc_list = [float(sc) for sc in sc_list]
            sc_list = from_2095_to_720(sc_list)
            scan.append(sc_list + [lg])
            param.append(APPLD_PARAMS[k])
    x = np.array(scan)
    y = np.array(param)
    return x, y

def from_2095_to_720(ranges):
    angles2 = np.linspace(-np.pi*3/4, np.pi*3/4, 721)[1:]
    inc = 2*np.pi/2096
    ranges2 = []
    for a in angles2:
        idx1 = int((a + np.pi-inc)/inc)
        idx2 = idx1+1
        r = (ranges[idx1]+ranges[idx2])/2
        ranges2.append(r)
    return ranges2

def preprocessing(x, y, laser_clip=4):
    x[x>laser_clip] = laser_clip
    x[:,:-1] = (x[:,:-1]-laser_clip/2.)/laser_clip
    x[:,-1] = x[:,-1]/np.pi
    return x, y


def find_local_goal(ts, lg_idx, lg_pd):
    lg = None
    while lg_pd.loc[lg_idx]["rosbagTimestamp"] < ts:
        lg_idx += 1
    x = lg_pd.loc[lg_idx]["x"]
    y = lg_pd.loc[lg_idx]["y"]
    lg = np.arctan2(y, x)
    return lg, lg_idx

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
