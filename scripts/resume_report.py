from os.path import exists, join
import json
import os
import numpy as np

BASE_PATH = join(os.getenv('HOME'), 'buffer_test')
world_to_report = [54, 94, 156, 68, 52, 101, 40, 135, 51, 42, 75, 67, 18, 53, 87, 36, 28, 61, 233, 25, 35, 20, 34, 79, 108, 46, 65, 90, 6, 73, 70, 10, 29, 167, 15, 31, 77, 116, 241, 155, 194, 99, 56, 149, 38, 261, 239, 234, 60, 173, 247, 178, 291, 16, 9, 21, 169, 257, 148, 296, 151, 259, 102, 145, 130, 205, 121, 105, 43, 242, 213, 171, 62, 202, 293, 224, 225, 152, 111, 55, 125, 200, 161, 1, 136, 106, 286, 139, 244, 230, 222, 238, 170, 267, 26, 132, 124, 23, 59, 3, 97, 119, 89, 12, 164, 39, 236, 263, 81, 188, 84, 11, 268, 192, 122, 22, 253, 219, 216, 137, 85, 195, 206, 212, 4, 274, 91, 248, 44, 131, 203, 63, 80, 37, 110, 50, 74, 120, 128, 249, 30, 14, 103, 49, 154, 82, 2, 143, 158, 147, 235, 83, 157, 142, 187, 185, 288, 45, 140, 271, 160, 146, 109, 223, 126, 98, 252, 134, 272, 115, 71, 117, 255, 141, 174, 33, 245, 92, 295, 281, 186, 260, 7, 166, 196, 66, 113, 153, 227, 107, 199, 298, 278, 114, 72, 165, 228, 176, 24, 162, 198, 180, 285, 232, 243, 207, 190, 262, 275, 172, 179, 269, 127, 86, 183, 273, 287, 215, 266, 95, 5, 299, 279, 13, 250, 96, 197, 177, 58, 289, 211, 220, 182, 282, 210, 280, 251, 283, 217, 276, 292, 221, 204, 191, 181, 209, 297, 264, 231, 254]

world_to_report = ["world_%d" %(w) for w in world_to_report]
path_to_report = join(BASE_PATH, "report.json")
path_to_default = join("scripts", "default_t20_train_cleaned.json")

def get_avg_len(path_to_report, world_to_report):
    with open(path_to_report) as f: 
        report = json.load(f)
        avg_lens = []
        for w in world_to_report:
            if w in report.keys():
                lens = [report[w]["ep_length"][i] if report[w]["succeed"][i] else 50\
                        for i in range(len(report[w]["ep_length"]))]
                avg_lens.append(np.mean(lens))
            else:
                print(w)
    return world_to_report, avg_lens
    
    
_, report_avg_lens = get_avg_len(path_to_report, world_to_report)   
_, default_avg_lens = get_avg_len(path_to_default, world_to_report)   

for i, w in enumerate(world_to_report):
    print("%s: policy: %f, default: %f" %(w, report_avg_lens[i], 2*default_avg_lens[i]))

print("Average over the worlds: policy: %f, default: %f" %(np.mean(report_avg_lens), 2*np.mean(default_avg_lens)))
