import os
import torch
import numpy as np
import joblib
import csv
import json


def preemptive_save(obj, save_path, type="torch"):
    # temp_path = save_path + ".tmp"
    save_path = str(save_path)
    extension = save_path.split('.')[-1]
    temp_path = save_path[:-len(extension)] + "tmp." + extension
    if type == "torch":
        torch.save(obj, temp_path)
    elif type == "numpy":
        np.save(temp_path, obj)
    elif type == "joblib":
        joblib.dump(obj, temp_path)
    elif type == "csv":
        with open(temp_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(obj)
    elif type == "json":
        json.dump(obj, open(temp_path, 'w'))
    else:
        raise NotImplementedError
    os.replace(temp_path, save_path)
