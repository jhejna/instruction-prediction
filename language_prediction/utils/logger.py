import os
import csv
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger(object):

    def __init__(self, path):
        self.path = path
        self.writer = SummaryWriter(log_dir=path)        
        self.tb_values = {}
        self.csv_values = {}
        self.csv_logger = None

    def record(self, key, value):
        self.tb_values[key] = value
        self.csv_values[key] = value

    def dump(self, step, dump_csv=False):
        for k in self.tb_values.keys():
            self.writer.add_scalar(k, self.tb_values[k], step)
        self.writer.flush()
        # TB values get cleared every time.
        self.tb_values.clear()
        if dump_csv:
            if self.csv_logger is None:
                self.csv_file_handler = open(os.path.join(self.path, "log.csv"), "wt")
                self.csv_logger = csv.DictWriter(self.csv_file_handler, fieldnames=list(self.csv_values.keys()))
                self.csv_logger.writeheader()
            self.csv_logger.writerow(self.csv_values)
            self.csv_file_handler.flush()

    def log_from_dict(self, metrics, prefix):
        '''
        This function records data from an input dict, and removes the keys from the dict once they have been logged.
        '''
        keys_to_remove = []
        for loss_name, loss_value in metrics.items():
            log_name = prefix + "/" + loss_name
            if isinstance(loss_value, list) and len(loss_value) > 0:
                self.record(log_name, np.mean(loss_value))
            else:
                self.record(log_name, loss_value)
            keys_to_remove.append(loss_name)
        for key in keys_to_remove:
            del metrics[key]
