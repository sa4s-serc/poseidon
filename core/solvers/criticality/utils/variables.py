import numpy as np

def init_all_vars(data):
    x_jr = np.zeros(shape=(int(len(data.nodes)),int(data.requests_received)))
    c_fj = np.zeros(shape=(int(len(data.functions)),int(len(data.nodes))))
    y_j = np.zeros(int(len(data.nodes)))
    S_active = np.zeros(shape=(int(len(data.functions)),int(len(data.nodes))))
    return x_jr, c_fj, y_j, S_active