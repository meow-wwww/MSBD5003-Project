import numpy as np

def read_txt_file(file_path, with_data_id=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if '' in lines:
        lines.remove('')
    if '\n' in lines:
        lines.remove('\n')
    array = np.zeros((len(lines), 2)).astype(int)
    for i, line in enumerate(lines):
        array[i] = np.array(list(map(int, line.strip().split())))
        
    if with_data_id:
        data_id = np.array(list(range(len(lines))))
        array = np.concatenate([data_id.reshape(-1, 1), array], axis=1)
    return array