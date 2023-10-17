import numpy as np

def main(data_list_path):
    # set seed
    np.random.seed(2023)
    with open(data_list_path, 'r') as f:
        data = [line.rstrip() for line in f]
    ind = np.arange(len(data))
    np.random.shuffle(ind)
    train_ind = ind[:int(len(data) * 0.8)]
    val_ind = ind[int(len(data) * 0.8):]

    train_data_list_path = data_list_path.replace('.list', '_train.list')
    val_data_list_path = data_list_path.replace('.list', '_val.list')

    with open(train_data_list_path, 'w') as f:
        for i in train_ind:
            f.write(data[i] + '\n')

    with open(val_data_list_path, 'w') as f:
        for i in val_ind:
            f.write(data[i] + '\n')

if __name__ == '__main__':
    data_list_path = './TheBeatles.list'
    main(data_list_path)