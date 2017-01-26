import numpy as np
from PIL import Image
import glob
from image_preprocess import process
import time

possible_actions = ['0', '3', '4']
label_to_indice = [0, -1, -1, 1, 2]

dataset_dir = '../dataset'
data = []

label_dir = glob.glob(dataset_dir + '/*')
print(label_dir)
for label in label_dir:
    filenames = glob.glob(label + '/*.jpg')
    print(label)
    action_indice = label_to_indice[int(label[-1])]
    # print(action_indice) 0,1,2
    for f in filenames:  # f = '../dataset/16/8_234.jpg'
        start_time = time.time()

        tmp = f[13:-4].split('_')
        simulation_id = int(tmp[0])
        frame_nb = int(tmp[1])

        image = Image.open(f)
        y_player, y_opponent, x_b, y_b, _, _, _ = process(image)

        data.append([simulation_id, frame_nb, action_indice, y_player, y_opponent, x_b, y_b])
        print(simulation_id, frame_nb, action_indice, y_player, y_opponent, x_b, y_b)
        duration = time.time() - start_time
        print(duration)
np.save('../data_unsorted.npy', np.array(data))

# Sort the data by simulation
data = np.array(data)
ind_sort = np.argsort(data[:, 0])
data = data[ind_sort]

# For each simulation sort data by frame nb
output = []
for i in range(1, 9):
    a1 = np.array([l for l in data if l[0] == i])
    b = np.argsort(a1[:, 1])
    a1 = a1[b]
    output += a1.tolist()
output = np.array(output)
np.save('../data_sorted.npy', np.array(output))

# Add the position of the ball two frame before to the data
output2 = np.zeros((output.shape[0], output.shape[1] + 2), dtype='int')
output2[:, :7] = output
output2[2:, 7:] = output[:-2, 5:]
np.save('../data.npy', np.array(output2))
