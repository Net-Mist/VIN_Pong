import numpy as np

a = np.load('../dataset.npy')
b = np.argsort(a[:, 0])
a = a[b]

output = []
for i in range(1, 9):
    a1 = np.array([l for l in a if l[0] == i])
    b = np.argsort(a1[:, 1])
    a1 = a1[b]
    output += a1.tolist()

output = np.array(output)
print(output)
print(output.shape)
output2 = np.zeros((output.shape[0], output.shape[1]+2), dtype='int')
output2[:, :7] = output

output2[2:, 7:] = output[:-2, 5:]
print(output2)