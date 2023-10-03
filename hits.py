import numpy as np

input_mat = [
  
  [0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 1, 1, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 1, 0, 0]
]

A = np.matrix(input_mat)
A_T = A.transpose()

a = np.matrix([1] * len(input_mat)).reshape(-1, 1)
h = np.matrix([1] * len(input_mat)).reshape(-1, 1)

a = a/sum(a)
h = h/sum(h)

iter = 0

while True:
  print(f'a{iter}', a, sep='\n')
  print(f'h{iter}', h, sep='\n', end='\n')

  h_new = np.matmul(A, a)
  h_new = h_new/sum(h_new)

  a_new = np.matmul(A_T, h)
  a_new = a_new/sum(a_new)

  h_diff = max(abs(h_new - h))
  a_diff = max(abs(a_new - a))

  if h_diff < 0.01 and a_diff < 0.01:
    break

  a = a_new.copy()
  h = h_new.copy()
  iter += 1