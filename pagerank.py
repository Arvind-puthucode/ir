import numpy as np
input_mat = [
  [1, 1, 1],
  [1, 0, 1],
  [0, 1, 1],
]

A = np.matrix(input_mat)
col_sums = np.sum(A, axis=1).reshape(-1, 1)
A = A/col_sums
A = A.transpose()

p = np.matrix([1/len(input_mat)] * len(input_mat)).reshape(-1, 1)
iter = 0

alpha = 0.5
A = (1-alpha)*A
A = A + (alpha/(len(input_mat)))

while True:

  print(f'p{iter}', p, sep='\n')

  new_p = np.matmul(A, p)
  new_p = new_p/sum(new_p)

  if max(abs(new_p - p)) < 0.01:
    break

  p = new_p.copy()
  iter += 1