# Generates sparse matrix with 4 non-zeros per row of size argv[1] in CSR format

import sys
import numpy as np
import random

def get_val():
    return np.random.normal(1, 1)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    n = int(sys.argv[1])
    nonzeros_per_row = 4
    assert n >= nonzeros_per_row
    values = [get_val() for _ in range(n * nonzeros_per_row)]
    rowIdx = [i * nonzeros_per_row for i in range(n + 1)]
    
    colIdx = []
    A = list(range(n))
    for _ in range(n):
        colIdx += random.sample(A, nonzeros_per_row)
    
    f = open(f"sparse_{n}", "w")
    f.write(f"{n} {n} {n * nonzeros_per_row} {nonzeros_per_row}\n")
    for v in values:
        f.write(f"{str(v)} ")
    f.write("\n")
    for v in rowIdx:
        f.write(f"{str(v)} ")
    f.write("\n")
    for v in colIdx:
        f.write(f"{str(v)} ")
    f.close()
