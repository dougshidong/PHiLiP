import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs



#filename = "precond_matrix.dat"
filename = "matrix.dat"
n = sum(1 for line in open(filename))
print(n)

A = csr_matrix((n,n), dtype=np.float64)
#A = lil_matrix((n,n), dtype=np.float64)
#A = coo_matrix((n,n), dtype=np.float64)
print(A.get_shape())
with open(filename) as infile:
    r = 0
    temp = lil_matrix((n,n), dtype=np.float64)
    for line in infile:
        if (r % 100 == 0):
            print(r, " out of ", n)

        #jrow = []
        #jcol = []
        #jval = []
        c = 0
        for word in line.split():
            v = float(word)
            #if (abs(v) > 1e-14):
            if (abs(v) > 0):
                #jrow.append(r);
                #jcol.append(c);
                #jval.append(v);
                #print(r,c,v)
                temp[r,c] = v

            c = c+1
        #jrow = np.array(row)
        #jcol = np.array(col)
        #jval = np.array(val)
        #temp = csr_matrix((val, (row, col)), shape = (n,n))
        #A = A + temp
        r = r + 1
        if r % 200 == 0:
            temp2 = temp.tocsr()
            A = A + temp2
            temp = lil_matrix((n,n), dtype=np.float64)
        #if r == 10:
            #break
    temp2 = temp.tocsr()
    A = A + temp2

print(A.getnnz())
print(A.get_shape())
#$eigvals = eigs(A, k=10, which='LM',return_eigenvectors=False)
#$print(eigvals)

eigvals = eigs(A, k=10, which='SM',return_eigenvectors=False)
print(eigvals)
