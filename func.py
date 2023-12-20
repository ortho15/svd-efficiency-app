import numpy as np
import math
from PIL import Image

def jpg_to_RGB(file):
    img = Image.open(file)
    R = np.array(img)[:,:,0]
    G = np.array(img)[:,:,1]
    B = np.array(img)[:,:,2]

    R = np.array(R, dtype=np.float64)
    G = np.array(G, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    return R,G,B

def eigens(R,G,B):
    three = {
            'R': R,
            'G': G,
            'B': B
            }
    cache_SVt = {}

    for color, A in three.items():
        AtA = np.dot(A.T, A)
        assert np.allclose(AtA.T, AtA)
        eigenValues, eigenVectors = np.linalg.eigh(AtA)
        idx = eigenValues.argsort()[::-1] # argsort produces ascending index, reversed to be descending

        eigenValuesSorted = eigenValues[idx] # now eigens are descending
        Singulars = [math.sqrt(x) for x in eigenValuesSorted if x > 0] # throw away negative eigenvalues as they don't matter

        V = eigenVectors[:,idx] # sorts the eigen vector columns with respect to the descending eigenvalues, keeping the columns of eigens
        
        cache_SVt[color] = (Singulars, V.T)
    
    Us, S, Vt = np.linalg.svd(three['R'])
    print("Vt:", Vt[np.ix_([0,1,2,3,4,5],[0])])
    print("cache_SVt", cache_SVt['R'][1].T[np.ix_([0,1,2,3,4,5],[0])])


    return cache_SVt # they are singular values and associated V transposed

def compress(r, three, cache_SVt):
    m = int(three['R'].shape[0])
    n = int(three['R'].shape[1])
    # print(m)
    U = {
        'R': np.zeros((m, m), dtype=np.float64),
        'G': np.zeros((m, m), dtype=np.float64),
        'B': np.zeros((m, m), dtype=np.float64),
    }
    for color in three.keys():
        A = three[color]
        for i in range(m):
            try:
                sing = cache_SVt[color][0][i] # gives the nth singular values
                v = cache_SVt[color][1].T[:,[i]] # gives a nth column vector of Vt
            except IndexError:
                break # break the creation of u_n since singular values are unavailable / ~0
            col = (1/sing) * np.dot(A, v)
            U[color][:,[i]] = col

    Sig = {
        'R': np.zeros((m,n), dtype=np.float64),
        'G': np.zeros((m,n), dtype=np.float64),
        'B': np.zeros((m,n), dtype=np.float64),
    }
    for color, A in three.items():
        for i in range(r):
            try:
                Sig[color][i, i] = cache_SVt[color][0][i]
            except IndexError:
                break
    
    reduced = {
        "R": np.empty((m,n), dtype=np.float64),
        "G": np.empty((m,n), dtype=np.float64),
        "B": np.empty((m,n), dtype=np.float64)
    }
    for color in three.keys():
        reduced[color] = np.linalg.multi_dot([U[color], Sig[color], cache_SVt[color][1]])

    matrix_to_img(reduced['R'], reduced['G'], reduced['B'])

def matrix_to_img(R,G,B):
    full = np.empty((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    full[:, :, 0] = R
    full[:, :, 1] = G
    full[:, :, 2] = B
    img = Image.fromarray(full)
    img.save('./assets/reduced.png')
    return './assets/reduced.png'
