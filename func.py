import numpy as np
import math
from PIL import Image

cache_eigens = {}

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
    global cache_eigens
    three = {
            'R': R,
            'G': G,
            'B': B
            }
    for color, A in three.items():
        AtA = np.dot(A.T, A)
        assert np.allclose(AtA.T, AtA)
        eigenValues, eigenVectors = np.linalg.eigh(AtA)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        eigenValues = [math.sqrt(x) for x in eigenValues if x > 0]
        # eigenValues = eigenValues + [0] * (A.shape[0] - len(eigenValues))
        
        cache_eigens[color] = (eigenValues, eigenVectors)

    return cache_eigens # they are singular values now

def compress(r, three):
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
            # print(cache_eigens[color][1][[i]].T.shape)
            # print(f"vndim: {v.ndim}")
            try:
                sing = cache_eigens[color][0][i]
                v = cache_eigens[color][1][[i]].T
                if i == 0:
                    print(f"sing: {sing}")
            except IndexError:
                break
            col = (1/sing) * np.dot(A, v)
            if i == 0:
                    print(f"col:\n{col[0]}")
            U[color][:,i] = np.squeeze(col)
            # print(U[color][:,0])
        # print(f"U color: {color} is done!")

    m
    Sig = {
        'R': np.zeros((m,n), dtype=np.float64),
        'G': np.zeros((m,n), dtype=np.float64),
        'B': np.zeros((m,n), dtype=np.float64),
    }
    for color, A in three.items():
        for i in range(r):
            try:
                Sig[color][i, i] = cache_eigens[color][0][i]
            except IndexError:
                break
        # print(f"Sigma {color} is done!")
    
    reduced = {
        "R": np.empty((m,n), dtype=np.float64),
        "G": np.empty((m,n), dtype=np.float64),
        "B": np.empty((m,n), dtype=np.float64)
    }
    for color in three.keys():
        # print("U", U[color].shape, U[color].ndim)
        # print("Sig", Sig[color].shape, Sig[color].ndim)
        # print("V", cache_eigens[color][1].T.shape, cache_eigens[color][1].T.ndim)
        reduced[color] = np.linalg.multi_dot([U[color], Sig[color], cache_eigens[color][1].T])
        # print(f"finished dot color: {color}!")

    # print("Reduced R", reduced['R'].shape)
    # print("Reduced G", reduced['G'].shape)
    # print("Reduced B", reduced['B'].shape)

    # reduced = {
    #     "R": None,
    #     "G": None,
    #     "B": None
    # }
    Us, S, Vt = np.linalg.svd(three['R'])
    print("their", S[:2])
    print("mine", cache_eigens['R'][0][:2])
    print("mine sigma", Sig['R'][0,0])
    print(Us[:,0][0])
    print(U['R'][:,0][0])
    print("Vt", Vt[:,0][0:2])
    print(cache_eigens[color][1][i].T[0:2])
        # reduced[color] = np.linalg.multi_dot([U, Sig, Vt])

    # matrix_to_img(reduced['R'], reduced['G'], reduced['B'])
    matrix_to_img(reduced['R'], reduced['G'], reduced['B'])

def matrix_to_img(R,G,B):
    full = np.empty((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    full[:, :, 0] = R
    full[:, :, 1] = G
    full[:, :, 2] = B
    img = Image.fromarray(full)
    img.save('./assets/reduced.png')
    # return './assets/reduced.png'