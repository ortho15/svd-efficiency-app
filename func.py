import numpy as np
import math
from PIL import Image

cache_eigens = {}

def jpg_to_RGB(file):
    img = Image.open(file)
    R = np.array(img)[:,:,0]
    G = np.array(img)[:,:,1]
    B = np.array(img)[:,:,2]

    return R,G,B

def eigens(R,G,B):
    global cache_eigens
    three = {
            'R': R,
            'G': G,
            'B': B
            }
    for name, A in three.items():
        AtA = np.dot(A.T, A)
        if not np.allclose(AtA, AtA.T, rtol=1e-05, atol=1e-08):
            return "AtA is not symmetric!"
        eigenValues, eigenVectors = np.linalg.eigh(AtA)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        neg_idx = next((index for index, value in enumerate(eigenValues) if value < 0), None)
        eigenValues = eigenValues[:neg_idx]
        # eigenVectors = eigenVectors[:neg_idx]
        eigenValues = [math.sqrt(x) for x in eigenValues]
        cache_eigens[name] = (eigenValues, eigenVectors)

    return cache_eigens # they are singular values now

def compress(r, three):
    U = {
        'R': np.zeros((three['R'].shape[0], three['R'].shape[0])),
        'G': np.zeros((three['R'].shape[0], three['R'].shape[0])),
        'B': np.zeros((three['R'].shape[0], three['R'].shape[0])),
    }
    for color in three.keys():
        for i in range(r):
            # print(cache_eigens[color][1][[i]].T.shape)
            v = cache_eigens[color][1][[i]].T
            # print(f"vndim: {v.ndim}")
            A = three[color]
            sing = cache_eigens[color][0][i]
            col = (1/sing) * np.dot(A, v)
            # print(col)
            U[color][:,i] = np.squeeze(col)
            # print(U[color][:,0])
        print(f"U color: {color} is done!")

    Sig = {
        'R': np.zeros((three['R'].shape[0], three['R'].shape[1])),
        'G': np.zeros((three['R'].shape[0], three['R'].shape[1])),
        'B': np.zeros((three['R'].shape[0], three['R'].shape[1])),
    }
    for color, A in three.items():
        for i in range(r):
            Sig[color][i, i] = cache_eigens[color][0][i]
        print(f"Sigma {color} is done!")
    
    reduced = {
        "R": None,
        "G": None,
        "B": None
    }
    for color in three.keys():
        print("U", U[color].shape, U[color].ndim)
        print("Sig", Sig[color].shape, Sig[color].ndim)
        print("V", cache_eigens[color][1].T.shape, cache_eigens[color][1].T.ndim)
        reduced[color] = np.dot(U[color], Sig[color], cache_eigens[color][1].T)
        print(f"finished dot color: {color}!")

    matrix_to_img(reduced['R'], reduced['G'], reduced['B'])

def matrix_to_img(R,G,B):
    full = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    full[:, :, 0] = R
    full[:, :, 1] = G
    full[:, :, 2] = B
    img = Image.fromarray(full)
    img.save('./assets/reduced.png')
    return './assets/reduced.png'