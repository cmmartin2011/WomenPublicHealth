from DPCA import *
import numpy as np
import time
from scipy.sparse.linalg import svds
from Preprocessing import *
import xlwings as xw
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.lines as mlines

def svd_left(matrix, n):
    U, _, _ = np.linalg.svd(matrix, full_matrices = True)
    return U[:, :n]

# Tucker3 Model: This function calculates orthogonal components in A, B and C (TuckALS3).
def T3(X, P, Q, R, ALSMaxIter):
    start_time = time.time()
    XA = frontal_cuts(X)
    XB = horizontal_cuts(X)
    XC = vertical_cuts(X)
    B = svd_left(XB, Q)
    C = svd_left(XC, R)
    for i in range(ALSMaxIter):
        YA = XA @ np.kron(C, B)
        A = svd_left(YA, P)
        YB = XB @ np.kron(C, A)
        B = svd_left(YB, Q)
        YC = XC @ np.kron(B, A)
        C = svd_left(YC, R)
    YA = XA @ np.kron(C, B)
    GA = A.T @ YA
    CoreG = GA.reshape(P, Q, R)  
    VarData = np.sum(X ** 2)
    EleAprox = np.einsum("ip,jq,kr,pqr -> ijk", A, B, C, CoreG)
    VarModel = np.sum(EleAprox ** 2)
    pFit = round((VarModel / VarData) * 100, 4)
    execution_time = round(time.time() - start_time, 2)
    return A, B, C, GA, pFit, execution_time

# This function builds all the Tucker3 models that are used to construct the convex hull.
def TableFits(X, P_max, Q_max, R_max, ALSMaxIter):
    results = []
    models = 0
    for P in range(1, P_max + 1):
        for Q in range(1, Q_max + 1):
            for R in range(1, R_max + 1):
                models+=1
                _, _, _, _, pFit, _ = T3(X, P, Q, R, ALSMaxIter)
                print(f"OK: Model={models}, P={P}, Q={Q}, R={R}, Fit={pFit}")  
                S = P + Q + R
                results.append([models, P, Q, R, S, pFit])
    Table = np.array(results)
    wb = xw.Book()
    sht = wb.sheets[0]
    sht.name = "Table of Models"
    headers = ["Model", "P", "Q", "R", "S", "Fit(%)"]
    sht.range("A1").value = headers
    sht.range("A2").value = Table.tolist()            
    return Table

# This function plots the convex hull.
def plot_ch(models):
    X = models[:,4]
    Y = models[:,5]
    points = np.column_stack((X, Y))
    hull = ConvexHull(points)
    plt.figure(figsize=(8,6))
    hull_indices = set(hull.vertices)
    all_indices = set(range(len(points)))
    interior_indices = all_indices - hull_indices
    plt.scatter(points[list(interior_indices),0], points[list(interior_indices),1],
                color = 'lightgray', s = 30, zorder = 2, label = 'Interior models')
    plt.scatter(points[list(hull_indices),0], points[list(hull_indices),1],
                color = 'lightcoral', s = 30, zorder = 3, label = 'Hull models')
    for simplex in hull.simplices:
        plt.plot(points[simplex,0], points[simplex,1],
                 color = 'mediumaquamarine', linewidth = 1.2, zorder = 1)
    for i in hull.vertices:
        x, y = points[i]
        plt.text(x, y + 0.5, str(i + 1), color = 'blue', fontsize = 8, ha = 'center', zorder = 4)
    plt.xlabel("Sum of number of components")
    plt.ylabel("Fit(%)")
    gray_point = plt.Line2D([], [], color = 'lightgray', marker = 'o', linestyle = 'None',
                            markersize = 6, label = 'Interior models')
    red_point = plt.Line2D([], [], color = 'lightcoral', marker = 'o', linestyle = 'None',
                           markersize = 6, label = 'Hull models')
    green_line = mlines.Line2D([], [], color = 'mediumaquamarine', linewidth = 1.2, label = 'Convex hull')
    plt.legend(handles = [gray_point, red_point, green_line], loc = 'upper left')
    plt.show()

def svd(matrix, n):
    U, _, _ = svds(matrix, k = n)
    return U

def frontal_cuts(X):
    I, J, K = X.shape
    XA = np.zeros((I, J * K))
    NumCol = 0
    for k in range(K):
        for j in range(J):
            for i in range(I):
                XA[i, NumCol] = X[i, j, k]
            NumCol += 1
    return XA

def horizontal_cuts(X):
    I, J, K = X.shape
    XB = np.zeros((J, I * K))
    NumCol = 0
    for k in range(K):
        for i in range(I):
            for j in range(J):
                XB[j, NumCol] = X[i, j, k]
            NumCol += 1
    return XB

def vertical_cuts(X):
    I, J, K = X.shape
    XC = np.zeros((K, I * J))
    NumCol = 0
    for j in range(J):
        for i in range(I):
            for k in range(K):
                XC[k, NumCol] = X[i, j, k]
            NumCol += 1
    return XC

# This function loads the data into memory if the frontal slices of the tensor are positioned one to the right of the other.
def get_tensor_row(datamatrix, I, J, K):
    X = np.zeros((I, J, K))
    NumVar = 0
    for r in range(I):
        NumTable = 0
        for c in range(J * K):
            X[r, NumVar, NumTable] = datamatrix[r, c]
            NumVar += 1
            if NumVar == J:
                NumVar = 0
                NumTable += 1
    return X

# This function loads the data into memory if the frontal slices of the tensor are positioned one below the other.
def get_tensor_column(datamatrix, I, J, K):
    X = np.zeros((I, J, K))
    NumIndividuo = 0
    for c in range(J):
        NumTable = 0
        for r in range(I * K):
            X[NumIndividuo, c, NumTable] = datamatrix[r, c]
            NumIndividuo += 1
            if NumIndividuo == I:
                NumIndividuo = 0
                NumTable += 1
    return X

# Tucker3 Model: This function calculates disjoint components in A, B, and C.
def disjoint_T3_ABC(X, P, Q, R, ALSMaxIter, Tol):
    start_time = time.time()
    XA = frontal_cuts(X)
    XB = horizontal_cuts(X)
    XC = vertical_cuts(X)
    B = svd(XB, Q)
    C = svd(XC, R)
    for i in range(ALSMaxIter):
        YA = XA @ np.kron(C, B)
        A = svd(YA, P)
        YB = XB @ np.kron(C, A)
        B = svd(YB, Q)
        YC = XC @ np.kron(B, A)
        C = svd(YC, R)
    _, _, _, A, _ = VS(YA.T, P, Tol)
    _, _, _, B, _ = VS(YB.T, Q, Tol)
    _, _, _, C, _ = VS(YC.T, R, Tol)
    YA = XA @ np.kron(C, B)
    GA = A.T @ YA
    CoreG = GA.reshape(P, Q, R)
    VarData = np.sum(X ** 2)
    EleAprox = np.einsum("ip,jq,kr,pqr -> ijk", A, B, C, CoreG)
    VarModel = np.sum(EleAprox ** 2)
    pFit = round((VarModel / VarData) * 100, 4)
    execution_time = round(time.time() - start_time, 2)
    return A, B, C, GA, pFit, execution_time