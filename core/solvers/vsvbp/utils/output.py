import numpy as np
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE

def output_x_and_c(data, solver, c, x_jr):
    c_matrix = np.empty(shape=(len(data.functions), len(data.nodes)))
    for j in range(len(data.nodes)):
        for f in range(len(data.functions)):
            c_matrix[f][j] = solver.Value(c[f, j]) if solver else c[f][j]
    print("---------------C_MATRIX [F,N]----------------")
    print(c_matrix)
    
    # Fill x matrix
    mat_mul = np.dot(data.req_distribution,np.transpose(x_jr))
    x_matrix = np.empty(shape=(len(data.nodes),len(data.functions),len(data.nodes)))
    for i in range(len(data.nodes)):
        for j in range(len(data.nodes)):
            for f in range(len(data.functions)):
                if sum(mat_mul[f])==0:
                    x_matrix[i][f][j] = 0
                else:
                    x_matrix[i][f][j] = mat_mul[f][j]/sum(mat_mul[f])
                if data.req_distribution.sum(axis=1)[f]==0:
                    x_matrix[i][f][j]=c_matrix[f][j]/c_matrix.sum(axis=1)[f]

    print("---------------X_MATRIX [FxN]----------------")
    print(x_matrix[0])
    return x_matrix, c_matrix

def output_xjr(data, solver, status, x, c, y):
    x_jr = np.zeros([int(len(data.nodes)),int(data.requests_received)])

    if status == OPTIMAL or status == FEASIBLE:
        print('SOLUTION:')
        for r in range(data.requests_received):
            for j in range(len(data.nodes)):
                if int(solver.Value(x[j,r])) == 1:
                    print(f'x[{j},{r}]: Request {r} has been allocated to node {j}')
                    x_jr[j][r]=1
        
        print('----------------------------------------------------------------------')
        for f in range(len(data.functions)):
            for j in range(len(data.nodes)):
                if int(solver.Value(c[f,j])) == 1:
                    print(f'c[{f},{j}]: Function {f} has been deployed on node {j}')

        print('----------------------------------------------------------------------')
        for j in range(len(data.nodes)):
            if int(solver.Value(y[j])) == 1:
                    print(f'y[{j}]: Node {j} is used') 
    return x_jr