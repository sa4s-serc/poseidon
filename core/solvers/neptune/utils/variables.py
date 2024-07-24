import numpy as np

#TODO: check var name 
def init_x(data, solver, x):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            for j in range(len(data.nodes)):
                x[i, f, j] = solver.NumVar(0, solver.infinity(), f"x[{i}][{f}][{j}]")
    

def init_c(data, solver, c):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            c[f, i] = solver.BoolVar(f"c[{f}][{i}]")
            
        
def init_n(data, solver, n):
    for i in range(len(data.nodes)):
        n[i] = solver.BoolVar(f"n[{i}]")

def init_moved_from(data, solver, moved_from):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            moved_from[f, i] = solver.BoolVar(f"moved_from[{f}][{i}]")

def init_moved_to(data, solver, moved_to):
    for f in range(len(data.functions)):
        for i in range(len(data.nodes)):
            moved_to[f, i] = solver.BoolVar(f"moved_to[{f}][{i}]")

def init_allocated(data, solver):
    return solver.IntVar(-np.ma.size(data.old_allocations_matrix), 0, "allocated")

def init_deallocated(data, solver):
    return  solver.IntVar(-np.ma.size(data.old_allocations_matrix), 0, "deallocated")