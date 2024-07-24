def init_x(data, model, x):
    print(len(data.nodes))
    for j in range(len(data.nodes)):
        for r in range(data.requests_received):
                x[j, r] = model.NewBoolVar(f'c[{j}][{r}]')

def init_c(data, model, c):
    for f in range(len(data.functions)):
        for j in range(len(data.nodes)):
            c[f, j] = model.NewBoolVar(f'c[{f}][{j}]')


def init_y(data, model, y):
    for j in range(len(data.nodes)): 
        y[j] = model.NewBoolVar(f'c[{j}]')