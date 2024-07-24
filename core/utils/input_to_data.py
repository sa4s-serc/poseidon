import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from .data import Data
from hurry.filesize import size

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

keys = [
    # "solver",
    "community",
    "namespace",
    "function_names",
    "function_memories",
    "gpu_function_names",
    "gpu_function_memories",
    "node_names",
    "node_memories",
    "node_cores",
    "gpu_node_names",
    "gpu_node_memories",
    "function_max_delays",
    "actual_cpu_allocations",
    "actual_gpu_allocations",
]

# Added ADAPTIVE AGENT option (PoseidonPPO)
solvers = ["NeptuneMinDelayAndUtilization", "NeptuneMinDelay", "NeptuneMinUtilization", "VSVBP", "Criticality", "CriticalityHeuristic", "MCF", "PoseidonPPO",]


def check_input(schedule_input):
    print(f"Checking scheduling input...")

    print(f"Checking that it contains all the required keys")
    for key in keys:
        assert key in schedule_input.keys(), f"Key `{key}` not in schedule input"

    # assert "type" in schedule_input["solver"]
    # assert schedule_input["solver"]["type"] in solvers
     
    functions = schedule_input.get('function_names', [])
    function_memories = schedule_input.get('function_memories', [])
    gpu_functions = schedule_input.get('gpu_function_names', [])
    gpu_function_memories = schedule_input.get('gpu_function_memories', [])
    print(f"Functions are: {functions}")
    print(f"GPU Functions are: {gpu_functions}")
    print(f"Function memories are: {[size(m) for m in function_memories]}")
    print(f"GPU Function memories are: {[size(m) for m in gpu_function_memories]}")

    print(f"Checking Function consistencies...")
    assert set(gpu_functions).issubset(set(functions))
    assert len(functions) == len(function_memories)
    assert len(gpu_functions) == len(gpu_function_memories)

    nodes = schedule_input.get('node_names', [])
    node_memories = schedule_input.get('node_memories', [])
    node_cores = schedule_input.get('node_cores', [])
    gpu_nodes = schedule_input.get('gpu_node_names', [])
    gpu_node_memories = schedule_input.get('gpu_node_memories', [])
    print(f"Nodes are: {nodes}")
    print(f"Nodes memories are: {[size(m) for m in node_memories]}")
    print(f"Nodes cores are: {node_cores}")
    print(f"GPU Nodes are: {gpu_nodes}")
    print(f"GPU Nodes memories are: {[size(m) for m in gpu_node_memories]}")

    print(f"Checking Nodes consistencies...")
    assert set(gpu_nodes).issubset(set(nodes))
    assert len(nodes) == len(node_memories)
    assert len(gpu_nodes) == len(gpu_node_memories)

    print("Everything seems consistent")

def data_to_solver_input(input, workload_coeff, with_db=True):
    aux_data = Data()
    setup_community_data(input, aux_data)
    setup_runtime_data(aux_data, input)
    create_mappings(aux_data)
    if with_db:
        update_data_from_db(aux_data)
    update_old_allocations(aux_data)


    data = Data(aux_data.nodes, aux_data.functions)
    data.node_memory_matrix = np.array(aux_data.node_memories)
    data.function_memory_matrix = np.array(aux_data.function_memories)
    data.node_delay_matrix = np.array(aux_data.node_delay_matrix)
    data.workload_matrix = np.array(aux_data.workload_on_source_matrix) * workload_coeff
    data.max_delay_matrix = np.array(aux_data.max_delay_matrix)
    data.response_time_matrix = np.array(aux_data.response_time_matrix)
    data.node_cores_matrix = np.array(aux_data.node_cores)
    data.cores_matrix = np.array(aux_data.cores_matrix)
    data.old_allocations_matrix = np.array(aux_data.old_cpu_allocations)
    data.core_per_req_matrix = np.array(aux_data.core_per_req_matrix)
    setup_budget_data(data)
    
    return data


def setup_community_data(input, data):
    data.community = input.get('community')
    data.namespace = input.get('namespace')
    data.functions = input.get('function_names', [])
    data.gpu_functions = input.get('gpu_function_names', [])
    data.gpu_functions_set = set(data.gpu_functions)
    data.gpu_functions_mask = np.array([f in data.gpu_functions_set for f in data.functions])
    data.nodes = input.get('node_names', [])
    data.gpu_nodes = input.get('gpu_node_names', [])
    data.gpu_nodes_set = set(data.gpu_nodes)
    data.gpu_nodes_mask = np.array([n in data.gpu_nodes_set for n in data.nodes])
    
    assert set(data.gpu_functions).issubset(set(data.functions))
    assert set(data.gpu_nodes).issubset(set(data.nodes))

    data.function_memories = input.get('function_memories')
    data.node_memories = input.get('node_memories')
    data.gpu_node_memories = input.get('gpu_node_memories')
    data.actual_cpu_allocations = input.get('actual_cpu_allocations')
    data.actual_gpu_allocations = input.get('actual_gpu_allocations')
    data.node_cores = input.get('node_cores')
    data.gpu_function_memories = input.get('gpu_function_memories')
    data.max_delay_matrix = [1000 for _ in range(len(data.function_memories))]


def setup_runtime_data(data, input):
    node_delay_matrix = input.get('node_delay_matrix', None)
    if node_delay_matrix:
        data.node_delay_matrix = node_delay_matrix
    else: 
        data.node_delay_matrix = [[1 if s != d else 0 for s in data.nodes] for d in data.nodes]
    data.gpu_node_delay_matrix = [[1 if s != d else 0 for s in data.nodes] for d in data.gpu_nodes]
    
    workload_on_source_matrix = input.get('workload_on_source_matrix', None)
    if len(workload_on_source_matrix):
         data.workload_on_source_matrix = np.array(workload_on_source_matrix)
    else:
        data.workload_on_source_matrix = np.array([[0 for _ in data.nodes] for _ in data.functions])
    
    workload_on_destination_matrix = input.get('workload_on_destination_matrix', None)
    if workload_on_destination_matrix:
         data.workload_on_destination_matrix = np.array(workload_on_destination_matrix)
    else:
        data.workload_on_destination_matrix = np.array([[0 for _ in data.nodes] for _ in data.functions])
    
    data.gpu_workload_on_destination_matrix = np.array([[0 for _ in data.gpu_nodes] for _ in data.gpu_functions])

    cores_matrix = input.get('cores_matrix', None)
    if cores_matrix:
        data.cores_matrix = cores_matrix
    else: 
        data.cores_matrix = np.array([[0 for _ in data.nodes] for _ in data.functions])
    data.response_time_matrix = [[0 for _ in data.nodes] for _ in data.functions]
    data.gpu_response_time_matrix = [[1 for _ in data.gpu_nodes] for _ in data.gpu_functions]
    data.old_cpu_allocations = np.array([[0 for _ in data.nodes] for _ in data.functions])
    data.old_gpu_allocations = np.array([[0 for _ in data.gpu_nodes] for _ in data.gpu_functions])
    data.core_per_req_matrix = np.array([[1 for _ in data.nodes] for _ in data.functions])

def setup_budget_data(data):
    
    data.node_costs = np.array([5 for _ in data.nodes])
    data.node_budget = 30

def create_mappings(data):
    data.node_map = {}
    data.func_map = {}
    data.gpu_node_map = {}
    data.gpu_func_map = {}
    for i, node in enumerate(data.nodes):
        data.node_map[node] = i
    for i, node in enumerate(data.gpu_nodes):
        data.gpu_node_map[node] = i
    for i, func in enumerate(data.functions):
        func = func.split("/")[1]
        data.func_map[func] = i
    for i, func in enumerate(data.gpu_functions):
        func = func.split("/")[1]
        data.gpu_func_map[func] = i


def update_data_from_db(data):
    username = "user"
    password = "password"
    database_host = "metrics-database.kube-system.svc.cluster.local"
    database_port = 5432
    postgres_str = (f"postgresql://{username}:{password}@{database_host}:{database_port}")
    interval = "'30 seconds'"
    cnx = create_engine(postgres_str)
    ar_df = pd.read_sql(
        sql=f"SELECT function, source, count(*) AS arrival_rate FROM metric WHERE timestamp > now() - INTERVAL {interval} AND namespace = '{data.namespace}' AND community = '{data.community}' GROUP BY function, source ",
        con=cnx)
    ard_df = pd.read_sql(
        sql=f"SELECT function, destination, gpu, count(*) AS arrival_rate FROM metric WHERE timestamp > now() - INTERVAL {interval} AND namespace = '{data.namespace}' AND community = '{data.community}' GROUP BY function, destination, gpu",
        con=cnx)
    rt_df = pd.read_sql(
        sql=f"SELECT function, destination, gpu, avg(latency) AS response_time FROM metric WHERE timestamp > now() - INTERVAL {interval} AND namespace = '{data.namespace}' AND community = '{data.community}' GROUP BY function, destination, gpu ",
        con=cnx)
    dl_df = pd.read_sql(
        sql=f"SELECT f,t,l FROM (SELECT from_node, to_node FROM ping GROUP BY from_node, to_node) as p1 INNER JOIN LATERAL (SELECT from_node as f, to_node as t, avg_latency as l FROM ping p2 WHERE p1.from_node = p2.from_node AND p1.to_node = p2.to_node ORDER BY timestamp DESC LIMIT 1) AS data ON true",
        con=cnx
    )
    cpu_df = pd.read_sql(
        sql=f"SELECT function, node, avg(cores) AS cores FROM resource WHERE timestamp > now() - INTERVAL {interval} AND namespace = '{data.namespace}' AND community = '{data.community}' GROUP BY function, node",
        con=cnx
    )

    print(f"ARRIVAL RATE SOURCE \n\n {ar_df}")
    print(f"ARRIVAL RATE DESTINATION \n\n {ard_df}")
    print(f"RESPONSE TIME \n\n {rt_df}")
    print(f"DELAYS \n\n {dl_df}")
    print(f"CPU CONSUMPTION \n\n {cpu_df}")
   
    for node, func, response_time, gpu in zip(rt_df['destination'], rt_df['function'], rt_df['response_time'],
                                              rt_df['gpu']):
        if gpu:
            data.gpu_response_time_matrix[data.gpu_func_map[func]][data.gpu_node_map[node]] = response_time
            pass
        else:
            data.response_time_matrix[data.func_map[func]][data.node_map[node]] = response_time

    for node, func, arrival_rate in zip(ar_df['source'], ar_df['function'], ar_df['arrival_rate']):
        data.workload_on_source_matrix[data.func_map[func]][data.node_map[node]] = arrival_rate

    for node, func, cores in zip(cpu_df['node'], cpu_df['function'], cpu_df['cores']):
        data.cores_matrix[data.func_map[func]][data.node_map[node]] = cores

    for node, func, response_time, gpu in zip(ard_df['destination'], ard_df['function'], ard_df['arrival_rate'],
                                              ard_df['gpu']):
        if gpu:
            data.gpu_workload_on_destination_matrix[data.gpu_node_map[node]][data.gpu_func_map[func]] = response_time
            pass
        else:
            data.workload_on_destination_matrix[data.func_map[func]][data.node_map[node]] = response_time

    for from_node, to_node, latency in zip(dl_df['f'], dl_df['t'], dl_df['l']):
        if from_node in data.node_map and to_node in data.node_map:
            data.node_delay_matrix[data.node_map[from_node]][data.node_map[to_node]] = latency
        if from_node in data.node_map and to_node in data.gpu_node_map:
            data.gpu_node_delay_matrix[data.gpu_node_map[from_node]][data.gpu_node_map[to_node]] = latency


def update_old_allocations(data):
    for function_key, x in data.actual_cpu_allocations.items():
        for node, ok in x.items():
            if x:
                func = function_key.split("/")[1]
                data.old_cpu_allocations[data.func_map[func]][data.node_map[node]] = ok

    data.core_per_req_matrix = np.nan_to_num(data.cores_matrix / data.workload_on_destination_matrix, nan=0)

    data.old_cpu_allocations = np.array(data.old_cpu_allocations, dtype=bool).astype(int)
    if data.old_cpu_allocations.sum() == 0:
        data.old_cpu_allocations = data.old_cpu_allocations + 1

    print("CPU allocations:")
    print(data.old_cpu_allocations)

    data.old_gpu_allocations = np.array(data.old_gpu_allocations, dtype=bool).astype(int)
    if data.old_gpu_allocations.sum() == 0:
        data.old_gpu_allocations = data.old_gpu_allocations + 1

    print("GPU allocations:")
    print(data.old_gpu_allocations)
