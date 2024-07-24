import numpy as np

def prepare_order_requests(data):
    # Matrix that assignes a function memory to each request [F x N]
    m_request = np.empty((100,int(10000)))
    for f in range(len(data.functions)):
        for r in range (data.requests_received):
            m_request[f][r] = data.function_memory_matrix[f]*data.req_distribution[f][r]
    # Sort the requests by their memory requirement --- returns position of the [] where request is found
    m_index = []
    for r in range (data.requests_received):
        for f in range (len(data.functions)):
            if m_request[f][r]!=0:
                m_index.append(m_request[f][r])
    
    data.requests_index=np.argsort(m_index,kind='stable')