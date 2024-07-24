import random
import numpy as np
import geopy
import rvo2


def rvoAlgorithm(data):
    pref_vel = 5 / (111.1 * 3600)
    sim = rvo2.PyRVOSimulator(1,  # float timeStep
                              1.5,  # float MAX neighborDist
                              data.num_users,  # size_t maxNeighbors
                              1,  # float timeHorizon --->  response time in the presence of other agents
                              2,  # float timeHorizonObst ---> response time in the presence of obstacles
                              1,  # float radius ---> of agents
                              pref_vel)  # float maxSpeed ---> of agents
    # tuple velocity=(0, 0)

    agents = []
    for u in range(data.num_users):
        user_latitude = data.user_coords[u, 0]
        user_longitude = data.user_coords[u, 1]
        agents.append(sim.addAgent((user_latitude, user_longitude)))

    circle = [data.D[0], data.D[1], 0.05 / 111.1]
    xc = circle[0]
    yc = circle[1]
    r = circle[2]

    sim.addObstacle([(xc + r, yc + r), (xc + r, yc - r), (xc - r, yc - r), (xc - r, yc + r)])
    sim.processObstacles()

    for a in range(len(agents)):
        sim.setAgentPrefVelocity(agents[a], (random.uniform(-pref_vel, pref_vel), random.uniform(-pref_vel, pref_vel)))

    positions_T = []
    for _ in range(data.T):
        sim.doStep()
        positions = [sim.getAgentPosition(agent_no) for agent_no in (agents)]
        positions_T.append(positions)

    return positions_T


def du_dt_function(data, time, positions):
    du_Dt = []  # Distance between danger and user
    for j in range(data.num_users):
        user_latitude = positions[time][j][0]
        user_longitude = positions[time][j][1]
        user_coordinates = (user_latitude, user_longitude)
        dist_geoDanger = geopy.distance.geodesic(user_coordinates, data.D).km
        du_Dt.append(dist_geoDanger)
    return du_Dt


def criticality(data, time, du_Dt):
    CR = []
    CR_requests = np.zeros(data.requests_received)

    # emj: emotional value		
    em_t = np.zeros(data.num_users)
    for j in range(data.num_users):
        if du_Dt[j] < (data.D_rad + data.U_per[j]):
            eq = ((time - data.T_1) / data.T) * (1 - (du_Dt[j] / (data.D_rad + data.U_per[j]))) * data.lambd * data.nej[
                j]
            em_t[j] = eq
        else:
            em_t[j] = 0
    # scrj(t) = sej + emj (t) (subjective criticality equation)		
    scr = np.zeros(data.num_users)
    for j in range(data.num_users):
        scr[j] = (data.se_j[j] + em_t[j])
    ocr_j = np.zeros(data.num_users)  # ocr objective criticality of individual uj at time step t
    nDiv = 5  # Number of partitions of the coverage area
    circles = data.D_rad / nDiv  # Radius of each circular partition
    covCircles = []  # Distance from the center
    ri = [0.5, 0.4, 0.3, 0.2, 0.1]
    for p in range(1, nDiv + 1):
        covCircles.append(circles * p)
    for j in range(data.num_users):
        if du_Dt[j] < covCircles[0]:
            ocr_j[j] = ri[0]
        elif du_Dt[j] < covCircles[1] and du_Dt[j] >= covCircles[0]:
            ocr_j[j] = ri[1]
        elif du_Dt[j] < covCircles[2] and du_Dt[j] >= covCircles[1]:
            ocr_j[j] = ri[2]
        elif du_Dt[j] < covCircles[3] and du_Dt[j] >= covCircles[2]:
            ocr_j[j] = ri[3]
        elif du_Dt[j] < covCircles[4] and du_Dt[j] >= covCircles[3]:
            ocr_j[j] = ri[4]
        else:
            ocr_j[j] = 0
        # Criticality equation
        L1 = 0.95
        L2 = 1  # weight lambda 2
        # Criticality
        cr = (L1 * scr[j]) + (L2 * ocr_j[j])
        CR.append(cr)

    for r in range(data.requests_received):
        for u in range(data.num_users):
            if data.req_by_user[u][r] == 1:
                CR_requests[r] = CR[u]

    CR_index = np.argsort(CR_requests, kind='stable')
    return CR_index, CR_requests
