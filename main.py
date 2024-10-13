"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
#x1 and x2 are time series states of agents 1 and 2 respectively
#n is the number of agents
#N is number of iterations for one time horizon
#controller[1] and controller[2] are the Game thereotic MPC controllers for agent 1 and 2 respectively

import numpy as np
from mpc_cbf import MPC
import config
from scenarios import DoorwayScenario, IntersectionScenario
from plotter import Plotter


# Add number of agents
n=2
N=50 #Number of iteration steps for each agent

controller=[None]*n #Control input intitialization
xf=np.zeros((n,3)) # initialization of final states
u = []
L=[]
u_proj = []
final_positions_both_agents=np.zeros((n*N,3)) # initialization of final states for both agents

x1=np.zeros((N,3)) # initialization of times series states of agent 1 
x2=np.zeros((N,3)) # initialization of times series states of agent 1 

xf_minus=np.zeros((n,3))

# Scenarios: "doorway" or "intersection"
scenario = DoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()

def main():
    c=0
    #Add all initial and goal positions of the agents here (Format: [x, y, theta])
    initial = scenario.initial
    goals = scenario.goals
    LL=[]
    first_time = True
    for j in range(N): # j denotes the j^th step in one time horizon
        for i in range(n):  # i is the i^th agent 

        # Define controller & run simulation for each agent i
            config.x0=initial[i,:]
            config.goal=goals[i,:]
            config.obs = scenario.obstacles.copy()

            # Ensures that other agents act as obstacles to agent i
            for k in range(n):
                if i!=k:
                    config.obs.append((initial[k,0], initial[k,1],0.08)) 

            #Liveliness "on" or "off" can be selected from here
            liveliness='on'

            #Initialization of MPC controller for the ith agent
            controller[i] = MPC(final_positions_both_agents,j,i,liveliness)

            #final_positions_both_agents stores the final positions of both agents
            #[1,:], [3,:], [5,:]... are final positions of agent 1
            #[2,:], [4,:], [6,:]... are final positions of agent 2
            final_positions_both_agents[c,:]=xf[i,:]
            lll=controller[i].run_simulation(final_positions_both_agents,j)
            LL.append(lll)
            #The position of agent i is propogated one time horizon ahead using the MPC controller
            x, uf, uf_proj, l =controller[i].run_simulation_to_get_final_condition(final_positions_both_agents,j,i,liveliness, first_time)
            xf[i,:] = x.ravel()            
            u.append(uf)
            u_proj.append(uf_proj)
            L.append(l)

            c += 1
   
        # Plots
        initial=xf #The final state is assigned to the initial state stack for future MPC

    #x1 and x2 are times series data of positions of agents 1 and 2 respectively
    for ll in range(N-1):
        x1[ll,:]=final_positions_both_agents[n*ll,:]
        x2[ll,:]=final_positions_both_agents[n*ll+1,:]
    
    # T=0.4
    # L=[]
    # # Computation of liveliness value L for agent 1
    # for i in range(len(x1)-2):
    #     vec1=((x2[i+1,0:2]-x2[i,0:2])-(x1[i+1,0:2]-x1[i,0:2]))/T
    #     vec2=x1[i,0:2]-x2[i,0:2]
    #     l=np.arccos(np.dot(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2)+0.001))
    #     l_sin=abs(np.arcsin(np.cross(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2)+0.001)))
    #     print(l, l_sin)
    #     L.append(l)

    # Discard the first element of both x1 and x2
    plotter.plot(scenario, x1[1:], x2[1:], u, u_proj, L)
       


if __name__ == '__main__':
    main()
