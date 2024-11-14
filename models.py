import torch.nn as nn
import torch
import config
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from qpth.qp import QPFunction
from model_utils import solver
from util import calculate_all_metrics

# Indices to make reading the code easier.

EGO_X_IDX = 0
EGO_Y_IDX = 1
EGO_THETA_IDX = 2
EGO_V_IDX = 3
OPP_X_IDX = 4
OPP_Y_IDX = 5
OPP_THETA_IDX = 6
OPP_V_IDX = 7
GOAL_DX_IDX = 8
GOAL_DY_IDX = 9

N_CL = 2
ANGULAR_VEL_IDX = 0
LINEAR_ACCEL_IDX = 1



class BarrierNet(nn.Module):
    # Input features: 8. [ego x, ego y, ego theta, ego v, opp x, opp y, opp theta, opp v]
    # Output controls: 2. [linear vel, angular vel].
    def __init__(self, model_definition, static_obstacles):
        super().__init__()
        self.model_definition = model_definition
        self.mean = torch.from_numpy(np.array(model_definition.input_mean)).to(config.device)
        self.std = torch.from_numpy(np.array(model_definition.input_std)).to(config.device)
        self.static_obstacles = static_obstacles

        # QP Parameters
        self.p1 = 0
        self.p2 = 0

        self.n_features = 8
        if model_definition.include_goal:
            self.n_features += 2
        
        self.fc1 = nn.Linear(self.n_features, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc22 = nn.Linear(model_definition.nHidden1, model_definition.nHidden22).double()
        self.fc23 = nn.Linear(model_definition.nHidden1, model_definition.nHidden23).double()
        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()
        self.fc32 = nn.Linear(model_definition.nHidden22, N_CL).double()
        self.fc33 = nn.Linear(model_definition.nHidden23, 1).double()

        self.s0 = Parameter(torch.ones(1).cuda()).to(config.device)
        self.s1 = Parameter(torch.ones(1).cuda()).to(config.device)

    

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        
        x21 = F.relu(self.fc21(x))
        x22 = F.relu(self.fc22(x))
        x23 = F.relu(self.fc23(x))
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        x33 = self.fc33(x23)
        x33 = 4*nn.Sigmoid()(x33)  # ensure CBF parameters are positive
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, x33, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, x33, sgn, nBatch):
        px = x0[:,EGO_X_IDX]
        py = x0[:,EGO_Y_IDX]
        theta = x0[:,EGO_THETA_IDX]
        v = x0[:,EGO_V_IDX]
        # dx = x0[:,GOAL_DX_IDX]
        # dy = x0[:,GOAL_DY_IDX]
        Q = Variable(torch.eye(N_CL))
        Q = Q.unsqueeze(0).expand(nBatch, N_CL, N_CL).to(config.device)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # obstacles = self.static_obstacles[:5].copy()
        obstacles = self.static_obstacles.copy()
        # obstacles = []
        obstacles.append((x0[:,OPP_X_IDX], x0[:,OPP_Y_IDX], config.agent_radius))

        G = []
        h = []
        for obs_x, obs_y, r in obstacles:
            R = config.agent_radius + r + config.safety_dist
            dx = (px - obs_x)
            dy = (py - obs_y)
            obs_v = x0[:, OPP_V_IDX]
            obs_theta = x0[:, OPP_THETA_IDX]
            obs_sin_theta = torch.sin(obs_theta)
            obs_cos_theta = torch.cos(obs_theta)

            barrier = dx**2 + dy**2 - R**2
            barrier_dot = 2*dx*v*cos_theta + 2*dy*v*sin_theta
            # barrier_dot = 2*dx*(v*cos_theta - obs_v*obs_cos_theta) + 2*dy*(v*sin_theta - obs_v*obs_sin_theta)
            Lf2b = 2*v**2
            LgLfbu1 = torch.reshape(-2*dx*v*sin_theta + 2*dy*v*cos_theta, (nBatch, 1)) 
            LgLfbu2 = torch.reshape(2*dx*cos_theta + 2*dy*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            obs_h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)
        
        print(len(G), len(h))
        print(G[0].shape, h[0].shape)
        # print(1/0)

        # Add in liveness CBF
        if config.smg_barriernet:
            G_lims, h_lims = [], []
            for i in range(len(x0)):
                lim_G = Variable(torch.tensor([0.0, 1.0]))
                lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                lim_h = Variable(torch.tensor([config.accel_limit])).to(config.device) + self.s0
                lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                G_lims.append(lim_G)
                h_lims.append(lim_h)

            G_lims = torch.cat(G_lims)
            h_lims = torch.cat(h_lims)
            G.append(G_lims)
            h.append(h_lims)

            G_lims, h_lims = [], []
            for i in range(len(x0)):
                lim_G = Variable(torch.tensor([0.0, -1.0]))
                lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                lim_h = Variable(torch.tensor([0.0])).to(config.device) + self.s1
                lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                G_lims.append(lim_G)
                h_lims.append(lim_h)

            G_lims = torch.cat(G_lims)
            h_lims = torch.cat(h_lims)
            G.append(G_lims)
            h.append(h_lims)


            # G_live, h_live = [], []
            # for i in range(len(x0)):
            #     ego_state = np.array([px[i].item(), py[i].item(), theta[i].item(), v[i].item()])
            #     opp_state = np.array([x0[i][OPP_X_IDX].item(), x0[i][OPP_Y_IDX].item(), x0[i][OPP_THETA_IDX].item(), x0[i][OPP_V_IDX].item()])
            #     l, _, _, _, intersecting = calculate_all_metrics(ego_state, opp_state)
            #     if False:
            #     # if l < config.liveness_threshold and intersecting:
            #         print("USING LIVENESS FILTER!!! FOR NOW FORCING IT TO SLOW DOWN", i)
            #         ego_v <= opp_v * smth
            #         acceleration_output <= p1(z)(opp_v - 3 * ego_v)
            #         # opp_v - 3 * ego_v >= 0.0
            #         # b(x) = opp_v - 3 * ego_v
            #         # u(t) <= p1(z)(opp_v - 3 * ego_v)
            #         barrier = x0[i][OPP_V_IDX] - config.zeta * v[i]
            #         live_G = Variable(torch.tensor([0.0, 1.0])).to(config.device)
            #         live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
            #         live_h = torch.reshape((x33[i,0])*barrier, (1, 1)).to(config.device)

            #         G_live.append(live_G)
            #         h_live.append(live_h)
            #     else:
            #         # live_G = Variable(torch.Tensor()).to(config.device)
            #         # live_G = Variable(torch.tensor([0.0, 0.0])).to(config.device)
            #         live_G = Variable(torch.tensor([0.0, 1.0]))
            #         live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
            #         live_h = Variable(torch.tensor([config.accel_limit * 2.0]))
            #         # live_h = Variable(torch.Tensor()).to(config.device)
            #         live_h = torch.reshape(live_h, (1, 1)).to(config.device)
            #         G_live.append(live_G)
            #         h_live.append(live_h)

            # G_live = torch.cat(G_live)
            # h_live = torch.cat(h_live)
            # print(G_live.shape, h_live.shape)
            # G.append(G_live)
            # h.append(h_live)


        
        print(len(G), len(h))
        
        G = torch.cat(G, dim=1).to(config.device)
        h = torch.cat(h, dim=1).to(config.device)
        # assert(G.shape == (nBatch, len(obstacles) + config.smg_barriernet, N_CL))
        # assert(h.shape == (nBatch, len(obstacles) + config.smg_barriernet))
        e = Variable(torch.Tensor()).to(config.device)
        
        # label_std, label_mean = np.array(self.model_definition.label_std), np.array(self.model_definition.label_mean)
        # print("Reference controls:", x31.cpu() * label_std + label_mean)

        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            print(x31[0].cpu())
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x



class BarrierNetDOpp(nn.Module):
    # Input features: 8. [ego x, ego y, ego theta, ego v, opp dx, opp dy, opp dtheta, opp dv]
    # Output controls: 2. [linear vel, angular vel].
    def __init__(self, model_definition, static_obstacles):
        super().__init__()
        self.model_definition = model_definition
        self.mean = torch.from_numpy(np.array(model_definition.input_mean)).to(config.device)
        self.std = torch.from_numpy(np.array(model_definition.input_std)).to(config.device)
        self.static_obstacles = static_obstacles

        # QP Parameters
        self.p1 = 0
        self.p2 = 0

        self.n_features = 8
        
        self.fc1 = nn.Linear(self.n_features, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc22 = nn.Linear(model_definition.nHidden1, model_definition.nHidden22).double()
        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()
        self.fc32 = nn.Linear(model_definition.nHidden22, N_CL).double()
    

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        
        x21 = F.relu(self.fc21(x))
        x22 = F.relu(self.fc22(x))
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, sgn, nBatch):
        px = x0[:,EGO_X_IDX]
        py = x0[:,EGO_Y_IDX]
        theta = x0[:,EGO_THETA_IDX]
        v = x0[:,EGO_V_IDX]
        Q = Variable(torch.eye(N_CL))
        Q = Q.unsqueeze(0).expand(nBatch, N_CL, N_CL).to(config.device)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        obstacles = self.static_obstacles.copy()
        obstacles.append((x0[:,OPP_X_IDX], x0[:,OPP_Y_IDX], config.agent_radius))

        G = []
        h = []
        for i, (obs_x, obs_y, r) in enumerate(obstacles):
            R = config.agent_radius + r + config.safety_dist
            dx = (px - obs_x) if i != len(obstacles) - 1 else obs_x
            dy = (py - obs_y) if i != len(obstacles) - 1 else obs_y
            barrier = dx**2 + dy**2 - R**2
            barrier_dot = 2*dx*v*cos_theta + 2*dy*v*sin_theta
            Lf2b = 2*v**2
            LgLfbu1 = torch.reshape(-2*dx*v*sin_theta + 2*dy*v*cos_theta, (nBatch, 1)) 
            LgLfbu2 = torch.reshape(2*dx*cos_theta + 2*dy*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            obs_h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)
        
        print(len(G), len(h))
        print(G[0].shape, h[0].shape)
        # print(1/0)
        
        G = torch.cat(G, dim=1).to(config.device)
        h = torch.cat(h, dim=1).to(config.device)
        assert(G.shape == (nBatch, len(obstacles), N_CL))
        assert(h.shape == (nBatch, len(obstacles)))
        e = Variable(torch.Tensor()).to(config.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x




class FCNet(nn.Module):
    def __init__(self, model_definition):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()


    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x21 = F.relu(self.fc21(x))
        x31 = self.fc31(x21)
        
        return x31
