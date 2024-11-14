import torch.nn as nn
import torch
import config
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from qpth.qp import QPFunction
from model_utils import solver
from util import calculate_all_metrics, calculate_is_not_live_torch, get_ray_intersection_point

# Indices to make reading the code easier.

EGO_X_IDX = 0
EGO_Y_IDX = 1
EGO_THETA_IDX = 2
EGO_V_IDX = 3

OPP_X_OFFSET = 0
OPP_Y_OFFSET = 1
OPP_THETA_OFFSET = 2
OPP_V_OFFSET = 3

N_CL = 2
ANGULAR_VEL_IDX = 0
LINEAR_ACCEL_IDX = 1



class BarrierNet(nn.Module):
    # Input features: 8. [ego x, ego y, ego theta, ego v, opp x, opp y, opp theta, opp v]
    # Output controls: 2. [linear vel, angular vel].
    def __init__(self, model_definition):
        super().__init__()
        self.model_definition = model_definition
        self.input_mean = torch.from_numpy(np.array(model_definition.input_mean)).to(config.device)
        self.input_std = torch.from_numpy(np.array(model_definition.input_std)).to(config.device)
        self.output_mean_np = np.array(model_definition.label_mean)
        self.output_std_np = np.array(model_definition.label_std)
        self.output_mean = torch.from_numpy(self.output_mean_np).to(config.device)
        self.output_std = torch.from_numpy(self.output_std_np).to(config.device)

        self.zeta = 2.0

        print("NUM MODEL INPUTS:", model_definition.get_num_inputs())
        
        self.fc1 = nn.Linear(model_definition.get_num_inputs(), model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc22 = nn.Linear(model_definition.nHidden1, model_definition.nHidden22).double()
        if self.model_definition.separate_penalty_for_opp:
            self.fc23 = nn.Linear(model_definition.nHidden1, model_definition.nHidden23).double()
        if self.model_definition.add_liveness_filter:
            self.fc24 = nn.Linear(model_definition.nHidden1, model_definition.nHidden24).double()

        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()
        self.fc32 = nn.Linear(model_definition.nHidden22, N_CL).double()
        if self.model_definition.add_liveness_filter:
            self.fc34 = nn.Linear(model_definition.nHidden24, 2).double()

        # self.s0 = Parameter(torch.ones(1), requires_grad=True)
        if model_definition.add_control_limits:
            self.s0 = Parameter(torch.ones(1), requires_grad=True).to(config.device)
            self.s1 = Parameter(torch.ones(1), requires_grad=True).to(config.device)
            self.s2 = Parameter(torch.ones(1), requires_grad=True).to(config.device)
            self.s3 = Parameter(torch.ones(1), requires_grad=True).to(config.device)
            

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.input_std + self.input_mean
        x = F.relu(self.fc1(x))
        
        x21 = F.relu(self.fc21(x))
        x31 = self.fc31(x21)

        x22 = F.relu(self.fc22(x))
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive

        if self.model_definition.add_liveness_filter:
            x24 = F.relu(self.fc24(x))
            x34 = self.fc34(x24)
            x34 = 4*nn.Sigmoid()(x34)  # ensure CBF parameters are positive
        else:
            x34 = None

        # print(x31, x32, x33, x34)

        # BarrierNet
        x = self.dCBF(x0, x31, x32, x34, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, x34, sgn, nBatch):
        theta = x0[:,EGO_THETA_IDX]
        v = x0[:,EGO_V_IDX]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        Q = Variable(torch.eye(N_CL))
        Q = Q.unsqueeze(0).expand(nBatch, N_CL, N_CL).to(config.device)

        G = []
        h = []

        # print("\n\n\nIteration")
        # print("Model inputs:", x0)

        for opp_idx in range(self.model_definition.n_opponents):
            if self.model_definition.static_obs_xy_only:
                if opp_idx == 0:
                    start_idx = opp_idx * 4 + 4
                    opp_x = x0[:, start_idx + OPP_X_OFFSET]
                    opp_y = x0[:, start_idx + OPP_Y_OFFSET]
                    opp_theta = x0[:, start_idx + OPP_THETA_OFFSET]
                    opp_vel = x0[:, start_idx + OPP_V_OFFSET]
                else:
                    start_idx = (opp_idx - 1) * 2 + 8
                    opp_x = x0[:, start_idx + OPP_X_OFFSET]
                    opp_y = x0[:, start_idx + OPP_Y_OFFSET]
                    opp_theta = torch.tensor(0.0).to(config.device)
                    opp_vel = torch.tensor(0.0).to(config.device)
            else:
                start_idx = opp_idx * 4 + 4
                opp_x = x0[:, start_idx + OPP_X_OFFSET]
                opp_y = x0[:, start_idx + OPP_Y_OFFSET]
                opp_theta = x0[:, start_idx + OPP_THETA_OFFSET]
                opp_vel = x0[:, start_idx + OPP_V_OFFSET]

            R = config.agent_radius + config.agent_radius + config.safety_dist
            if self.model_definition.x_is_d_goal:
                dx, dy = -opp_x, -opp_y
            else:
                dx = (x0[:,EGO_X_IDX] - opp_x)
                dy = (x0[:,EGO_Y_IDX] - opp_y)
            opp_sin_theta = torch.sin(opp_theta)
            opp_cos_theta = torch.cos(opp_theta)
        
            barrier = dx**2 + dy**2 - R**2
            # print("\tInputs:", opp_x, opp_y, opp_theta, opp_vel)
            # print("\tBarrier:", barrier)  
            barrier_dot = 2*dx*(v*cos_theta - opp_vel*opp_cos_theta) + 2*dy*(v*sin_theta - opp_vel*opp_sin_theta)
            # Lf2b = 2*(v*v + opp_vel*opp_vel + 2*v*opp_vel*torch.cos(theta - opp_theta))
            Lf2b = 2*(v*v + opp_vel*opp_vel - 2*v*opp_vel*torch.cos(theta + opp_theta))
            LgLfbu1 = torch.reshape(-2*dx*v*sin_theta + 2*dy*v*cos_theta, (nBatch, 1))
            LgLfbu2 = torch.reshape(2*dx*cos_theta + 2*dy*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            penalty = x32
            obs_h = (torch.reshape(Lf2b + (penalty[:,0] + penalty[:,1])*barrier_dot + (penalty[:,0] * penalty[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)

            # print("\n\nINPUT:", x0)
            # print("Dx:", dx, "Dy:", dy, "vx:", v*cos_theta, "Opp vx:", opp_vel*opp_cos_theta, "vy:", v*sin_theta, "Opp vy:", opp_vel*opp_sin_theta)
            # print("Opp barrier:", barrier)
            # print("Barrier dot:", barrier_dot)
            # print("Lf2b:", Lf2b)
            # print("LgLfbu1:", LgLfbu1)
            # print("Penalty:", penalty)
            # print("Obs g:", obs_G, "Obs h:", obs_h)

        # Add control limits as soft inequality constraints.
        if self.model_definition.add_control_limits:
            upper_G_a_lims, upper_h_a_lims = [], []
            lower_G_a_lims, lower_h_a_lims = [], []
            upper_G_w_lims, upper_h_w_lims = [], []
            lower_G_w_lims, lower_h_w_lims = [], []
            for i in range(len(x0)):
                lim_G = Variable(torch.tensor([0.0, 1.0]))
                lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                lim_h = Variable(torch.tensor([config.accel_limit * s0])).to(config.device)
                lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                upper_G_a_lims.append(lim_G)
                upper_h_a_lims.append(lim_h)

                # lim_G = Variable(torch.tensor([0.0, -1.0]))
                # lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                # lim_h = Variable(torch.tensor([config.accel_limit * 1.1])).to(config.device)
                # lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                # lower_G_a_lims.append(lim_G)
                # lower_h_a_lims.append(lim_h)

                # lim_G = Variable(torch.tensor([1.0, 0.0]))
                # lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                # lim_h = Variable(torch.tensor([config.omega_limit * 1.5])).to(config.device)
                # lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                # upper_G_w_lims.append(lim_G)
                # upper_h_w_lims.append(lim_h)

                # lim_G = Variable(torch.tensor([-1.0, 0.0]))
                # lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                # lim_h = Variable(torch.tensor([config.omega_limit * 1.5])).to(config.device)
                # lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                # lower_G_w_lims.append(lim_G)
                # lower_h_w_lims.append(lim_h)

            G.append(torch.cat(upper_G_a_lims))
            h.append(torch.cat(upper_h_a_lims))
            G.append(torch.cat(lower_G_a_lims))
            h.append(torch.cat(lower_h_a_lims))

            G.append(torch.cat(upper_G_w_lims))
            h.append(torch.cat(upper_h_w_lims))
            G.append(torch.cat(lower_G_w_lims))
            h.append(torch.cat(lower_h_w_lims))


        # Add in liveness CBF
        if self.model_definition.add_liveness_filter:
            G_live, h_live = [], []
            for i in range(len(x0)):
                ego_pos = np.array([0.0, 0.0])
                ego_theta = x0[i, EGO_THETA_IDX].item()
                ego_vel = x0[i, EGO_V_IDX].item()
                opp_pos = np.array([x0[i, 4 + OPP_X_OFFSET].item(), x0[i, 4 + OPP_Y_OFFSET].item()])
                opp_theta = x0[i, 4 + OPP_THETA_OFFSET].item()
                opp_vel = x0[i, 4 + OPP_V_OFFSET].item()

                center_intersection = get_ray_intersection_point(ego_pos, ego_theta, opp_pos, opp_theta)

                vec_to_opp = np.array([opp_pos[1] - ego_pos[1], opp_pos[0] - ego_pos[0]])
                unit_vec_to_opp = vec_to_opp / np.linalg.norm(vec_to_opp)
                initial_closest_to_opp = ego_pos + unit_vec_to_opp * (config.agent_radius)
                opp_closest_to_initial = opp_pos - unit_vec_to_opp * (config.agent_radius)
                intersection = get_ray_intersection_point(initial_closest_to_opp, ego_theta, opp_closest_to_initial, opp_theta)
                if center_intersection is None or intersection is None or ego_vel == 0 or opp_vel == 0:
                    # print("No intersection!", x31*self.output_std + self.output_mean)
                    lim_G = Variable(torch.tensor([0.0, 1.0]))
                    lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                    lim_h = Variable(torch.tensor([config.accel_limit * 100.0])).to(config.device)
                    lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                    G_live.append(lim_G)
                    h_live.append(lim_h)
                    continue

                d0_center = np.linalg.norm(ego_pos - intersection)
                d1_center = np.linalg.norm(opp_pos - intersection)

                t0 = d0_center / ego_vel
                t1 = d1_center / opp_vel

                d0 = np.linalg.norm(initial_closest_to_opp - intersection)
                d1 = np.linalg.norm(opp_closest_to_initial - intersection)

                # # if x0[i, EGO_V_IDX] > x0[i, 4 + OPP_V_OFFSET]:
                # #     print("STARTING:", x0[i, :8])
                # #     print("\tEgo pos:", ego_pos, "Ego theta:", np.degrees(ego_theta), " Opp pos:", opp_pos, "Opp theta:", np.degrees(opp_theta))
                # #     print("\tIntersection:", intersection, "D0:", d0, "D1:", d1)

                if t0 >= t1: # If slower agent
                    # barrier = (d0 * opp_vel -  d1 * ego_vel)
                    # penalty = x34[i, 0]
                    # live_G = Variable(torch.tensor([0.0, d1]).to(config.device)).to(config.device)
                    # live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                    # live_h = torch.reshape(penalty * barrier, (1, 1)).to(config.device)
                    barrier = d0 / ego_vel - d1 / opp_vel
                    penalty = x34[i, 0]
                    live_G = Variable(torch.tensor([0.0, d0 / (ego_vel ** 2.0)]).to(config.device)).to(config.device)
                    live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                    live_h = torch.reshape(penalty * barrier, (1, 1)).to(config.device)
                else: # If faster agent
                    # barrier = (d1 * ego_vel - d0 * opp_vel)
                    # penalty = x34[i, 1]
                    # live_G = Variable(torch.tensor([0.0, -d1]).to(config.device)).to(config.device)
                    # live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                    # live_h = torch.reshape(penalty * barrier, (1, 1)).to(config.device)
                    barrier = d1 / opp_vel - d0 / ego_vel
                    penalty = x34[i, 1]
                    live_G = Variable(torch.tensor([0.0, -d0 / (ego_vel ** 2.0)]).to(config.device)).to(config.device)
                    live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                    live_h = torch.reshape(penalty * barrier, (1, 1)).to(config.device)
                    # print("\nClosest points:", initial_closest_to_opp, opp_closest_to_initial)
                    # print("Intersects:", center_intersection, intersection)
                    # print("D0", d0, "D1", d1, "Vel", ego_vel, "opp V", opp_vel)
                    # print("Barrier:", barrier)
                    # print("Ineq:", live_G, live_h)

                G_live.append(live_G)
                h_live.append(live_h)

            G_live = torch.cat(G_live)
            h_live = torch.cat(h_live)
            # print("Shapes:", G_live.shape, h_live.shape)
            G.append(G_live)
            h.append(h_live)

        # print("Num ineq", len(G), len(h))
        # for i in range(len(G)):
        #     print(f"Inequality {i}", G[i], h[i])
        
        G = torch.cat(G, dim=1).to(config.device)
        h = torch.cat(h, dim=1).to(config.device)

        # print(G.shape, h.shape)

        # num_ineq = len(obstacles) + len(opps) + self.model_definition.add_control_limits * 2 + self.model_definition.add_liveness_filter
        # assert(G.shape == (nBatch, num_ineq, N_CL))
        # assert(h.shape == (nBatch, num_ineq))
        e = Variable(torch.Tensor()).to(config.device)

        x31_actual = x31*self.output_std + self.output_mean
        # print("Ref control:", x31_actual)
        if self.training or sgn == 1:
            x = QPFunction(verbose = 0)(Q.double(), x31_actual.double(), G.double(), h.double(), e, e)
            # x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)

            # x = F.relu(x + config.accel_limit) - config.accel_limit  # ensure CBF parameters are positive
            # x = -(F.relu(-x + config.accel_limit) - config.accel_limit)  # ensure CBF parameters are positive

            x = (x - self.output_mean) / self.output_std
        else:
            # print(x31[0].cpu())
            try:
                # x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
                x = solver(Q[0].double(), x31_actual[0].double(), G[0].double(), h[0].double())
                x = np.array([x[0], x[1]])
            except Exception as e:
                print("ERROR WHEN SOLVING FOR OPTIMIZER, USING REFERENCE CONTROL INSTEAD:", x31)
                x = x31_actual[0].cpu()

            x = (x - self.output_mean_np) / self.output_std_np
        
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
