import time
import do_mpc
from casadi import *
import config
from config import DynamicsModel
import numpy as np
from memory_profiler import profile

class Environment:
    def __init__(self, initial_states, goals):
        self.num_agents = len(initial_states)
        self.initial_states = initial_states
        self.goals = goals
        self.history = [initial_states.copy()]
        self.compute_history = []

        self.model = self.define_model()
        self.simulator = self.define_simulator()

    def define_model(self, call_setup=True):
        """Configures the dynamical model of the system (and part of the objective function).

        x_{k+1} = x_k + B*u_k*T_s
        Returns:
          - model(do_mpc.model.Model): The system model
        """

        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # Num states, State Space matrices, and Input Space matrices
        _x = model.set_variable(var_type='_x', var_name='x', shape=(config.num_states, 1))
        _u = model.set_variable(var_type='_u', var_name='u', shape=(config.num_controls, 1))
        A, B = self.get_dynamics(_x)

        # Set right-hand-side of ODE for all introduced states (_x).
        x_next = _x + A*config.sim_ts + B@_u*config.sim_ts
        model.set_rhs('x', x_next, process_noise=False)  # Set to True if adding noise

        # Setup model
        if call_setup:
            model.setup()
        return model

    @staticmethod
    def get_dynamics(x):
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            return Environment.get_single_integrator_dynamics(x)
        else:
            return Environment.get_double_integrator_dynamics(x)

    """Defines the system input matrices A and B for single-integrator dynamics."""
    @staticmethod
    def get_single_integrator_dynamics(x):
        A = SX.zeros(3, 1)

        a = 1e-9  # Small positive constant so system has relative degree 1
        # [[cos(theta), 0]
        #  [sin(theta), 0]
        #  [0,          1]]
        B = SX.zeros(3, 2)
        B[0, 0] = cos(x[2])
        B[0, 1] = -a*sin(x[2])
        B[1, 0] = sin(x[2])
        B[1, 1] = a*cos(x[2])
        B[2, 1] = 1
        return A, B

    """Defines the system input matrices A and B for double-integrator dynamics."""
    @staticmethod
    def get_double_integrator_dynamics(x):
        A = SX.zeros(4, 1)
        A[0] = x[3] * cos(x[2]) # x_dot = v * cos(theta)
        A[1] = x[3] * sin(x[2]) # y_dot = v * sin(theta)
        # dx = 1/2 at^2 + vt
        # dx = 1/2 a_xt^2 + v_xt
        # v_xt = v cos(theta)
        # a_xt = a 

        B = SX.zeros(4, 2)
        B[2, 0] = 1 # dtheta = omega
        B[3, 1] = 1 # dv = a

        a = 1e-9  # Small positive constant so system has relative degree 1
        B[0, 1] = -a*sin(x[2])
        B[1, 1] = a*cos(x[2])
        return A, B

    """Defines the system input matrices A and B for double-integrator dynamics."""
    @staticmethod
    def get_double_integrator_dynamics_np(x):
        A = np.zeros((4,))
        A[0] = x[3] * np.cos(x[2]) # x_dot = v * cos(theta)
        A[1] = x[3] * np.sin(x[2]) # y_dot = v * sin(theta)

        B = np.zeros((4, 2))
        B[2, 0] = 1 # dtheta = omega
        B[3, 1] = 1 # dv = a
        return A, B

    """Configures the simulator."""
    def define_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=config.sim_ts)
        simulator.setup()

        return simulator

    """Sets the initial state in all components."""
    def reset_state(self, x0):
        self.simulator.reset_history()
        self.simulator.x0 = x0
    
    def apply_state_lims(self, state):
        if state[3] > config.v_limit:
            state[3] = config.v_limit
        if state[3] < -config.v_limit:
            state[3] = -config.v_limit
        return state

    def apply_control_lims(self, control):
        if control[0] > config.omega_limit:
            control[0] = config.omega_limit
        if control[0] < -config.omega_limit:
            control[0] = -config.omega_limit
        if control[1] > config.accel_limit:
            control[1] = config.accel_limit
        if control[1] < -config.accel_limit:
            control[1] = -config.accel_limit
        return control

    # @profile
    def run_simulation(self, sim_iteration, controllers, logger):
        """Runs a closed-loop control simulation."""
        self.sim_iteration = sim_iteration
        sim_time = self.sim_iteration * config.sim_ts

        new_states = np.zeros((self.num_agents, config.num_states))
        outputted_controls = np.zeros((self.num_agents, config.num_controls))
        use_for_training = []
        compute_times = []
        for agent_idx in range(self.num_agents):
            # config.logging = agent_idx == 1
            # print(f"\nRunning Agent: {agent_idx}")
            controller = controllers[agent_idx]
            initial_state = self.initial_states[agent_idx, :]
            opp_state = self.initial_states[1-agent_idx, :].copy()
            # If single-integrator dynamics, add velocity to this state.
            if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
                opp_vel = 0.0 if len(self.history) < 2 else np.linalg.norm(opp_state[:2] - self.history[-2][1-agent_idx, :2]) / config.sim_ts
                opp_state = np.append(opp_state, [opp_vel])
            self.reset_state(initial_state)
            cycle_start_time = time.time()
            controller.reset_state(initial_state, opp_state)
            u1 = self.apply_control_lims(controller.make_step(sim_time, initial_state))
            compute_times.append(time.time() - cycle_start_time)
            x1 = self.simulator.make_step(u1)
            new_states[agent_idx, :] = self.apply_state_lims(x1.ravel())
            outputted_controls[agent_idx, :] = u1.ravel()
            # print(f"Initial state: {initial_state}, Output control: {outputted_controls[agent_idx, :]}, New state: {new_states[agent_idx, :]}")
            use_for_training.append(controller.use_for_training)

        # if sim_time >= abs(config.agent_zero_offset):
        logger.log_iteration(self.initial_states, self.goals, outputted_controls, use_for_training, compute_times)
        self.compute_history.append(compute_times)
        self.initial_states = new_states.copy()
        self.history.append(new_states.copy())
        return new_states, outputted_controls
