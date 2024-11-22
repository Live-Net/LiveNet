import numpy as np
import random
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from bridson import poisson_disc_samples

class Scenario(BaseScenario):
    def make_world(self, sort_obs=True, use_numba=False):
        world = World(use_numba)
        self.np_rnd = np.random.RandomState(0)
        self.random = random.Random()
        self.sort_obs = sort_obs
        # set any world properties first
        world.dim_c = 2
        num_agents = 128
        self.num_agents = 128
        self.num_landmarks = 128
        num_landmarks = 128
        world.collaborative = True
        self.agent_size = 0.15
        self.world_radius = 2.2
        self.n_others = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):

         # metrics to keep track of
        #world.current_time_step = 0
        # to track time required to reach goal
        #world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        #world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        #world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)


        self.l_locations = poisson_disc_samples(width=self.world_radius * 2, height=self.world_radius * 2,
                                                r=self.agent_size * 4.5)
        while len(self.l_locations) < len(world.landmarks):
            self.l_locations = poisson_disc_samples(width=self.world_radius * 2, height=self.world_radius * 2,
                                                    r=self.agent_size * 4.5)
            # print('regenerate l location')

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        l_locations = np.array(self.random.sample(self.l_locations, len(world.landmarks))) - self.world_radius
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = l_locations[i, :]
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.collide_th = 2 * world.agents[0].size

    def info_callback(self, agent: Agent, world: World):
        # TODO modify this
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        goal = world.get_entity("landmark", agent.id)
        #dist = np.sqrt(np.sum(np.square(agent.state.p_pos - goal.state.p_pos)))
        #world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        #if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
        #    world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            #if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
            #    world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            #"Dist_to_goal": world.dist_left_to_goal[agent.id],
            #"Time_req_to_goal": world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            #"Num_obst_collisions": world.num_obstacle_collisions[agent.id],
            "Agent_speed": np.linalg.norm(agent.state.p_vel),
        }
        return agent_info


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        """
        Vectorized reward function
        Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        """

        rew, rew1 = 0, 0

        if agent == world.agents[0]:
            """
            for l in world.landmarks:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
                rew1 -= min(dists)
            """
            l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(len(world.agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            a_pos1 = a_pos.repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = a_pos.repeat(len(world.agents), axis=0)
            dist = np.sqrt(np.sum(np.square(l_pos - a_pos1), axis=2))
            rew = np.min(dist, axis=0)
            rew = -np.sum(rew)
            if agent.collide:
                dist_a = np.sqrt(np.sum(np.square(a_pos1 - a_pos2), axis=2))
                n_collide = (dist_a < self.collide_th).sum() - len(world.agents)
                rew -= n_collide

        return rew

    def observation(self, agent, world):
        """
        :param agent: an agent
        :param world: the current world
        :return: obs: [18] np array,
        [0-1] self_agent velocity
        [2-3] self_agent location
        [4-9] landmarks location
        [10-11] agent_i's relative location
        [12-13] agent_j's relative location
        Note that i < j
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        entity_dist = np.sqrt(np.sum(np.square(np.array(entity_pos) - agent.state.p_pos), axis=1))
        entity_dist_idx = np.argsort(entity_dist)
        entity_pos = [entity_pos[i] for i in entity_dist_idx[:self.n_others + 1]]

        other_dist = np.sqrt(np.sum(np.square(np.array(other_pos) - agent.state.p_pos), axis=1))
        dist_idx = np.argsort(other_dist)
        other_pos = [other_pos[i] for i in dist_idx[:self.n_others]]
        #other_pos = sorted(other_pos, key=lambda k: [k[0], k[1]])
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
        self.random.seed(seed)