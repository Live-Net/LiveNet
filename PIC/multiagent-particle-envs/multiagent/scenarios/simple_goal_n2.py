import numpy as np
import random
from multiagent.core_vec import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, sort_obs=True, use_numba=False):
        world = World(use_numba)
        self.np_rnd = np.random.RandomState(0)
        self.random = random.Random()
        self.sort_obs = sort_obs
        # Set world properties
        world.dim_c = 2
        num_agents = 2  # Two agents
        self.num_agents = num_agents
        num_landmarks = 2  # Two goals, one for each agent
        num_obstacles = 16  # Sixteen static obstacles
        world.collaborative = False  # Each agent has its own goal
        self.agent_size = 0.1
        self.world_radius = 2.2
        self.n_others = 1  # Number of other agents to observe
        self.n_obstacles = 16  # Number of obstacles to observe

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.size = self.agent_size
            agent.id = i

        # Add landmarks (goals)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            # landmark.size = 0.05   # 0.05
            # print("momo")
            # print(landmark.size)
            landmark.id = i
            landmark.name = f'goal {i}'
            landmark.collide = False
            landmark.movable = False
            landmark.is_goal = True  # Mark as goal

        # Add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.size = self.agent_size
            obstacle.id = i
            obstacle.name = f'obstacle {i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.color = np.array([0.5, 0.5, 0.5])  # Gray color
            obstacle.is_goal = False

        # Assign each agent its goal
        for i, agent in enumerate(world.agents):
            agent.goal = world.landmarks[i]

        # Collect all entities
        # world.entities = world.agents + world.landmarks + world.obstacles

        # No walls in this environment
        world.walls = []

        # Make initial conditions
        self.reset_world(world)

        return world
    
    def reset_world(self, world):
        """
        Reset the world by placing agents, goals, and obstacles using self.np_rnd.uniform.
        Ensure no overlapping entities.
        """
        world.num_agent_collisions = np.zeros(self.num_agents)
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        # Random properties for agents
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        # Random properties for landmarks (goals)
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
        # Obstacles are gray (already set in make_world)

        # Place obstacles randomly within the world
        obstacle_debug = []
        for obstacle in world.obstacles:
            placed = False
            while not placed:
                obstacle.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
                obstacle.state.p_vel = np.zeros(world.dim_p)
                # Check for overlaps with previously placed obstacles
                overlap = False
                for other in world.obstacles:
                    if other is obstacle or other.state.p_pos is None:
                        continue
                    delta_pos = obstacle.state.p_pos - other.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    if dist < obstacle.size * 2:
                        overlap = True
                        break
                if not overlap:
                    placed = True
                    obstacle_debug.append(obstacle.state.p_pos.tolist() + [obstacle.size])

        # Place agents randomly, avoiding obstacles
        agents_debug = []
        for agent in world.agents:
            placed = False
            while not placed:
                agent.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                # Check for overlaps with obstacles
                overlap = False
                for obstacle in world.obstacles:
                    delta_pos = agent.state.p_pos - obstacle.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    if dist < agent.size + obstacle.size:
                        overlap = True
                        break
                if not overlap:
                    placed = True
                    agents_debug.append(agent.state.p_pos.tolist() + [agent.size])
                agent.state.c = np.zeros(world.dim_c)

        # Place goals randomly, avoiding obstacles and agents
        landmarks_debug = []
        for i, landmark in enumerate(world.landmarks):
            placed = False
            while not placed:
                landmark.state.p_pos = self.np_rnd.uniform(-self.world_radius, +self.world_radius, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                # Check for overlaps with obstacles and agents
                overlap = False
                for obstacle in world.obstacles:
                    delta_pos = landmark.state.p_pos - obstacle.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    if dist < landmark.size + obstacle.size:
                        overlap = True
                        break
                for agent in world.agents:
                    delta_pos = landmark.state.p_pos - agent.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    if dist < landmark.size + agent.size:
                        overlap = True
                        break
                if not overlap:
                    placed = True
                    landmarks_debug.append(landmark.state.p_pos.tolist() + [landmark.size])

        self.collide_th = 2 * world.agents[0].size
        
        # print("RESET ENV DEBUG INFO:")
        # print(f"obstacles = {obstacle_debug}")
        # print(f"agents = {agents_debug}")
        # print(f"goals = {landmarks_debug}")
        # print("")



    def reward(self, agent, world):
        """
        Agents are rewarded based on the negative distance to their respective goals,
        penalized for collisions with other agents and obstacles.
        """
        rew, rew1 = 0, 0
        # if agent == world.agents[0]:
            # Distance to the agent's goal
        goal = agent.goal
        dist_to_goal = np.sqrt(np.sum(np.square(agent.state.p_pos - goal.state.p_pos)))
        # Reward is negative distance to the goal
        rew = -dist_to_goal

        # Penalize collision with other agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                if self.is_collision(agent, other_agent):
                    rew -= 10  # Penalty for collision with another agent

        # Penalize collision with obstacles
        for obstacle in world.obstacles:
            if self.is_collision(agent, obstacle):
                rew -= 10  # Penalty for collision with an obstacle

        return rew

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

    def observation(self, agent, world):
        """
        Returns the observation for an agent.

        Observation includes:
        - Agent's own velocity (2D)
        - Agent's own position (2D)
        - Relative position to its goal (2D)
        - Relative positions of other agents (2D * n_others)
        - Relative positions of nearby obstacles (2D * n_obstacles)
        """
        # Agent's own velocity and position
        obs = [agent.state.p_vel, agent.state.p_pos]

        # Relative position to the agent's goal
        goal_pos = agent.goal.state.p_pos - agent.state.p_pos
        obs.append(goal_pos)

        # Positions of other agents
        other_agents = []
        for other in world.agents:
            if other is agent:
                continue
            other_agents.append(other.state.p_pos - agent.state.p_pos)
        # Ensure we have n_others entries
        n_others = self.n_others
        other_agents = sorted(other_agents, key=lambda x: np.linalg.norm(x))[:n_others]
        # Pad with zeros if needed
        while len(other_agents) < n_others:
            other_agents.append(np.zeros(world.dim_p))
        obs.extend(other_agents)

        # Positions of obstacles
        obstacle_pos = []
        for obstacle in world.obstacles:
            obstacle_pos.append(obstacle.state.p_pos - agent.state.p_pos)
        # Ensure we have n_obstacles entries
        n_obstacles = self.n_obstacles
        obstacle_pos = sorted(obstacle_pos, key=lambda x: np.linalg.norm(x))[:n_obstacles]
        # Pad with zeros if needed
        while len(obstacle_pos) < n_obstacles:
            obstacle_pos.append(np.zeros(world.dim_p))
        obs.extend(obstacle_pos)

        # Flatten the observation
        obs = np.concatenate(obs)

        # print("HEYO")
        # print(f"obs = {obs.shape}") 
        return obs

    def info_callback(self, agent: Agent, world: World):
        # Collect info about collisions and agent speed
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1
            for obstacle in world.obstacles:
                if self.is_collision(agent, obstacle):
                    world.num_obstacle_collisions[agent.id] += 1

        agent_info = {
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obstacle_collisions": world.num_obstacle_collisions[agent.id],
            "Agent_speed": np.linalg.norm(agent.state.p_vel),
        }
        return agent_info

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
        self.random.seed(seed)
