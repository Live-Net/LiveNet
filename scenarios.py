import config
from config import DynamicsModel
import numpy as np
import matplotlib.patches as patches

# Add all initial and goal positions of the agents here (Format: [x, y, theta])

def rotate_point(point, center, angle):
    center_to_point = point - center
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(rotation_matrix, center_to_point)
    return rotated + center

def rotate_objs(objs, center, angle):
    new_objs = []
    for obj in objs:
        new_pos = rotate_point(obj[:2], center, angle)
        new_heading = obj[2] + angle
        new_objs.append(np.array([new_pos[0], new_pos[1], new_heading]))
    return np.array(new_objs)


class DoorwayScenario:
    def __init__(self, initial_x=-1, initial_y=0.5, goal_x=2, goal_y=0.15, start_facing_goal=False, initial_vel=0.0):
        self.num_agents = 2
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.config = (initial_x, initial_y, goal_x, goal_y, start_facing_goal, initial_vel)
        self.initial = np.array([[self.initial_x, self.initial_y, 0],
                    [self.initial_x, -self.initial_y, 0]])
        self.goals = np.array([[self.goal_x, -self.goal_y, 0.0],
                    [self.goal_x, self.goal_y, 0.0]])
        if start_facing_goal:
            self.initial[0][2] = np.arctan2(self.goals[0][1] - self.initial[0][1], self.goals[0][0] - self.initial[0][0])
            self.initial[1][2] = np.arctan2(self.goals[1][1] - self.initial[1][1], self.goals[1][0] - self.initial[1][0])
        if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
            # Set initial state to 0 velocity and goal to 0 velocity.
            vels = np.ones((self.num_agents, 1)) * initial_vel
            zeros = np.zeros((self.num_agents, 1))
            self.initial = np.hstack((self.initial, vels))
            self.goals = np.hstack((self.goals, zeros))
        self.ox = 1
        self.obstacles = []
        self.obs_starting_y = 0.25
        self.obs_length = 1.0
        for y in np.arange(self.obs_starting_y, 2, 0.1):
            self.obstacles.append((self.ox, y, 0.1))
            self.obstacles.append((self.ox, -y, 0.1))
        # self.plot_bounds = np.array([[-2.5, -1.5], [5.0, 1.5]])
        self.plot_bounds = np.array([[-1.5, -2], [2.5, 2]])

    def plot(self, ax):
        for obs_x, obs_y, r in self.obstacles:
            circle = patches.Circle((obs_x, obs_y), r, linewidth=1,edgecolor='k',facecolor='k',fill=True)
            ax.add_patch(circle)
        ax.scatter(self.goals[0, 0], self.goals[0, 1], c='r', marker='x', s=500)
        ax.scatter(self.goals[1, 0], self.goals[1, 1], c='b', marker='x', s=500)
        # ax.scatter(self.goals[0, 0], self.goals[0, 1], c='r', marker='x', s=100)
        # ax.scatter(self.goals[1, 0], self.goals[1, 1], c='b', marker='x', s=100)

    def __str__(self):
        return f"Doorway Scenario with config: {self.config}"


    def save_str(self):
        return f"s_doorway_{self.config[0]}_{self.config[1]}_{self.config[2]}_{self.config[3]}_{self.config[4]}_{self.config[5]}"


class NoObstacleDoorwayScenario:
    def __init__(self, rotation=0):
        self.num_agents = 2
        goal_y = 0.4
        self.initial = np.array([[-1, 0.5, 0],
                    [-1, -0.5, 0]])
        self.goals = np.array([[2.3, -goal_y, 0.0],
                    [2.3, goal_y, 0.0]])
        self.plot_bounds = [[-2.5, -2.5], [2.5, 2.5]]

        self.env_center = np.array([(2.3-1) / 2.0, 0.0])
        # Perform rotation
        self.initial = rotate_objs(self.initial, self.env_center, rotation)
        self.goals = rotate_objs(self.goals, self.env_center, rotation)
        self.plot_bounds = np.array([rotate_point(bound, self.env_center, rotation) for bound in self.plot_bounds])

        if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
            # Set initial state to 0 velocity and goal to 0 velocity.
            zeros = np.zeros((self.num_agents, 1))
            self.initial = np.hstack((self.initial, zeros))
            self.goals = np.hstack((self.goals, zeros))
        self.obstacles = []
    
    def plot(self, ax):
        # ax.scatter(self.goals[0, 0], self.goals[0, 1], c='r', marker='x', s=100)
        # ax.scatter(self.goals[1, 0], self.goals[1, 1], c='b', marker='x', s=100)
        ax.scatter(self.goals[0, 0], self.goals[0, 1], c='r', marker='x', s=1500)
        ax.scatter(self.goals[1, 0], self.goals[1, 1], c='b', marker='x', s=1500)


class IntersectionScenario:
    def __init__(self, start=1.0, goal=1.0, start_vel=0.0):
        self.initial = np.array([[0.0, -start, np.pi / 2, start_vel],
                      [-start, 0.0, 0.0, start_vel]])
        self.goals = np.array([[0.0, goal, np.pi / 2, 0.0],
                    [goal, 0.0, 0.0, 0.0]
                    ])
        self.config = (start, goal, start_vel)
        self.ox=-0.3
        self.ox1=0.3

        self.obstacles=[(self.ox, 0.3, 0.1),(self.ox, 0.4, 0.1),(self.ox, 0.5, 0.1),(self.ox, 0.6, 0.1),(self.ox, 0.7, 0.1),(self.ox, 0.8, 0.1),(self.ox, 0.9, 0.1),
                  (self.ox, 1.0, 0.1), (self.ox, -0.3, 0.1),(self.ox, -0.4, 0.1),(self.ox, -0.5, 0.1),(self.ox, -0.6, 0.1),(self.ox, -0.7, 0.1),      
                (self.ox, -0.8, 0.1),(self.ox, -0.9, 0.1),(self.ox, -1.0, 0.1),

                (self.ox1, 0.3, 0.1),(self.ox1, 0.4, 0.1),(self.ox1, 0.5, 0.1),(self.ox1, 0.6, 0.1),(self.ox1, 0.7, 0.1),(self.ox1, 0.8, 0.1),(self.ox1, 0.9, 0.1),(self.ox1, 1.0, 0.1),
                  (self.ox1, -0.3, 0.1),(self.ox1, -0.4, 0.1),(self.ox1, -0.5, 0.1),(self.ox, -0.6, 0.1),(self.ox1, -0.7, 0.1),(self.ox1, -0.8, 0.1),(self.ox1, -0.9, 0.1),(self.ox1, -1.0, 0.1),
                  
                  (0.3,self.ox, 0.1), (0.4,self.ox, 0.1),( 0.5,self.ox, 0.1),(0.6,self.ox, 0.1),( 0.7,self.ox, 0.1),(0.8,self.ox, 0.1),(0.9,self.ox, 0.1),(1.0,self.ox, 0.1),
                  (-0.3,self.ox, 0.1), (-0.4,self.ox, 0.1),(-0.5,self.ox, 0.1),(-0.6,self.ox, 0.1),(-0.7,self.ox, 0.1),(-0.8,self.ox, 0.1),(-0.9,self.ox, 0.1),(-1.0,self.ox, 0.1),

                (0.3,self.ox1, 0.1), ( 0.4,self.ox1, 0.1),(0.5,self.ox1, 0.1),(0.6, self.ox1, 0.1),(0.7,self.ox1, 0.1),(0.8,self.ox1, 0.1),(0.9,self.ox1, 0.1),( 1.0, self.ox1, 0.1),
                (-0.3,self.ox1, 0.1), ( -0.4,self.ox1, 0.1),(-0.5, self.ox1, 0.1),( -0.6, self.ox1, 0.1),( -0.7,self.ox1, 0.1),( -0.8,self.ox1, 0.1),( -0.9,self.ox1, 0.1),(-1.0,self.ox1, 0.1)]
        self.plot_bounds = np.array([[-1.5, -1.5], [1.5, 1.5]])


    def plot(self, ax):
        for obs_x, obs_y, r in self.obstacles:
            circle = patches.Circle((obs_x, obs_y), r, linewidth=1,edgecolor='k',facecolor='k',fill=True)
            ax.add_patch(circle)
        ax.scatter(self.goals[0, 0], self.goals[0, 1], c='r', marker='x', s=500)
        ax.scatter(self.goals[1, 0], self.goals[1, 1], c='b', marker='x', s=500)


        # length=1
        # ox, ox1 = self.ox, self.ox1
        # rect = patches.Rectangle((ox-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect1 = patches.Rectangle((ox1-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect2 = patches.Rectangle((ox1-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect3 = patches.Rectangle((ox-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect4 = patches.Rectangle((-length,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect5 = patches.Rectangle((-length,ox1-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect6 = patches.Rectangle((ox1-0.1,ox1-0.1),1-ox1+0.2,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # rect7 = patches.Rectangle((ox1,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        # ax.add_patch(rect)
        # ax.add_patch(rect1)
        # ax.add_patch(rect2)
        # ax.add_patch(rect3)
        # ax.add_patch(rect4)
        # ax.add_patch(rect5)
        # ax.add_patch(rect6)
        # ax.add_patch(rect7)
    
    def __str__(self):
        return f"Intersection Scenario with config {self.config}"


    def save_str(self):
        return f"s_intersection_{self.config[0]}_{self.config[1]}_{self.config[2]}"
