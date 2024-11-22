import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_environment(obstacles, agents, goals):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot obstacles
    for obs in obstacles:
        x, y, size = obs
        circle = patches.Circle((x, y), size, color='gray', alpha=0.6, label='Obstacle' if obs == obstacles[0] else "")
        ax.add_patch(circle)
    
    # Plot agents
    for agent in agents:
        x, y, size = agent
        circle = patches.Circle((x, y), size, color='blue', alpha=0.8, label='Agent' if agent == agents[0] else "")
        ax.add_patch(circle)

    # Plot goals
    for goal in goals:
        x, y, size = goal
        circle = patches.Circle((x, y), size, color='green', alpha=0.9, label='Goal' if goal == goals[0] else "")
        ax.add_patch(circle)

    # Set axis limits
    all_entities = obstacles + agents + goals
    xs = [e[0] for e in all_entities]
    ys = [e[1] for e in all_entities]
    padding = 0.5  # Add some padding around the environment
    ax.set_xlim(min(xs) - padding, max(xs) + padding)
    ax.set_ylim(min(ys) - padding, max(ys) + padding)

    # Labels and legend
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Environment Visualization')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')

    plt.grid(True)
    plt.show()


# Example usage:
obstacles = [[0.21477941728022892, 0.9468332120386458, 0.1], [0.4521588547152331, 0.1974860051863462, 0.1], [-0.3359188829088193, 0.641934097493287, 0.1], [-0.274616270444153, 1.7238012034411512, 0.1], [2.040116146204529, -0.5128573171665781, 0.1], [1.2835901675637242, 0.12713764691277973, 0.1], [0.29939606881330216, 1.8726252084877086, 0.1], [-1.8874413439292976, -1.816631081313221, 0.1], [-2.111039051262567, 1.4635273204109274, 0.1], [1.2238897041793422, 1.6280534522860044, 0.1], [2.1059207058241616, 1.316297682553584, 0.1], [-0.16949080608709988, 1.2343283756604042, 0.1], [-1.679592526176694, 0.6156524938411049, 0.1], [-1.569245535400196, 1.9565432350181693, 0.1], [0.09613261570031538, -0.37548746404169625, 0.1], [-1.0359553067396414, 1.2066282335105534, 0.1]]
agents = [[-0.1929385382471864, 0.3011093750220535, 0.1], [-2.1173248780800376, 0.5175961871338591, 0.1]]
goals = [[0.4932211799786542, 0.5145095862489305, 0.05], [1.9524915454643466, 0.8000093160553271, 0.05]]

plot_environment(obstacles, agents, goals)
