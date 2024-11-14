import numpy as np
import torch
import torch.nn as nn
from . import config_macbf
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def generate_data_np(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros((num_agents, 4), dtype=np.float32)
    goals = np.zeros((num_agents, 2), dtype=np.float32)
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(0, side_length, size=(2,))
        if i > 0:
            dist_min = np.linalg.norm(states[:i, :2] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        states[i, :2] = candidate
        i += 1
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i, :2]
        if i > 0:
            dist_min = np.linalg.norm(goals[:i] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        goals[i] = candidate
        i += 1
    return states, goals


# Torch Version
def generate_data_torch(num_agents, dist_min_thres, device):
    with torch.no_grad():
        side_length = torch.sqrt(torch.tensor(max(1.0, num_agents / 8.0), device=device))
        states = torch.zeros((num_agents, 4), dtype=torch.float32, device=device)
        goals = torch.zeros((num_agents, 2), dtype=torch.float32, device=device)
        i = 0
        while i < num_agents:
            candidate = torch.rand(2, device=device) * side_length
            if i > 0:
                dist_min = torch.norm(states[:i, :2] - candidate, dim=1).min()
                if dist_min <= dist_min_thres:
                    continue
            states[i, :2] = candidate
            i += 1
        i = 0
        while i < num_agents:
            candidate = (torch.rand(2, device=device) - 0.5) + states[i, :2]
            if i > 0:
                dist_min = torch.norm(goals[:i] - candidate, dim=1).min()
                if dist_min <= dist_min_thres:
                    continue
            goals[i] = candidate
            i += 1
    return states, goals


def generate_wall_hole(wall_x, hole_y_center, hole_height):
    hole_y_min = hole_y_center - hole_height / 2
    hole_y_max = hole_y_center + hole_height / 2
    wall_y_positions = np.linspace(
        config_macbf.Y_MIN, config_macbf.Y_MAX, (config_macbf.Y_MAX - config_macbf.Y_MIN) * config_macbf.RES + 1
    )
    wall_y_positions = wall_y_positions[
        (wall_y_positions <= hole_y_min) | (wall_y_positions >= hole_y_max)
    ]
    wall_x_positions = np.ones_like(wall_y_positions) * wall_x
    wall_x_positions = np.expand_dims(wall_x_positions, axis=1)
    wall_y_positions = np.expand_dims(wall_y_positions, axis=1)
    return np.concatenate([wall_x_positions, wall_y_positions], axis=1)


def generate_border():
    bx, by = [], []
    bx.append(
        np.linspace(
            config_macbf.X_MIN, config_macbf.X_MAX, (config_macbf.X_MAX - config_macbf.X_MIN) * config_macbf.RES + 1
        )
    )
    by.append(np.ones_like(bx[-1]) * config_macbf.Y_MIN)
    bx.append(
        np.linspace(
            config_macbf.X_MIN, config_macbf.X_MAX, (config_macbf.X_MAX - config_macbf.X_MIN) * config_macbf.RES + 1
        )
    )
    by.append(np.ones_like(bx[-1]) * config_macbf.Y_MAX)
    by.append(
        np.linspace(
            config_macbf.Y_MIN, config_macbf.Y_MAX, (config_macbf.Y_MAX - config_macbf.Y_MIN) * config_macbf.RES + 1
        )
    )
    bx.append(np.ones_like(by[-1]) * config_macbf.X_MIN)
    by.append(
        np.linspace(
            config_macbf.Y_MIN, config_macbf.Y_MAX, (config_macbf.Y_MAX - config_macbf.Y_MIN) * config_macbf.RES + 1
        )
    )
    bx.append(np.ones_like(by[-1]) * config_macbf.X_MAX)
    bx = np.expand_dims(np.concatenate(bx, axis=0), axis=1)
    by = np.expand_dims(np.concatenate(by, axis=0), axis=1)
    border_points = np.concatenate([bx, by], axis=1)
    return border_points


"""Rotates points by theta around a given origin."""
def rotate_points(points, theta, origin):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    # Shift points to origin, rotate, and shift back
    translated_points = points - origin
    rotated_points = np.dot(translated_points, rotation_matrix) + origin
    return rotated_points


def generate_social_mini_game_data(theta):
    x_min, x_max = 0.0, 10.0
    y_min, y_max = 0.0, 10.0
    wall_x = 5.0
    hole_y_center = 5.0
    hole_height = 0.2
    hole_y_min = hole_y_center - hole_height / 2
    hole_y_max = hole_y_center + hole_height / 2

    num_agents = 2
    agent_states = np.zeros((num_agents, 4), dtype=np.float32)
    agent_goals = np.zeros((num_agents, 2), dtype=np.float32)
    agent_offset = 2.0

    y_positions = np.linspace(y_min + agent_offset, y_max - agent_offset, num_agents)
    y_positions_reversed = y_positions[::-1]

    agent_states[:num_agents, 1] = y_positions
    # agent_states[4:, 1] = y_positions
    agent_states[:num_agents, 0] = x_min + agent_offset
    # agent_states[4:, 0] = x_max - agent_offset
    
    agent_goals[:num_agents, 1] = y_positions_reversed
    # agent_goals[4:, 1] = y_positions
    agent_goals[:num_agents, 0] = x_max - agent_offset
    # agent_goals[4:, 0] = x_min + agent_offset
    
    # Wall Representation
    border_points = generate_border()  # Assuming this is an Nx2 array of (x, y) points
    wall_points = generate_wall_hole(wall_x, hole_y_center, hole_height)  # Assuming this is also an Nx2 array

    # Wall Agents (for input to network)
    wall_agent_res = config_macbf.DIST_MIN_THRES * 1.5
    wall_y_positions = np.arange(y_min, y_max + wall_agent_res, wall_agent_res)
    wall_y_positions = wall_y_positions[
        (wall_y_positions < hole_y_min) | (wall_y_positions > hole_y_max)
    ]
    num_wall_agents = len(wall_y_positions)
    wall_agent_states = np.zeros((num_wall_agents, 4), dtype=np.float32)
    wall_agent_goals = np.zeros((num_wall_agents, 2), dtype=np.float32)
    wall_agent_states[:, 0] = wall_x
    wall_agent_states[:, 1] = wall_y_positions
    wall_agent_states[:, 2:] = 0.0
    wall_agent_goals[:, 0] = wall_x
    wall_agent_goals[:, 1] = wall_y_positions

    # Center of rotation
    origin = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2])

    # Rotate agent and wall agent positions around the origin
    agent_states[:, :2] = rotate_points(agent_states[:, :2], theta, origin)
    agent_goals = rotate_points(agent_goals, theta, origin)
    wall_agent_states[:, :2] = rotate_points(wall_agent_states[:, :2], theta, origin)
    wall_agent_goals = rotate_points(wall_agent_goals, theta, origin)
    
    # Rotate border and wall points around the origin
    border_points = rotate_points(border_points, theta, origin)
    wall_points = rotate_points(wall_points, theta, origin)

    return (
        agent_states,
        agent_goals,
        wall_agent_states,
        wall_agent_goals,
        border_points,
        wall_points,
    )


# Input: [num_agents, k, 6], 6: [del_x, del_y , del_vx, del_vy, identity, d_norm]
# Output: [num_agents, k, 1], 1: h
class CBFNetwork(nn.Module):
    def __init__(self):
        super(CBFNetwork, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, neighbors]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        h = self.conv4(x)
        h = h.permute(0, 2, 1)  # [batch, neighbors, 1]
        return h


# Input: [num_agents, k, 5], 5: [del_x, del_y , del_vx, del_vy, identity]
# Output: [num_agents, 2], 2: [a_x, a_y]
class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 + 4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, s, g, neighbor_features):
        x = neighbor_features.permute(0, 2, 1)  # [batch, features, neighbors]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Max over neighbors
        state_goal_diff = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
        x = torch.cat([x, state_goal_diff], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        gains = 2.0 * torch.sigmoid(self.fc4(x)) + 0.2
        k1, k2, k3, k4 = torch.chunk(gains, 4, dim=1)
        zeros = torch.zeros_like(k1)
        gain_x = -torch.cat([k1, zeros, k2, zeros], dim=1)
        gain_y = -torch.cat([zeros, k3, zeros, k4], dim=1)
        state = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
        a_x = torch.sum(state * gain_x, dim=1, keepdim=True)
        a_y = torch.sum(state * gain_y, dim=1, keepdim=True)
        a = torch.cat([a_x, a_y], dim=1)
        return a


'''
Dynamics:
    x_{k+1} = x_k + vx_k * T_s + 0.5 * ax_k * T_s^2
    y_{k+1} = y_k + vy_k * T_s + 0.5 * ay_k * T_s^2
    vx_{k+1} = vx_k + ax_k * T_s
    vy_{k+1} = vy_k + ay_k * T_s
'''
def dynamics_macbf(s, a):
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt


def take_step_obstacles(s, a, wall_agents=None):
    
    s_next = s + dynamics_macbf(s, a) * config_macbf.TIME_STEP
    # s_next[:, :2] = 0.5*a*config.TIME_STEP**2 + s_next[:, :2]

    neighbor_features_next, _ = compute_neighbor_features(
        s_next,
        config_macbf.DIST_MIN_THRES,
        config_macbf.TOP_K,
        wall_agents=wall_agents,
        include_d_norm=False,
        indices=None,
    )

    ttc_mask = ttc_dangerous_mask(
        config_macbf.DIST_MIN_CHECK,
        0,
        neighbor_features_next,
    )

    # Compute the mask
    mask = ttc_mask.any(dim=1)

    s_next = torch.where(ttc_mask.any(dim=1, keepdim=True), s, s_next)

    # Set the velocities [2:] of s_next where the mask is True
    s_next[mask, 2:] = 0  # or any other value you want to set

    return s_next


def ttc_dangerous_mask(r, ttc, neighbor_features):
    """
    s: agent states, shape [num_agents, 4]
    r: safe distance threshold
    ttc: time-to-collision threshold
    indices: indices of agent neighbors (from topk)
    neighbor_features: neighbor features including obstacles, shape [num_agents, k + num_obstacles, feature_dim]
    """
    # Extract features
    x = neighbor_features[..., 0]  # [num_agents, num_neighbors]
    y = neighbor_features[..., 1]
    vx = neighbor_features[..., 2]
    vy = neighbor_features[..., 3]
    eye = neighbor_features[..., 4]

    # Avoid self-interactions by adding eye to x and y
    x = x + eye
    y = y + eye

    # Compute quadratic equation coefficients
    alpha = vx**2 + vy**2
    beta = 2 * (x * vx + y * vy)
    gamma = x**2 + y**2 - r**2

    # Compute discriminant and handle negative values
    discriminant = beta**2 - 4 * alpha * gamma
    discriminant = torch.clamp(discriminant, min=0.0)

    dist_dangerous = gamma < 0

    has_two_positive_roots = (discriminant > 0) & (gamma > 0) & (beta < 0)
    sqrt_discriminant = torch.sqrt(discriminant)
    alpha_safe = torch.where(alpha == 0, torch.full_like(alpha, 1e-6), alpha)

    root1 = (-beta - sqrt_discriminant) / (2 * alpha_safe)
    root2 = (-beta + sqrt_discriminant) / (2 * alpha_safe)

    root_less_than_ttc = ((root1 > 0) & (root1 < ttc)) | ((root2 > 0) & (root2 < ttc))
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc

    return ttc_dangerous  # Shape: [num_agents, k + num_obstacles]


def barrier_loss(h, s, r, ttc, indices):
    neighbor_features_cbf, _ = compute_neighbor_features(
        s, config_macbf.DIST_MIN_THRES, config_macbf.TOP_K
    )
    mask = ttc_dangerous_mask(
        config_macbf.DIST_MIN_CHECK, ttc, neighbor_features_cbf
    )
    h = h.squeeze(2)  # If h has shape [num_agents, k, 1]
    h = h.view(-1)  # Shape: [num_agents * k]
    mask = mask.view(-1)  # Shape: [num_agents * k]
    dang_h = h[mask]
    safe_h = h[~mask]
    loss_dang = torch.mean(torch.relu(dang_h + 1e-3)) if dang_h.numel() > 0 else 0
    loss_safe = torch.mean(torch.relu(-safe_h)) if safe_h.numel() > 0 else 0
    acc_dang = torch.mean((dang_h <= 0).float()) if dang_h.numel() > 0 else -1.0
    acc_safe = torch.mean((safe_h > 0).float()) if safe_h.numel() > 0 else -1.0
    return loss_dang, loss_safe, acc_dang, acc_safe


# The action network learns to increase the value of the cbf
def derivative_loss(h, s, a, cbf_net, alpha, indices):
    s_next = s + dynamics_macbf(s, a) * config_macbf.TIME_STEP
    neighbor_features, _ = compute_neighbor_features(
        s_next,
        config_macbf.DIST_MIN_THRES,
        config_macbf.TOP_K,
        include_d_norm=True,
        indices=indices,
    )
    h_next = cbf_net(neighbor_features)
    deriv = (h_next - h) + config_macbf.TIME_STEP * alpha * h
    deriv = deriv.view(-1)
    neighbor_features_cbf, _ = compute_neighbor_features(
        s, config_macbf.DIST_MIN_THRES, config_macbf.TOP_K
    )
    dang_mask = ttc_dangerous_mask(
        config_macbf.DIST_MIN_CHECK, config_macbf.TIME_TO_COLLISION_CHECK, neighbor_features_cbf
    )
    dang_mask = dang_mask.view(-1)
    dang_deriv = deriv[dang_mask]
    safe_deriv = deriv[~dang_mask]
    loss_dang_deriv = (
        torch.mean(torch.relu(-dang_deriv + 1e-3)) if dang_deriv.numel() > 0 else 0
    )
    loss_safe_deriv = (
        torch.mean(torch.relu(-safe_deriv)) if safe_deriv.numel() > 0 else 0
    )
    acc_dang_deriv = (
        torch.mean((dang_deriv >= 0).float()) if dang_deriv.numel() > 0 else -1.0
    )
    acc_safe_deriv = (
        torch.mean((safe_deriv >= 0).float()) if safe_deriv.numel() > 0 else -1.0
    )
    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv


def action_loss(a, s, g, state_gain):
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    action_ref_norm = torch.sum(action_ref**2, dim=1)
    action_net_norm = torch.sum(a**2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss


# Never used
def action_loss_np(a, s, g):
    state_gain = torch.tensor(
        np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32
    ).to(s.device)
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    action_ref_norm = torch.sum(action_ref**2, dim=1)
    action_net_norm = torch.sum(a**2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss


def compute_neighbor_features(s, r, k, wall_agents=None, include_d_norm=False, indices=None):
    num_agents = s.size(0)
    s_diff_agents = s.unsqueeze(1) - s.unsqueeze(0)  # Shape: [num_agents, num_agents, 4]
    distances_agents = (torch.norm(s_diff_agents[:, :, :2], dim=2) + 1e-4)  # Shape: [num_agents, num_agents]
    eye = torch.eye(num_agents, device=s.device)
    distances_agents = distances_agents + eye * 1e6  # Remove this line

    # Handle Wall Agents
    if wall_agents is not None:
        s_diff_obstacles = s.unsqueeze(1) - wall_agents.unsqueeze(0)  # Shape: [num_agents, num_wall_agents, 4]
        distances_obstacles = (torch.norm(s_diff_obstacles[:, :, :2], dim=2) + 1e-4)  # Shape: [num_agents, num_wall_agents]
        distances = torch.cat([distances_agents, distances_obstacles], dim=1)
        s_diff = torch.cat([s_diff_agents, s_diff_obstacles], dim=1)
    else:
        distances = distances_agents
        s_diff = s_diff_agents
    
    # Select top k closest agents
    if indices is None:
        _, indices = torch.topk(-distances, k=k, dim=1)  # Negative for topk
    
    neighbor_features = s_diff[torch.arange(num_agents).unsqueeze(1), indices]
    
    # Create a type indicator: 1 for self-interaction, 0 otherwise
    agent_indices = torch.arange(num_agents).unsqueeze(1).to(s.device)
    type_indicator = (indices == agent_indices).unsqueeze(2).float()
    neighbor_features = torch.cat([neighbor_features, type_indicator], dim=2)
    
    if include_d_norm:
        d_norm_agents = (distances[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2) - r)
        neighbor_features = torch.cat([neighbor_features, d_norm_agents], dim=2)    
    # Return neighbor features and indices (indices correspond to agent neighbors)
    
    return neighbor_features, indices


"""
NOT USED
Adapt to create solid walls and obstacles
Incorporate SMG creation
"""

# Not Used
def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi * 2, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle


# Not used
def generate_obstacle_rectangle(center, sides, num=12):
    a, b = sides  # side lengths
    n_side_1 = int(num // 2 * a / (a + b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3

    # Define rectangle sides
    side_1 = np.concatenate(
        [
            np.linspace(-a / 2, a / 2, n_side_1, endpoint=False).reshape(-1, 1),
            b / 2 * np.ones(n_side_1).reshape(-1, 1),
        ],
        axis=1,
    )
    side_2 = np.concatenate(
        [
            a / 2 * np.ones(n_side_2).reshape(-1, 1),
            np.linspace(b / 2, -b / 2, n_side_2, endpoint=False).reshape(-1, 1),
        ],
        axis=1,
    )
    side_3 = np.concatenate(
        [
            np.linspace(a / 2, -a / 2, n_side_3, endpoint=False).reshape(-1, 1),
            -b / 2 * np.ones(n_side_3).reshape(-1, 1),
        ],
        axis=1,
    )
    side_4 = np.concatenate(
        [
            -a / 2 * np.ones(n_side_4).reshape(-1, 1),
            np.linspace(-b / 2, b / 2, n_side_4, endpoint=False).reshape(-1, 1),
        ],
        axis=1,
    )

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle


def ttc_dangerous_mask_legacy(s, r, ttc, indices):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    s_diff = s_diff[
        torch.arange(num_agents).unsqueeze(1), indices
    ]  # [num_agents, k, 4]
    x, y, vx, vy = (
        s_diff[..., 0],
        s_diff[..., 1],
        s_diff[..., 2],
        s_diff[..., 3],
    )  # [num_agents, k]

    # Avoid self-interactions by setting self-distances to a large value
    eye = torch.eye(num_agents, device=s.device)
    eye = eye[torch.arange(num_agents).unsqueeze(1), indices]
    x = x + eye
    y = y + eye
    alpha = vx**2 + vy**2
    beta = 2 * (x * vx + y * vy)
    gamma = x**2 + y**2 - r**2

    discriminant = beta**2 - 4 * alpha * gamma
    dist_dangerous = gamma < 0

    has_two_positive_roots = (discriminant > 0) & (gamma > 0) & (beta < 0)
    sqrt_discriminant = torch.sqrt(discriminant)
    root1 = (-beta - sqrt_discriminant) / (2 * alpha)
    root2 = (-beta + sqrt_discriminant) / (2 * alpha)
    root_less_than_ttc = ((root1 > 0) & (root1 < ttc)) | ((root2 > 0) & (root2 < ttc))
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc
    return ttc_dangerous  # Shape: [num_agents, k], k: True if dangerous


def compute_neighbor_features_legacy(s, r, k, include_d_norm=False, indices=None):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    distances = torch.norm(s_diff[:, :, :2], dim=2) + 1e-4  # [num_agents, num_agents]
    if indices is None:
        _, indices = torch.topk(-distances, k=k, dim=1)
    neighbor_features = s_diff[torch.arange(num_agents).unsqueeze(1), indices]
    eye = torch.eye(num_agents, device=s.device)
    eye = eye[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2)
    neighbor_features = torch.cat([neighbor_features, eye], dim=2)
    if include_d_norm:
        d_norm = (
            distances[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2) - r
        )
        neighbor_features = torch.cat([neighbor_features, d_norm], dim=2)
    return (
        neighbor_features,
        indices,
    )  # [num_agents, k, 6], [num_agents, k], 6 -> [del_x, del_y , del_vx, del_vy, identity, d_norm]
