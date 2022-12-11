import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random


class Environment:
    def __init__(self, world_map):
        self.world_h = len(world_map)
        self.world_w = len(world_map[0])
        self.cargos = dict()
        self.desired_space = []
        # change r in a world map to -1
        modified_world = []
        for row in world_map:
            modified_row = []
            for c in row:
                if c == "r":
                    modified_row.append(-1)
                else:
                    modified_row.append(int(c))
            modified_world.append(modified_row)
        self.world = modified_world
        self.world = np.array(self.world)

        for i in range(len(world_map)):
            for j in range(len(world_map[0])):
                if world_map[i][j] == "r":
                    self.desired_space.append((i, j))
                if world_map[i][j].isdigit() and int(world_map[i][j]) != 0:
                    if world_map[i][j] not in self.cargos:
                        self.cargos[world_map[i][j]] = []
                    self.cargos[world_map[i][j]].append((i, j))
        for key in self.cargos:
            self.cargos[key] = np.array(self.cargos[key])

        self.desired_space = np.array(self.desired_space)

        self.init_cargos = self.cargos.copy()
        self.init_desired_space = self.desired_space.copy()
        self.init_world = self.world.copy()

    def move_cargo(self, cargo, action):
        if action == "D":
            new_pos = self.cargos[cargo] + np.array([1, 0])
        elif action == "U":
            new_pos = self.cargos[cargo] + np.array([-1, 0])
        elif action == "R":
            new_pos = self.cargos[cargo] + np.array([0, 1])
        elif action == "L":
            new_pos = self.cargos[cargo] + np.array([0, -1])
        else:
            raise ValueError("Action must be one of D, U, R, L")

        if np.any(new_pos < 0) or np.any(
            new_pos >= np.array([self.world_h, self.world_w])
        ):
            raise ValueError("New position is out of bounds")
        self.world[self.cargos[cargo][:, 0], self.cargos[cargo][:, 1]] = 0
        for i, j in self.desired_space:
            self.world[i, j] = -1
        self.world[new_pos[:, 0], new_pos[:, 1]] = int(cargo)
        self.cargos[cargo] = new_pos

    def check_move(self, cargo, action):
        if action == "D":
            new_pos = self.cargos[cargo] + np.array([1, 0])
        elif action == "U":
            new_pos = self.cargos[cargo] + np.array([-1, 0])
        elif action == "R":
            new_pos = self.cargos[cargo] + np.array([0, 1])
        elif action == "L":
            new_pos = self.cargos[cargo] + np.array([0, -1])
        else:
            raise ValueError("Action must be one of D, U, R, L")

        if np.any(new_pos < 0) or np.any(
            new_pos >= np.array([self.world_h, self.world_w])
        ):
            return False
        return True

    def world_visualisation(self):
        world = [[0 for _ in range(self.world_w)] for _ in range(self.world_h)]
        for space in self.desired_space:
            world[space[0]][space[1]] = "r"

        for key in self.cargos:
            for cargo_cell in self.cargos[key]:
                world[cargo_cell[0]][cargo_cell[1]] = key

        for i in range(len(world)):
            for j in range(len(world[0])):
                print(world[i][j], end=" ")
            print()

    def goal_distance(self, cargo):
        distances = []
        ids = []
        points = []
        for cargo_y, cargo_x in self.cargos[cargo]:
            min_y = 1e8
            min_x = 1e8
            id_y = -1
            id_x = -1
            min_cargo_point = (-1, -1)

            for des_y, des_x in self.desired_space:
                if abs(des_y - cargo_y) + abs(des_x - cargo_x) < min_y + min_x:
                    min_y = abs(des_y - cargo_y)
                    min_x = abs(des_x - cargo_x)
                    id_y = des_y
                    id_x = des_x
                    min_cargo_point = (cargo_y, cargo_x)
            distances.append(min_y + min_x)
            ids.append((id_y, id_x))
            points.append(min_cargo_point)
        min_id = np.argmax(distances)
        return distances[min_id], *ids[min_id], *points[min_id]

    def reset(self):
        self.cargos = self.init_cargos.copy()
        self.desired_space = self.init_desired_space.copy()
        self.world = self.init_world.copy()

    def get_cargo_overlaps(self):
        total_cells = []
        for cargo in self.cargos:
            for cell in self.cargos[cargo]:
                total_cells.append(cell)
        total_cells = np.array(total_cells)
        unique, counts = np.unique(total_cells, axis=0, return_counts=True)
        overlaped_cells = np.sum(counts > 1)
        return overlaped_cells

    def get_cargo_overlaps_with_desired(self):
        ret = 0
        for des_cell in self.desired_space:
            for cargo in self.cargos:
                for cargo_cell in self.cargos[cargo]:
                    if des_cell[0] == cargo_cell[0] and des_cell[1] == cargo_cell[1]:
                        ret += 1
        return ret

    def is_done(self):
        reward = 0
        reward -= self.get_cargo_overlaps()
        reward += self.get_cargo_overlaps_with_desired()
        cargo_cells = 0
        for cargo in self.cargos:
            cargo_cells += len(self.cargos[cargo])
        return reward == cargo_cells


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


MAX_STEPS = 1000

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = 4

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)

steps_done = 0


def select_action(state):
    global steps_done
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        # print("policy")
        with torch.no_grad():
            return (policy_net(state.to(device)).max(1)[1].view(1, 1)).to("cpu")

    else:
        # print("random")
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def get_state(env, cargo=None):
    state = env.world
    state = np.array(state)
    if cargo is not None:
        for other_cargo in env.cargos:
            if other_cargo != cargo:
                state[env.cargos[other_cargo][:, 0], env.cargos[other_cargo][:, 1]] = -2
    state = state.reshape(1, 1, env.world_h, env.world_w)
    state = torch.from_numpy(state)
    state = state.float()
    return state


def get_reward(env, possible):
    reward = 0
    reward += env.get_cargo_overlaps_with_desired()
    reward -= env.get_cargo_overlaps()
    reward -= 1
    if not possible:
        reward = -100
    return torch.tensor([reward], dtype=torch.float)


def get_cargo_state(env, state, cargo):
    state = state.detach().numpy()
    state = state.reshape(env.world_h, env.world_w)
    # print(state)
    for other_cargo in env.cargos:
        if other_cargo != cargo:
            state[env.cargos[other_cargo][:, 0], env.cargos[other_cargo][:, 1]] = -2
        else:
            state[env.cargos[other_cargo][:, 0], env.cargos[other_cargo][:, 1]] = 1
    new_state = state.reshape(1, 1, env.world_h, env.world_w)
    new_state = torch.from_numpy(new_state)
    new_state = new_state.float()
    return new_state


memory = ReplayMemory(10000)
optimizer = optim.Adam(policy_net.parameters())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE).to(device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states.to(device)).max(1)[0].detach()
    )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_cargo_cells(environment):
    cells = 0
    for cargo in environment.cargos:
        cells += len(environment.cargos[cargo])
    return cells


actions_list = ["D", "U", "R", "L"]

total_train_rewards = []
episode_results = []
results = {"failed": 0, "success": 0}


def train(env, n_episodes=1000):

    for i_episode in range(n_episodes):
        env.reset()
        init_cargo_pos = env.cargos.copy()
        state = get_state(env)
        cargo_paths = dict()
        episode_reward_sum = 0
        steps = 0
        illegal_moves = 0
        while True:

            if steps > MAX_STEPS:
                results["failed"] += 1
                break

            for cargo in env.cargos.keys():
                steps += 1
                cargo_state = get_cargo_state(env, state, cargo)
                action = select_action(cargo_state)
                is_possible_move = env.check_move(cargo, actions_list[action])

                if not cargo_paths.get(cargo):
                    cargo_paths[cargo] = []

                cargo_paths[cargo].append(actions_list[action])
                if is_possible_move:
                    env.move_cargo(cargo, actions_list[action])
                else:
                    illegal_moves += 1

                reward = get_reward(env, is_possible_move)
                if env.is_done():
                    reward += env.world_h * env.world_w * 10 * get_cargo_cells(env)

                next_state = get_state(env)
                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model()
                episode_reward_sum += reward.item()
                # cargo_paths.append(path)
                if env.is_done():
                    break

            if env.is_done():
                results["success"] += 1
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print("Episode: {}, Reward: {}".format(i_episode, episode_reward_sum))
        total_train_rewards.append(episode_reward_sum)
        episode_results.append(env.is_done())
        print("Is done: {}".format(env.is_done()))
        print("Cargo positions: ")
        for cargo in env.cargos.keys():
            print(cargo, end=": ")
            for pos in env.cargos[cargo]:
                print(pos, end=" ")
            print()
        print("Initial cargo positions: ")
        for cargo in init_cargo_pos.keys():
            print(cargo, end=": ")
            for pos in init_cargo_pos[cargo]:
                print(pos, end=" ")
            print()
        print(f"steps performed: {steps}")
        print(f"illegal moves: {illegal_moves}")
        # print(f'cargo paths: {cargo_paths}')
        print("-" * 40)


def test(env, n_episodes=1, output_file="output.txt"):
    cargo_paths = dict()
    for _ in range(n_episodes):
        env.reset()
        state = get_state(env)
        cargo_paths = dict()
        episode_reward_sum = 0
        steps = 0
        total_test_reward = 0
        while True:

            if steps > MAX_STEPS:
                break

            for cargo in env.cargos.keys():
                steps += 1
                cargo_state = get_cargo_state(env, state, cargo)
                action = (
                    policy_net(cargo_state.to(device)).max(1)[1].view(1, 1).to("cpu")
                )
                is_possible_move = env.check_move(cargo, actions_list[action])

                if not cargo_paths.get(cargo):
                    cargo_paths[cargo] = []

                cargo_paths[cargo].append(actions_list[action])
                if is_possible_move:
                    env.move_cargo(cargo, actions_list[action])
                else:
                    pass

                reward = get_reward(env, is_possible_move)
                if env.is_done():
                    reward += 1000

                episode_reward_sum += reward.item()
                total_test_reward += reward.item()
                next_state = get_state(env)
                state = next_state
                if env.is_done():
                    break

            if steps % 100 == 0:
                file = open(output_file, "a")
                max_len = max([len(cargo_paths[cargo]) for cargo in cargo_paths.keys()])
                for id in range(max_len):
                    for cargo in cargo_paths.keys():
                        if id < len(cargo_paths[cargo]):
                            file.write(cargo + ": " + cargo_paths[cargo][id] + "\n")
                cargo_paths = dict()

            if env.is_done():
                break

        if len(cargo_paths.keys()) != 0:
            file = open(output_file, "a")
            max_len = max([len(cargo_paths[cargo]) for cargo in cargo_paths.keys()])
            for id in range(max_len):
                for cargo in cargo_paths.keys():
                    if id < len(cargo_paths[cargo]):
                        file.write(cargo + ": " + cargo_paths[cargo][id] + "\n")
            cargo_paths = dict()

        total_train_rewards.append(episode_reward_sum)
        episode_results.append(env.is_done())
        print("Is done: {}".format(env.is_done()))
        print("Reward: {}".format(total_test_reward))
        print("Cargo positions: ")
        for cargo in env.cargos.keys():
            print(cargo, end=": ")
            for pos in env.cargos[cargo]:
                print(pos, end=" ")
            print()
        print(f"steps performed: {steps}")
        print(f"cargo paths: {cargo_paths}")
        env.world_visualisation()
        print("-" * 40)
    return cargo_paths


def complete_task(input_file, output_file):
    global MAX_STEPS
    file = open(input_file, "r")
    world_map = []
    for line in file:
        line = line.strip()
        line = line.split()
        world_map.append(line)
    file.close()
    env = Environment(world_map)
    MAX_STEPS = env.world_h * env.world_w * 2
    train(env, n_episodes=1000)
    test(env, n_episodes=1, output_file=output_file)


if __name__ == "__main__":
    complete_task("input.txt", "output.txt")
