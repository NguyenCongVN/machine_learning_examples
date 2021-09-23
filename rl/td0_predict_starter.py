import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, ACTION_SPACE
from iterative_policy_evaluation import print_values, print_policy


def random_action(a, eps=0.1):
    p = np.random.random()
    if p > eps:
        return a
    else:
        return np.random.choice(ACTION_SPACE)


def play_game(policy, grid, eps, max_step=20):
    # Khởi tạo vị trí ban đầu cố định. Thay vì bất kì thì sử dụng epsilon-greedy để tiến hành explore
    start_state = (2, 0)

    # set vị trí hiện tại
    grid.set_state(start_state)
    s = grid.current_state()

    # Khởi tạo arr chứa kết quả -> chọn random action ban đầu bằng cách cho eps = 1
    a = random_action(policy[s], eps)
    state_action_reward = [(s, a, 0)]

    # Tiến hành chạy theo policy và lấy lại state,action,reward
    for _ in range(max_step):
        r = grid.move(a)
        s = grid.current_state()

        if grid.game_over():
            state_action_reward.append((s, None, r))
            break
        else:
            a = random_action(policy[s])
            state_action_reward.append((s, a, r))

    # Tính return value
    G = 0
    gamma = 0.9
    state_action_return = []
    isFirstTime = True
    for s, a, r in reversed(state_action_reward):
        if isFirstTime:
            isFirstTime = False
        else:
            state_action_return.append((s, a, G))
        G = r + gamma * G
    state_action_return.reverse()
    return state_action_return


def max_dict(d):
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key, max_value


if __name__ == '__main__':
    # khởi tạo grid
    grid = standard_grid()

    # Khởi tạo random policy
    policy = {}
    for state in grid.actions.keys():
        policy[state] = np.random.choice(ACTION_SPACE)
    print('Random policy :')
    print_policy(policy, grid)

    # Khởi tạo V với state s nếu như là terminal thì không khởi tạo
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0
    #
    deltas = []
    TIME_PLAY = 1000
    LEARNING_RATE = 0.1
    eps = 1
    gamma = 0.9
    for i in range(TIME_PLAY):
        # reset lại sau mỗi lần play
        # Khởi tạo vị trí ban đầu cố định. Thay vì bất kì thì sử dụng epsilon-greedy để tiến hành explore
        start_state = (2, 0)

        # set vị trí hiện tại
        grid.set_state(start_state)

        s = grid.current_state()

        delta = 0
        while not grid.game_over():
            a = random_action(policy[s])
            r = grid.move(a)
            s_next = grid.current_state()
            old_V = V[s]
            V[s] = V[s] + LEARNING_RATE * (r + gamma * V[s_next] - V[s])
            delta = max(delta, np.abs(V[s] - old_V))

            s = s_next
        deltas.append(delta)
    plt.plot(deltas)
    plt.show()
    print_values(V, grid)
