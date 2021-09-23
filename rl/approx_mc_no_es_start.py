import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, ACTION_SPACE
from iterative_policy_evaluation import print_values, print_policy


def s2x(s):
    return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])


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
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    print_policy(policy, grid)

    #
    deltas = []
    TIME_PLAY = 10000
    LEARNING_RATE = 0.1
    eps = 1
    theta = np.random.randn(4) / 2
    for i in range(TIME_PLAY):
        # nhận lại state , action , và giá trị return sau mỗi lần chơi
        states_action_return = play_game(policy, grid, eps)

        # khởi tạo danh sách state đã gặp
        seen_state_action_pair = set()

        # Khởi tạo Biggest change
        biggest_change = 0
        for s, a, G in states_action_return:
            sa = (s, a)
            if sa not in seen_state_action_pair:
                seen_state_action_pair.add(sa)
                old_theta = theta.copy()
                x = s2x(s)
                theta = theta + LEARNING_RATE * (G - theta.dot(x)) * x
                print(G)
                biggest_change = max(biggest_change, np.abs(theta - old_theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    V = {}
    for s in grid.all_states():
        V[s] = theta.dot(s2x(s))

    print_values(V, grid)
