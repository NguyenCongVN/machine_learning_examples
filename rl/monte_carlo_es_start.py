import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid, ACTION_SPACE
from iterative_policy_evaluation import print_values, print_policy


def play_game(policy, grid, max_step=20):
    # Khởi tạo vị trí ban đầu bất kì
    states = list(grid.actions.keys())
    start_index = np.random.choice(len(states))
    start_state = states[start_index]

    # set vị trí hiện tại
    grid.set_state(start_state)
    s = grid.current_state()

    # Khởi tạo arr chứa kết quả
    a = np.random.choice(ACTION_SPACE)
    state_action_reward = [(s, a, 0)]

    # Tiến hành chạy theo policy và lấy lại state,action,reward
    for _ in range(max_step):
        r = grid.move(a)
        s = grid.current_state()

        if grid.game_over():
            state_action_reward.append((s, None, r))
            break
        else:
            a = policy[s]
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

    # Khởi tạo Q với state s và action a nếu như là terminal thì không khởi tạo
    Q = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0
        else:
            pass
    print(Q)
    #
    deltas = []
    TIME_PLAY = 10000
    LEARNING_RATE = 0.1
    for i in range(TIME_PLAY):
        # nhận lại state , action , và giá trị return sau mỗi lần chơi
        states_action_return = play_game(policy, grid)

        # khởi tạo danh sách state đã gặp
        seen_state_action_pair = set()

        # Khởi tạo Biggest change
        biggest_change = 0
        for s, a, G in states_action_return:
            sa = (s, a)
            if sa not in seen_state_action_pair:
                seen_state_action_pair.add(sa)
                old_Q = Q[s][a]
                Q[s][a] = old_Q + LEARNING_RATE * (G - old_Q)
                biggest_change = max(biggest_change, np.abs(Q[s][a] - old_Q))
        deltas.append(biggest_change)

        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print("final policy:")
    print_policy(policy, grid)
