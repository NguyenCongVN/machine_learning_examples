from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic_starter import print_policy, print_values
import numpy as np


def PlayGame(grid, policy, max_steps=20):
    # xuất phát tại vị trí random trong các state trong policy
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    # Gán giá trị khởi tạo cho vị trí bắt đầu
    start_pos = grid.current_state()
    state_rewards = [(start_pos, 0)]

    # Move theo policy và nhận lại reward cho đến khi game over
    steps = 0
    while not grid.game_over():
        a = policy[start_pos]
        r = grid.move(a)
        start_pos = grid.current_state()
        state_rewards.append((start_pos, r))

        # Tính và giới hạn số step
        steps += 1
        if steps >= max_steps:
            break

    # Tính giá trị return G và trả lại giá trị
    # Khởi tạo G
    G = 0
    state_returns = []
    # Kiểm tra lần đầu tính vì bỏ qua terminal state đầu tiên
    isFirstTime = True
    gamma = 0.9
    for state, reward in reversed(state_rewards):
        if isFirstTime:
            isFirstTime = False
        else:
            state_returns.append((state, G))
        G = reward + gamma * G
    state_returns.reverse()
    return state_returns


if __name__ == '__main__':
    # Tạo môi trường
    grid = standard_grid()

    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # Tạo policy
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    print_policy(policy=policy, grid=grid)

    # Khởi tạo Value , state và return
    V = {}
    s = grid.all_states()
    r = {}
    # Khởi tạo giá trị cho return hoặc gán 0 cho V nếu như state là terminal hoặc không thể đến được state bằng action
    for state in s:
        if state not in grid.actions:
            V[state] = 0
        else:
            r[state] = []

    TIME_PLAY = 100
    for i in range(TIME_PLAY):
        # Chơi và nhận lại giá trị return theo state
        stateAndReturn = PlayGame(grid, policy)

        # Tính giá trị sample mean cho các giá trị return
        seenState = set()
        for s, returnValue in stateAndReturn:
            if s not in seenState:
                seenState.add(s)
                r[s].append(returnValue)
                V[s] = np.mean(r[s])

    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
