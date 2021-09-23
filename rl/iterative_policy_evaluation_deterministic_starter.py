import numpy as np
from grid_world_starter import standard_grid, ACTION_SPACE


def print_policy(policy, grid):
    for row in range(grid.rows):
        print('\n-----------------')
        for col in range(grid.cols):
            a = policy.get((row, col), ' ')
            print('  %s ' % a, end='')

def print_values(V, g):
    for i in range(g.rows):
        print("---------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")

if __name__ == '__main__':
    grid = standard_grid()

    policy = {
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 0): 'U',
        (2, 0): 'U',
        (2, 1): 'R',
        (2, 2): 'U',
        (2, 3): 'L',
        (1, 2): 'U',
    }

    print_policy(policy=policy, grid=grid)

    rewards = {}

    transition_probs = {}

    # Khởi tạo reward và xác suất chuyển dựa vào state1 ,state2 và action cho các ô trong grid
    for i in range(grid.rows):
        for j in range(grid.cols):
            s1 = (i, j)
            if not grid.is_terminal(s1):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s1, a)
                    transition_probs[(s1, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s1, a, s2)] = grid.rewards.get(s2)

    print(transition_probs)
    # Khởi tạo V cho các ô
    V = {}
    for state in grid.all_states():
        V[state] = 0

    # Biến min check equal
    MIN_CHECK_EQUAL = 1e-3
    # Tiến hành chạy cập nhật V
    gamma = 0.9  # discount factor

    # repeat until convergence
    it = 0
    while True:
        max_change = 0
        for state in grid.all_states():
            old_value = V[state]
            new_value = 0
            if not grid.is_terminal(state):
                for action in ACTION_SPACE:
                    for next_state in grid.all_states():
                        # action prob
                        action_prob = 1 if policy.get(state) == action else 0

                        # Tính value
                        reward = rewards.get((state, action, next_state), 0)
                        new_value += action_prob * transition_probs.get((state, action, next_state) , 0) * (
                                    reward + gamma * V[next_state])

                V[state] = new_value
                max_change = max(max_change, np.abs(new_value - old_value))

        print("iter:", it, "biggest_change:", max_change)
        print_values(V, grid)
        it += 1

        if max_change < MIN_CHECK_EQUAL:
            break
    print("\n\n")
