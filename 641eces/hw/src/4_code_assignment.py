# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Tai Duc Nguyen - ECES 641 - Programming Assignment 4

# %%
import numpy as np

# %% [markdown]
# # Part 1

# %%
A = np.array([
    [1/3, 2/3, 0],
    [1/3, 0, 2/3],
    [0, 2/3, 1/3],
], dtype=np.float64)
A


# %%
def get_prob_mm(mat, s0, st, steps):
    state_trans = np.zeros((1, mat.shape[0]))
    state_trans[0, s0] = 1
    for i in range(steps):
        state_trans = state_trans @ mat
    return state_trans[0, st]


# %%
print("\n## Problem 1 ##\n")
print(f"P(s1(2) | s3(0)) = {get_prob_mm(A, 2, 0, 2)}")
print(f"P(s1(10) | s3(0)) = {get_prob_mm(A, 2, 0, 10)}")
print(f"P(s1(50) | s3(0)) = {get_prob_mm(A, 2, 0, 50)}")
print(f"P(s1(100) | s3(0)) = {get_prob_mm(A, 2, 0, 100)}")

# %% [markdown]
# # Part 2

# %%
B = np.array([
    [9/10, 1/10],
    [1/10, 9/10],
    [1/2, 1/2],
], dtype=np.float64)
I = np.array([1, 0, 0])
O = np.array([0, 1, 1])


# %%
def get_prob_hmm(trans_mat, emit_mat, init_state, obs):
    forward_mat = {}
    n_state = len(init_state)
    max_timestep = len(obs)-1

    forward_mat[0] = np.array([0]*n_state, dtype=float)
    for i in range(n_state):
        forward_mat[0][i] = init_state[i] * emit_mat[i, obs[0]]

    def helper(time_step):
        if time_step in forward_mat:
            return forward_mat[time_step]

        forward_mat[time_step] = np.array([0]*n_state, dtype=float)
        prev_mat = helper(time_step-1)
        for i in range(n_state):
            sum_ = 0.0
            for j in range(n_state):
                sum_ += prev_mat[j] * trans_mat[i, j] * emit_mat[i, obs[time_step]]
            forward_mat[time_step][i] += sum_
        return forward_mat[time_step]
    
    helper(max_timestep)
    
    likelihood = 0
    for p in forward_mat[max_timestep]:
        likelihood += p
    return likelihood, forward_mat


# %%
likelihood, forward_mat = get_prob_hmm(A, B, I, O)
print("\n## Problem 2 ##\n")
print(f"P(O | I, A, B) = {likelihood}")
print(f"Forward matrix is {{timestep, P(state_i | prev_states)}} = {forward_mat}")


