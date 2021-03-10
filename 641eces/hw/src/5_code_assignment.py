# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Problem 1

# %%
import numpy as np
import itertools
from collections import Counter
import os, sys
from Bio import SeqIO


# %%
mers3_list = list(itertools.product('ACGT', repeat=3))
mers3_list = [''.join(list(seq)) for seq in mers3_list]
mers3_hash = list(itertools.product('0123', repeat=3))
base3_arr = np.array([16, 4, 1])
mers3_hash = [np.sum(np.array(list(map(int,h)))*base3_arr) for h in mers3_hash]

mers3_hash_dict = dict(zip(mers3_list, mers3_hash))


# %%
print("\n## Problem 1 ##\n")
print('Part A')
print(mers3_hash_dict['ATC'] + 1)


# %%
sliding_window = lambda seq, win_len, step: [seq[i:i+win_len] for i in np.arange(0, len(seq), step) if i + win_len <= len(seq)]


# %%
count_kmers = lambda seqs: dict(Counter(seqs))


# %%
print('Part B')
print(count_kmers(sliding_window('ATTATTGC', 3, 1)))

# %% [markdown]
# # Problem 2

# %%
A = np.array([
    [0.5, 0.1, 0.4],
    [0.2, 0.7, 0.1],
    [0.5, 0.3, 0.2],
], dtype=np.float64)
B = np.array([
    [0.5, 0.3, 0.2],
    [0.3, 0.2, 0.5],
    [0.1, 0.6, 0.3],
], dtype=np.float64)
I = np.array([1/3, 1/3, 1/3])
O = np.array([0, 2, 1, 2, 0])


# %%
def viterbi(trans_mat, emit_mat, init_state, obs, return_fm=False):
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

    most_likely_hs = [np.argmax(forward_mat[i]) for i in sorted(forward_mat.keys())]
    
    if return_fm:
        return most_likely_hs, forward_mat
    else:
        return most_likely_hs


# %%
hidden_states = viterbi(A, B, I, O)
print("\n## Problem 2 ##\n")
print(f"Hidden states = {hidden_states}")

# %% [markdown]
# # Problem 3

# %%
DATA_DIR = '../data'


# %%
with open(os.path.join(DATA_DIR, "T_vulcanus_rbcl.fasta"), "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
    t_vulcanus = ''.join(list(records[0].seq))

with open(os.path.join(DATA_DIR, "S_thermotolerans.fasta"), "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
    s_thermo = ''.join(list(records[0].seq))

with open(os.path.join(DATA_DIR, "Limnohabitans.fasta"), "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
    limnohabitans = ''.join(list(records[0].seq))

with open(os.path.join(DATA_DIR, "uncultured.fasta"), "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
    uncultured = ''.join(list(records[0].seq))


# %%
kmers = 3
step = 1
t_vulcanus_kmers_count = count_kmers(sliding_window(t_vulcanus, kmers, step))
s_thermo_kmers_count = count_kmers(sliding_window(s_thermo, kmers, step))
limnohabitans_kmers_count = count_kmers(sliding_window(limnohabitans, kmers, step))
uncultured_kmers_count = count_kmers(sliding_window(uncultured, kmers, step))


# %%
eps = 2e-26
t_vulcanus_kmers_arr = np.zeros((len(mers3_hash_dict), 1))
s_thermo_kmers_arr = np.zeros((len(mers3_hash_dict), 1))
limnohabitans_kmers_arr = np.zeros((len(mers3_hash_dict), 1))
uncultured_kmers_arr = np.zeros((len(mers3_hash_dict), 1))

for mer in t_vulcanus_kmers_count:
    t_vulcanus_kmers_arr[mers3_hash_dict[mer]] += t_vulcanus_kmers_count[mer]

for mer in s_thermo_kmers_count:
    s_thermo_kmers_arr[mers3_hash_dict[mer]] += s_thermo_kmers_count[mer]

for mer in limnohabitans_kmers_count:
    limnohabitans_kmers_arr[mers3_hash_dict[mer]] += limnohabitans_kmers_count[mer]

for mer in uncultured_kmers_count:
    uncultured_kmers_arr[mers3_hash_dict[mer]] += uncultured_kmers_count[mer]


# %%
t_vulcanus_llh = np.log(t_vulcanus_kmers_arr/np.sum(t_vulcanus_kmers_arr) + eps)
s_thermo_llh = np.log(s_thermo_kmers_arr/np.sum(s_thermo_kmers_arr) + eps)
limnohabitans_llh = np.log(limnohabitans_kmers_arr/np.sum(limnohabitans_kmers_arr) + eps)


# %%
LLH_t_vulcanus = np.sum(uncultured_kmers_arr * t_vulcanus_llh)
LLH_s_thermo = np.sum(uncultured_kmers_arr * s_thermo_llh)
LLH_limnohabitans = np.sum(uncultured_kmers_arr * limnohabitans_llh)


# %%
print("\n## Problem 3 ##\n")
print(f"L(t_vulcanus|x)={LLH_t_vulcanus},\nL(s_thermo|x)={LLH_s_thermo}\nL(limnohabitans|x)={LLH_limnohabitans}")


# %%
print(f"Highest LLH is {LLH_s_thermo}, belong to class S_thermotolerans")


