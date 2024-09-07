from hmm_model import HMM
import json
import numpy as np


with open("sampled_10k_observations.json", "r") as f:
    sampled_10k_obs = json.load(f)

obs_seq_list = []
for item in sampled_10k_obs.items():
    _, obs_seq = item
    obs_seq_list.append(obs_seq)
    
hmm = HMM(15)

B_trainable = True
T, B, rho, avgKL_T_list, avgKL__B_list = hmm.baum_welch(obs=np.array(obs_seq_list)[:3000], n_iter=20, B_trainable=B_trainable)

if B_trainable:
    avgKLDivergences_B_trainable = {
        "AvgKLDivergenceList-T matrix" : avgKL_T_list,
        "AvgKLDivergenceList-B matrix" : avgKL__B_list
    }

    with open("avgKLDivergences_B_trainable.json", "w") as f:
        json.dump(avgKLDivergences_B_trainable, f)
else:
    avgKLDivergences_B_non_trainable = {
        "AvgKLDivergenceList-T matrix" : avgKL_T_list,
        "AvgKLDivergenceList-B matrix" : avgKL__B_list
    }

    with open("avgKLDivergences_B_non_trainable.json", "w") as f:
        json.dump(avgKLDivergences_B_non_trainable, f)