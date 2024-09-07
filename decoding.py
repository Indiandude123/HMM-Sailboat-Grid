from hmm_model import HMM
import json

hmm = HMM(15)

with open("sampled_observations.json", "r") as f:
    sampled_obs = json.load(f)

decoded_hidden_state_seqs = {}
for item in sampled_obs.items():
    trajectory_no, obs_seq = item
    decoded_hidden_state_sequence = hmm.viterbi(obs_seq)
    decoded_hidden_state_seqs[trajectory_no] = decoded_hidden_state_sequence
    
with open('decoded_trajectories.json', 'w') as f:
    json.dump(decoded_hidden_state_seqs, f)