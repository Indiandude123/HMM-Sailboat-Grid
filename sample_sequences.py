from hmm_model import HMM
import json


hmm = HMM(N=15)
sampled_trajectories = {}
for i in range(20):
    trajectory_i = hmm.sample_hidden_state_sequence(t=30, seed=i)
    sampled_trajectories[i] = trajectory_i
    
    
with open('sampled_trajectories.json', "w") as f:
    json.dump(sampled_trajectories, f)


sampled_observations = {}
for i in range(20):
    observation_seq_i = hmm.sample_observation_state_sequence(sampled_trajectories[i])
    sampled_observations[i] = observation_seq_i
    
with open('sampled_observations.json', "w") as f:
    json.dump(sampled_observations, f)