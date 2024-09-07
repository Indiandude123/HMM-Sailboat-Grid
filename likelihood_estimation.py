from hmm_model import HMM
import json

hmm = HMM(15)

with open("sampled_observations.json", "r") as f:
    sampled_obs = json.load(f)

observation_likelihoods = {}
for item in sampled_obs.items():
    trajectory_no, obs_seq = item
    obs_likelihood = hmm.forward_inference(obs_seq)
    observation_likelihoods[trajectory_no] = obs_likelihood

with open('likelihoods.json', 'w') as f:
    json.dump(observation_likelihoods, f)