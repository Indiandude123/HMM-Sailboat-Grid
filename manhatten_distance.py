import numpy as np
import json

# def mean_manhatten_distances(true_trajectories, predicted_trajectories):
#     mean_manhatten_distances_list = []
#     for t in range(true_trajectories.shape[1]):
#         time_step_mean_dist = np.mean(np.abs(true_trajectories[:, t, :] - predicted_trajectories[:, t, :]), axis = 0)
#         mean_manhatten_distances_list.append(list(time_step_mean_dist))
#     return mean_manhatten_distances_list

def manhattan_distance(predicted, true):
    return np.sum(np.abs(predicted - true), axis=-1)

def mean_manhattan_distance(predicted_trajectories, true_trajectories):
    predicted_trajectories = np.array(predicted_trajectories)
    true_trajectories = np.array(true_trajectories)
    if predicted_trajectories.shape != true_trajectories.shape:
        raise ValueError("Predicted and true states must have the same shape.")

    distances = manhattan_distance(predicted_trajectories, true_trajectories)
    mean_distances = np.mean(distances, axis=0)
    return list(mean_distances)

        
        
with open("sampled_trajectories.json", "r") as f:
    sampled_trajectories = json.load(f)

with open('decoded_trajectories.json', 'r') as f:
    decoded_trajectories = json.load(f)

sampled_traj = []
for item in sampled_trajectories.items():
    _, traj = item
    sampled_traj.append(traj)
# sampled_traj = np.array(sampled_traj)

decoded_traj = []
for item in decoded_trajectories.items():
    _, traj = item
    decoded_traj.append(traj)
# decoded_traj = np.decoded_traj)

mean_manhatten_distances = mean_manhattan_distance(decoded_traj, sampled_traj)

with open('mean_manhatten_distances.json', 'w') as f:
    json.dump(mean_manhatten_distances, f)


