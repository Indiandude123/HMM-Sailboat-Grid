from grid import Grid
import json

g = Grid()

with open("sampled_trajectories.json", "r") as f:
    sampled_trajectories = json.load(f)
    
# for i in range(20):
#     trajectory_i =  sampled_trajectories[str(i)]
#     g.draw_path(sequence=trajectory_i)
#     g.show(path = "plots/sampled_trajectory" + str(i) + ".png")
#     g.clear()
    
with open("decoded_trajectories.json", "r") as f:
    decoded_trajectories = json.load(f)


for i in range(20):
    strajectory_i = sampled_trajectories[str(i)]
    dtrajectory_i = decoded_trajectories[str(i)]
    g.draw_path(sequence=strajectory_i)
    g.draw_path(sequence=dtrajectory_i)
    g.show(path = "plots/trajectory_comparison" + str(i) + ".png")
    g.clear()