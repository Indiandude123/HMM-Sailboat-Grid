import matplotlib.pyplot as plt
import json 


# Data
iterations = list(range(1, 21))  # Assuming there are 20 iterations

# with open("avgKLDivergences_B_non_trainable.json", "r") as f:
#     avgKLDivergences_B_non_trainable = json.load(f)
    
with open("avgKLDivergences_B_trainable.json", "r") as f:
    avgKLDivergences_B_non_trainable = json.load(f)

avg_kl_div_t = avgKLDivergences_B_non_trainable["AvgKLDivergenceList-T matrix"]

avg_kl_div_b = avgKLDivergences_B_non_trainable["AvgKLDivergenceList-B matrix"]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, avg_kl_div_t, label='Avg KL Divergence - T matrix', marker='o')
plt.plot(iterations, avg_kl_div_b, label='Avg KL Divergence - B matrix', marker='o')
plt.xlabel('Iteration')
plt.ylabel('Average KL Divergence')
plt.title('Average KL Divergence for Transition and Emission Matrices')
plt.legend()
plt.grid(True)
plt.show()
