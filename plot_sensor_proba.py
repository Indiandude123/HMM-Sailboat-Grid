from grid import Grid
from hmm_model import HMM
import numpy as np

g = Grid()
hmm = HMM(15)
 

all_sensors_proba = hmm.get_B_list()
hidden_states = hmm.get_all_hidden_states()
num_states = hmm.get_num_hidden_states()
hidden_states_idx = hmm.create_hidden_state_index_map()

sensor_probab_grid = np.zeros((4, 15, 15))

for sensor_no in range(4):
    sensor_probas = all_sensors_proba[sensor_no]
    for i in range(num_states):
        x, y = hidden_states[i]
        sensor_probab_grid[sensor_no, x-1, y-1] = sensor_probas[i]
    g.plot_sensor_probabilities_heatmap(sensor_probab_grid[sensor_no], sensor_no, path = "plots/sensor_" + str(sensor_no+1) + "_probability" + ".png")
    g.clear()
        
        
