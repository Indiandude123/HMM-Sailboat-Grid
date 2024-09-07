
# Hidden Markov Model Project

This project involves various implementations and analysis of Hidden Markov Models (HMMs). The project includes code for decoding, likelihood estimation, sampling sequences, and parameter learning for an HMM-based system. Additionally, scripts for plotting results, computing distances, and visualizing sensor probabilities are provided.

## Directory Structure

- **Code Files:**
  - `decoding.py`: Script for decoding sequences using the HMM.
  - `grid.py`: Defines the grid or state space for the model.
  - `hmm_model.py`: Core implementation of the Hidden Markov Model.
  - `learning_parameters.py`: Manages learning of parameters for the model.
  - `likelihood_estimation.py`: Estimation of likelihoods based on sequences.
  - `manhatten_distance.py`: Computes Manhattan distances between observations or trajectories.
  - `plot_avgKL.py`: Plots average KL divergences.
  - `plot_sensor_proba.py`: Plots sensor probabilities.
  - `plot.py`: Generic plotting script.
  - `sample_10k_sensor_obs.py`: Samples 10k sensor observations.
  - `sample_sequences.py`: Samples sequences for the HMM.
  
- **Data and Results:**
  - `avgKLDivergences_B_trainable.json`: Average KL divergences for a trainable observation matrix  model.
  - `avgKLDivergences_B_non_trainable.json`: KL divergences for a non-trainable observation matrix  model.
  - `sampled_trajectories.json`: Sampled trajectories data.
  - `sampled_observations.json`: Sampled observations data.
  - `likelihoods.json`: Likelihood estimates.

- **Plots:**
  - `avgKLDivergences_B_trainable_plot.png`: Plot for the average KL divergences of the trainable observation matrix model.
  - `avgKLDivergences_B_non_trainable_plot.png`: Plot for the average KL divergences of the non-trainable observation matrix model.
  - Other plots are stored in the `plots/` directory.

- **Documentation:**
  - `AIL722_A1-5.pdf`: Assignment description.
  - `report.pdf`: A report detailing the project methodology and results.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Install the necessary Python dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Scripts**:
   You can run the different scripts based on your needs:
   - To decode sequences:
     ```bash
     python decoding.py
     ```
   - To estimate likelihoods:
     ```bash
     python likelihood_estimation.py
     ```
   - To sample sequences:
     ```bash
     python sample_sequences.py
     ```

4. **Generate Plots**:
   To generate various plots for analysis, run:
   ```bash
   python plot_avgKL.py
   python plot_sensor_proba.py
   ```

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Authors
- **Anamitra** (Primary Contributor)

## License
This project is licensed under the MIT License.
