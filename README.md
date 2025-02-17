# Parallelizing Dyna-Q Learning: Accelerating Reinforcement Learning Through Networked Architectures

PDQN (Parallelized Dyna-Q Network) is a model-based reinforcement learning algorithm that combines the efficiency of Dyna-Q learning with the parallel processing capabilities of the Parallelized Q-Network (PQN) architecture [link to paper/repo].  Dyna-Q leverages a learned model of the environment to perform planning, which can significantly accelerate learning. PDQN extends PQN by incorporating this planning component within a distributed GPU setting, further boosting performance.  This implementation maintains PQN's simplicity, performance, and structural integrity.  We compare PDQN's performance against PQN within the MinAtar environment.

## 🚀 Usage (highly recommended with Docker)

Steps:

1. Ensure you have Docker and the NVIDIA Container Toolkit installed.  See the installation guide: [NVIDIA Container Toolkit link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. (Optional) Set your WANDB key in the `docker/Dockerfile`.  This is recommended for experiment tracking.
3. Build the Docker image: `bash docker/build.sh`
4. Run a Docker container: `bash docker/run.sh`
5. Run a training script: `python purejaxql/pdqn_minatar.py +alg=pdqn_minatar`.  The `+alg=pdqn_minatar` argument specifies the PDQN algorithm.

## Experiment Configuration

The `tune` function can be used for hyperparameter search.  See the configuration file for PDQN-specific hyperparameter values.

## Experiment Results

Our experiments on the MinAtar environment demonstrate that PDQN slightly outperforms PQN in terms of final performance and seems to stabilize learning. More environments need to be tested in order to confirm or deny current findings.
