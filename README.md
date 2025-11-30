# Vision-Integrated Humanoid Locomotion using PPO

**Authors:** Chitraksh V, V Akhil, Nithish Goud S

This repository houses a comprehensive framework for training a MuJoCo Humanoid agent to achieve stable bipedal locomotion. By bridging **Computer Vision** with **Deep Reinforcement Learning (DRL)**, this project moves beyond random initialization. We implement a pipeline that extracts skeletal posture from a 2D image of a human and translates it into the 3D joint space of a robot, allowing the agent to "mimic" a starting pose before initiating its walking gait via Proximal Policy Optimization (PPO).

## 📊 Visual Results

### Gait Demonstration

Below is the agent recovering from the initialized pose and establishing a stable walking rhythm.

> _High-fidelity video available at `/videos/demo.mp4`_

## 🚀 Key Features

- **Proximal Policy Optimization (PPO):** We utilize a continuous action-space PPO agent trained on the `Humanoid-v5` Gymnasium environment to master balance and locomotion.
- **Vision-Based Kinematics:** A custom pipeline processes static images to extract 33 distinct landmarks, converting them into 25 mechanical joint angles compatible with the MuJoCo physics engine.
- **Active Postural Stabilization:** To counter the "backward lean" often observed in early RL training, we implement a dynamic stabilization force applied to the torso, proportional to hip actuation.
- **Pre-Trained Checkpoints:** The repository includes `model.pt`, a policy network trained for over 1000 epochs, ready for immediate inference.

## 📂 Project Architecture

Based on the file structure, the project is organized as follows:

```plaintext
Humanoid_lib/
├── checkpoints/             # Storage for model weights saved during training
├── lib/                     # Core RL implementation
│   ├── DQN_Agent.py         # Agent class containing Actor-Critic networks
│   ├── buffer_DQN.py        # Replay buffer for trajectory storage & GAE calculation
│   └── utils.py             # Environment wrappers and logging utilities
├── logs/                    # TensorBoard event logs
├── pose_estimation/         # Computer Vision Pipeline
│   ├── imageprocessing.py   # Image resizing and normalization
│   ├── keypoints.py         # MediaPipe extraction & skeleton overlay logic
│   └── Kinematic.py         # Mathematics for 2D->3D joint angle conversion
├── videos/                  # Output renderings of the agent
├── get_joint_data.py        # Debug tool for inspecting Mujoco joint limits
├── test_ppo.py              # INFERENCE: Runs the visualizer with the trained model
├── train_ppo.py             # TRAINING: Starts the PPO training loop
├── req.txt                  # Dependency list
└── model.pt                 # Default pre-trained model
```

## 🛠️ Installation & Setup

**Prerequisites:** Python 3.12 is recommended to ensure compatibility with MediaPipe and PyTorch.

1.  **Clone and Navigate**
    Ensure you are in the root directory of the project (`Humanoid_lib`).

2.  **Environment Setup**

    ```bash
    # Create virtual environment
    python3.12 -m venv venv

    # Activate environment
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r req.txt
    ```

## 💻 Usage Instructions

### 1\. Inference (Running the Demo)

To see the robot mimic a pose and walk using the pre-trained weights:

1.  Place your target image (e.g., `image.jpg`) in the root folder.
2.  Run the test script:
    ```bash
    python test_ppo.py
    ```
3.  Enter the image filename when prompted. The simulation window will launch in fullscreen.

### 2\. Training from Scratch

To train a new agent using the PPO algorithm:

1.  Execute the training script:
    ```bash
    python train_ppo.py
    ```
2.  **Optional Arguments:** You can modify hyperparameters directly in the script or add command-line parsers (e.g., `--n-epochs=2000`).
3.  **Monitoring:** Track loss metrics and reward curves via TensorBoard:
    ```bash
    tensorboard --logdir=logs
    ```

## 🧠 Technical Modules

### Pose Estimation Module

Located in `pose_estimation/`, this module acts as the bridge between the real world and the simulation.

- **`keypoints.py`**: Utilizes Google's MediaPipe to infer a 3D topological map of the human body.
- **`Kinematic.py`**: Performs mathematical transformation (Inverse Kinematics) to map biological keypoints to the specific degrees of freedom (DoF) available in the MuJoCo Humanoid XML definition.

### Reinforcement Learning Core

Located in `lib/`, this module handles the decision-making process.

- **`PPO_Agent.py` / `agent_ppo.py`**: Despite the naming convention, this file implements the PPO Actor-Critic architecture. The **Actor** outputs mean joint torques, while the **Critic** estimates state value functions.
- **`buffer_PPO.py`**: Handles experience replay storage and computes Generalized Advantage Estimation (GAE) to stabilize training updates.
