README

# Project Environment Setup

This document provides instructions for setting up the project environment.

## Prerequisites

- Python 3.8 or higher
- mujoco 210 (newest version)
- no mujoco-py, only using mujoco
- mamba (optional, but recommended)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/zniihgnexy/niryo_project.git
    cd niryo_project
    ```

2. **Create a virtual environment:**

    ```sh
    mamba env create -f mamba_mujoco_base.yml
    ```

3. **Activate the virtual environment:**

    ```sh
    conda activate mujoco
    ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Simulation

**Run the main simulation script:**

```sh
python main_simulation.py
python main_simulation_multi.py
```
These two are demonstration of the simulation. The first one is a single stage of moving task robot simulation, and the second one is a multi-instructions simulation.

See the videos below:

single-instruction:
<video controls src="single_instruction.mp4" title="Title"></video>

multi-instruction:
<video controls src="multi_intruction.mp4" title="Title"></video>

## Notes for Language model

This experiment is based on GPT-4 language model, therefore the API key is required. Please setup your own key in the llmAPI/api.py file. (You can get the key from https://beta.openai.com/account/api-keys)

For now the api file has not been uploaded.