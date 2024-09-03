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

## Running the Simulation Demo

**Run the simulation scripts for single instruction and multiple instructions:**

```sh
python main_simulation.py
python main_simulation_multi.py
```
These two are example videos of the simulation. The first one is a single instruction type of moving task robot simulation, and the second one is a complex logic of multi-instructions simulation.

See the videos below:

**Single-instruction:** In this video, the input command is "move the queen to C2". The robot will move the small green ball (queen at B6) to square C2.

https://github.com/user-attachments/assets/400ba2c1-fa2e-46e6-ac44-b870bc80d0c0


**multi-instruction:** IN this video, the input command is "move teh queen to its further square and move the pawn to its diagonal square". The robot will move the queen (ball at B6) to square C6 and the pawn (ball at B3) to square C2.

https://github.com/user-attachments/assets/71c3d2f9-7d9a-4fef-a84f-e60a670d1be6



## Notes for Language model

This experiment is based on GPT-4 language model, therefore the API key is required. Please setup your own key in the llmAPI/api.py file. (You can get the key from https://beta.openai.com/account/api-keys)

For now the api file has not been uploaded.

## Project Structure

The project structure is as follows:
