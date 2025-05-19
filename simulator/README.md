# Udacity Simulator

A copy of the simulator binary can be found on [Google Drive](https://drive.google.com/file/d/1OI2AVAJdxMzKv20G3sogP4zAQbNR8aGb/view?usp=sharing).
# Udacity Gym Environment

Install Python dependencies using pip:
```shell
pip3 install -r requirements.txt
```

## Udacity Gym

Create a Gym:
```python
from udacity.gym import UdacityGym
# Start Simulator
simulator = UdacitySimulator(
    sim_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
simulator.start()

# Create Gym
env = UdacityGym(
    simulator=simulator,
    track="lake",
)
```

Run an experiment:

```python
observation, _ = env.reset(track="lake")

for _ in range(200):
    action = agent(observation)
    observation, reward, terminated, truncated, info = env.step(action)
```
