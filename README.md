## Sisyphus Environments

### Installation

This package can be installed via pip.
Simply navigate into the root directory of the repository and run
```bash
pip install .
```

If you plan on using the real robot, additional dependencies have to be installed.
```bash
# Controller for the Robotis gripper
pip install git+https://github.com/TimSchneider42/python-rhp12rn-controller.git

# Natnet client used to process Optitrack streams
pip install git+https://github.com/TimSchneider42/python-natnet-client

# Optionally, if you want to use the Telegram bot for controlling and watching the robot remotely
pip install opencv-python pyrogram
```

### Usage
Gym environments used in our IROS paper [Active Exploration for Robotic Manipulation](https://sites.google.com/view/aerm/home).
See `example/random_action.py` for a usage example.
To recreate the conditions used in our experiments, invoke the script as follows:
```bash
# For the experiment without holes in the table (`-f` controls the finger friction, setting it higher decreases the task difficulty)
python random_action.py -f 0.3

# For the experiment with holes in the table
python random_action.py -f 0.5 -H 
```
See our paper for further details.