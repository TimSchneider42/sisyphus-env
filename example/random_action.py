import argparse

import numpy as np

from robot_gym.core.base_task import EpisodeInvalidException

from sisyphus_env import mk_sisyphus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--friction", type=float, default=0.3, help="Friction of the ball.")
    parser.add_argument("-H", "--holes", action="store_true", help="Add holes to the table.")
    parser.add_argument("-k", "--keyboard-control", action="store_true",
                        help="Activate keyboard robot control (requires root).")
    parser.add_argument("-i", "--invert-controls", action="store_true",
                        help="Invert left/right and up/down in keyboard control.")
    parser.add_argument("--headless", action="store_true", help="Run environment in headless mode.")

    args = parser.parse_args()

    if args.keyboard_control:
        import keyboard

    env = mk_sisyphus(headless=args.headless, use_holes=args.holes, ball_friction=args.friction)

    try:
        while True:
            env.reset()
            env.render()
            done = False
            try:
                while not done:
                    if args.keyboard_control:
                        inv = -1.0 if args.invert_controls else 1.0
                        ang_act = 0.0
                        if keyboard.is_pressed("a"):
                            ang_act = 0.5 * inv
                        elif keyboard.is_pressed("d"):
                            ang_act = -0.5 * inv
                        lin_act = np.zeros(2)
                        if keyboard.is_pressed("left"):
                            lin_act[0] = -inv
                        elif keyboard.is_pressed("right"):
                            lin_act[0] = inv
                        if keyboard.is_pressed("up"):
                            lin_act[1] = inv
                        elif keyboard.is_pressed("down"):
                            lin_act[1] = -inv
                        action = np.concatenate([lin_act, [ang_act]])
                    else:
                        action = env.action_space.sample()
                    obs, rew, done, info = env.step(action)
                    prev_rew = rew
                    env.render()
            except EpisodeInvalidException as ex:
                print("Caught episode invalid exception: {}".format(ex))
    finally:
        env.close()
