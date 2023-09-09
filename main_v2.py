import numpy as np
import torch
import gym
import argparse
import os

from collections import deque
from gym.wrappers.pixel_observation import PixelObservationWrapper

# for rendering
import cv2
import matplotlib.pyplot as plt

# defined modules
import TD3_v2
import utils_v2


video_count = 0

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    eval_env = PixelObservationWrapper(eval_env)

    # for rendering
    global video_count
    video_count += 1
    frames = []

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        # create a state queue, initialize with all initial state
        state_queue = deque(
            [ utils_v2.pre_process_img(state["pixels"]) for _ in range(args.state_queue_length) ], maxlen=args.state_queue_length
        )
        state = torch.cat(list(state_queue), 1).cpu().numpy()
        # state.shape should be (c * stack layer, x, y) by now
        # print("eval_state0", state.shape)
        
        prev_state = state
        prev_action = np.zeros((6))

        while not done:
            # for rendering
            frames.append(eval_env.render(mode="rgb_array"))
            
            action = policy.select_action(state)
            # print("action", action.shape)
            state, reward, done, _ = eval_env.step(action)

            # Stack frames
            state_queue.append(utils_v2.pre_process_img(state["pixels"]))
            
            # print([x.shape for x in state_queue])

            state = torch.cat(list(state_queue), 1).cpu().numpy()

            # print("equal", np.array_equal(state, prev_state))
            # prev_state = state

            print("eval prev_action", prev_action.shape)
            print(prev_action)
            print("eval action", action.shape)
            print(action)
            print("equals", np.array_equal(prev_action, action))
            prev_action = action
            
            # print("eval_state1", state.shape)

            avg_reward += reward

    # for rendering
    output_video = "./image/output" + str(video_count) + ".avi"
    frame_rate = 24
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    avg_reward /= eval_episodes

    print("--------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("--------------------------------")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    # parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.5, type=float)    # Std of Gaussian exploration noise
    # parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=128, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--state_queue_length", default=3)          # Length of the state_queue (number of frames stacked together as input)

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
        
    env = gym.make(args.env)
    # to express state in terms of pixels
    env = PixelObservationWrapper(env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # state_dim = env.observation_space["pixels"].shape
    state_queue_length = 3      # number of frames stacked together
    in_channels = 3             # RGB 3 channels
    out_channels = 64
    kernel_size = 5
    image_dim = 72
    state_dim = (in_channels * state_queue_length, image_dim, image_dim)
    

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        # "state_dim": state_dim,
        "state_queue_length": state_queue_length,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "image_dim": image_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3_v2.ImageInputTD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils_v2.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    # create a state queue, initialize with all initial state
    state_queue = deque(
        [ utils_v2.pre_process_img(state["pixels"]) for _ in range(args.state_queue_length) ], maxlen=args.state_queue_length
    )
    state = torch.cat(list(state_queue), 1).cpu().numpy()
    # state.shape should be (c * stack layer, x, y) by now
    print("state0", state.shape)

    prev_action = np.zeros((action_dim))
    



    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                - np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        print("prev_action", prev_action.shape)
        print(prev_action)
        print("action", action.shape)
        print(action)
        print("equals", np.array_equal(prev_action, action))

        prev_action = action

        
        
        # Perform action
        next_state, reward, done, _ = env.step(action)
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0

        # Stack frames
        state_queue.append(utils_v2.pre_process_img(next_state["pixels"]))
        next_state = torch.cat(list(state_queue), 1).cpu().numpy()
        
        # print("state", state.shape)
        # print("next_state", next_state.shape)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            print("----------TRAINING---------")
            policy.train(replay_buffer, args.batch_size)
        
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # create a state queue, initialize with all initial state
            state_queue = deque(
                [ utils_v2.pre_process_img(state["pixels"]) for _ in range(args.state_queue_length) ], maxlen=args.state_queue_length
            )
            state = torch.cat(list(state_queue), 1).cpu().numpy()
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")