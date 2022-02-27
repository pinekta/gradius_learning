import retro
import os
import time
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from baselines.common.retro_wrappers import *
from stable_baselines.bench import Monitor
from util import CustomRewardAndDoneEnv, callback, log_dir
from stable_baselines.common import set_global_seeds

# 環境の生成
env = retro.make(game='Gradius-Nes', state='Level1')
env = CustomRewardAndDoneEnv(env)
env = StochasticFrameSkip(env, n=4, stickprob=0.25)
env = Downsample(env, 2)
env = Rgb2gray(env)
env = FrameStack(env, 4)
env = ScaledFloatFrame(env)
env = TimeLimit(env, max_episode_steps=4500)
env = Monitor(env, log_dir, allow_early_resets=True)
print('行動空間: ', env.action_space)
print('状態空間: ', env.observation_space)

# シードの指定
env.seed(0)
set_global_seeds(0)

# ベクトル環境の生成
env = DummyVecEnv([lambda: env])

# モデルの生成
model = PPO2(policy=CnnPolicy, env=env, verbose=0, learning_rate=0.000025)

# モデルの読み込み
# model = PPO2.load('gradius_model', env=env, verbose=0)

model.learn(total_timesteps=20000000, callback=callback)

state = env.reset()
total_reward = 0
while True:
    env.render()

    time.sleep(1/120)

    # 推論
    action, _ = model.predict(state)

    state, reward, done, info = env.step(action)
    total_reward += reward[0]

    if done:
        print('reward:', total_reward)
        state = env.reset()
        total_reward = 0
