import retro
import time

import sys
from gym.spaces import *

# 空間の出力
def print_spaces(label, space):
    # 空間の出力
    print(label, space)
    # Box/Discreteの場合は最大値と最小値も表示
    if isinstance(space, Box):
        print(' 最小値: ', space.low)
        print(' 最大値: ', space.high)
    if isinstance(space, Discrete):
        print(' 最小値: ', 0)
        print(' 最大値: ', space.n-1)

# 環境の生成 (1)
env = retro.make(game='Gradius-Nes', state='Level1')

# 行動空間と状態空間の型の出力
print_spaces('行動空間: ', env.action_space)
print_spaces('状態空間: ', env.observation_space)

# ランダム行動による動作確認
state = env.reset()
while True:
    # 環境の描画
    env.render()

    # スリープ
    time.sleep(1/60)

    # 行動
    action = env.action_space.sample()

    # 1ステップ実行
    state, reward, done, info = env.step(action)
    print('reward:', reward)

    # エピソード完了
    if done:
        print('done')
        state = env.reset()
