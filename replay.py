import retro
import time

# bkファイルの読み込み
movie = retro.Movie('./stage1_clear.bk2')
movie.step()

# 環境の生成
env = retro.make(
   game=movie.get_game(),
   state=None,
   use_restricted_actions=retro.Actions.ALL,
   players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()

# 再生ループ
while movie.step():
   # スリープ (240で4倍速)
   time.sleep(1/240)
   # キーの取得
   keys = []
   for p in range(movie.players):
       for i in range(env.num_buttons):
           keys.append(movie.get_key(i, p))

   # 1ステップ実行
   env.step(keys)

   # 環境の描画
   env.render()
