import itertools

N = 4  # 4×4 のとき

agents = list(range(N))

# 4! = 24 通りの選好パターンを列挙
# prefs[i] が i 番目の選好 (例: (0,1,2,3) や (2,0,3,1) など)
prefs = list(itertools.permutations(agents))
P = len(prefs)  # 24

# perm -> index の辞書
perm_to_idx = {perm: i for i, perm in enumerate(prefs)}

# succ[p][x] : 選好パターン p において，x より好きな相手のリスト
# succeq[p][x]: x を含めて x 以上に好きな相手のリスト
succ = [[[] for _ in agents] for _ in range(P)]
succeq = [[[] for _ in agents] for _ in range(P)]

for pid, perm in enumerate(prefs):
    # perm は「好みの順に並んだ相手のラベル」を与えるタプル
    # 例: perm = (2,0,3,1)
    # pos[a] はラベル a の「順位（小さいほど好き）」を表す
    pos = {a: i for i, a in enumerate(perm)}
    for x in agents:
        better = [y for y in agents if pos[y] < pos[x]]
        succ[pid][x] = better
        # 「x 以上に好き」= x より好きな相手 + x 自身
        succeq[pid][x] = better + [x]

# -----------------------------
# change[p][k] の生成
# -----------------------------
# 4人だとラベルのペアは C(4,2) = 6 通り
pairs = list(itertools.combinations(agents, 2))  # [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
num_pairs = len(pairs)  # 6

# change[p][k] :
#   選好パターン p に対して，
#   pairs[k] = (a,b) のラベル入れ替え (a<->b) をしたときの選好パターンのインデックス
change = [[0] * num_pairs for _ in range(P)]

for pid, perm in enumerate(prefs):
    for k, (a, b) in enumerate(pairs):
        # 「ラベル a と b を入れ替える」ので，
        # 各要素 x を b (x==a のとき)，a (x==b のとき)，それ以外はそのまま に写像する
        new_perm = tuple(
            b if x == a else a if x == b else x
            for x in perm
        )
        change[pid][k] = perm_to_idx[new_perm]

# ここまでで，
# - 全 24 通りの選好 prefs
# - succ, succeq
# - change (4人用の 6ペア分)
# が自動生成できています

print(succ)
