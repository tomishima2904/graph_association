# 概要 (Abstract)
日本語wikidataを学習用データセットとし、Pytorch-Biggraphを用いて埋め込み表現を学習した。その埋め込み表現を利用した連想実験を行う。

# 実験 (Experiments)

## 環境(Environment)

()はあんま関係ないけど一応記載しておく。
- `Python 3.6.8`
- `torchbigraph==1.0.1`
- `pandas==1.1.5`
- `(torch==1.3.0)`
- `(CUDA Version 10.1.243)`
- 使用したdockerイメージは`lbjcom/cuda10.1-pytorch3.6-warp-ctc-apex`

## 使用方法 (Usage)
```
python src/graph_association.py \
    --comparator='dot' \
    --threshold=5 \
    --top_k=150 \
    --use_cuda='f' \
    --with_vec='f'
```
コマンドライン引数で適当に[オプション](https://github.com/tomishima2904/graph_association/blob/master/src/config.py)を変えることができる。

# ディレクトリ・ファイルについて (About directories and files)
主要なファイルのみ説明する。

## data

### dataset/fit2022.csv
実験用データセット。『まとめる語想起』132題のうち、20代男子大学生3人の解答が正解の連想語と一致し、なおかつ、カテゴリーが単語で表すことのできる**80題**を採用。構成は以下の通り。
```
answer,stims,category
スポーツ,"['バレーボール', '水泳', 'マラソン', '体操', 'サッカー']",仲間
薬,"['鎮痛剤', '湿布', 'カプセル', '軟膏', '錠剤']",仲間
                        ︙
クリスマス,"['ケーキ', 'サンタクロース', 'ツリー', '靴下', 'プレゼント']",行事
カレー,"['じゃがいも', '人参', '肉', 'たまねぎ', 'ルウ']",メニュー
```

### dataset/fit2022v2.csv
以下の語句はwikidataにタイトルとして存在しなかったので、`dataset/fit2022.csv`から以下の語句を含む例文を削除した**72題**。。
```
[布団干し, 長針, 短針, 尾ひれ, ガスの炎, 木の幹, 新学期, 救急室, レントゲン室, リハビリ訓練室, 両替所]
```


### jawiki-20220601-id2title.pickle
展開するとkeyにid、valueにタイトルを持つ辞書(json)型となっている。`{'5': 'アンパサンド', '11': '日本語', ...}`の様になっている。

## models

### embeddings_all_0_v50.pickle
展開するとkeyにid, valueに学習済み分散表現を持つ辞書型となっている。ファイルサイズがかなり大きい(**3.5GB**)のでgit管理下にはしていない。このファイルが欲しい場合は`/work/{僕のユーザ名}/wikipedia_graph_embedding/model/jawiki_split_1/`にある。


## results

### *dataset*_*comparator*_*topk*.json
`graph_association.compare()`の実行によって得ることのできる結果(`--with_vec='True'`の場合)。使い方及び構成は以下。
```
from utils.filehandler import json_reader
compared_words = json_reader(results_date_time.json)

"""
以下のような構成となっている。
compared_words[b*n]  # keyは刺激語のid
                ├── ['stim']  # str: 刺激語のタイトル
                ├── ['vec']  # list: 刺激語のidに対応する埋め込み表現(ベクトル)
                └── ['associated'][r]
                                    ├── ['id']  # str: タイトルに対応するwikidataのid
                                    ├── ['title']  # str: 刺激語のベクトルに近かかったタイトル
                                    ├── ['vec']  # list: 刺激語のidに対応する埋め込み表現(ベクトル)
                                    └── ['score']  # float: 刺激語とタイトルの類似度(デフォルトはdot積の値)
"""

```


### results_*date*_*time*.json
`graph_association()`の実行によって得ることのできる結果。使い方及び構成は以下。
```
from utils.filehandler import json_reader
results = json_reader(results_date_time.json)

print(type(results))  # dict
print(len(results))  # 問題数(b==約80題)
print(len(results[b]['stims']))  # 刺激語の数(n==5語)
print(len(results[b]['stims'][n]['associated']))  # 刺激語の埋め込み表現に近い上位の語群の数(r==デフォルトは上位150語)

"""
以下のような構成となっている。
results[b]
        ├── ['ans']  # str: 人間の正解語
        ├── ['cat']  # str: 正解語が属するカテゴリー(例えば白が正解語ならカテゴリーは色)
        ├── ['stims'][n]
        │             ├── ['id']  # str: 刺激語に対応するwikidataのid
        │             ├── ['stim']  # str: 刺激語
        │             └── ['associated'][r]
        │                                ├── ['id']  # str: タイトルに対応するwikidataのid
        │                                ├── ['title']  # str: 刺激語のベクトルに近かかったタイトル
        │                                └── ['score']  # float: 刺激語とタイトルの類似度(デフォルトはdot積の値)
        └── ['results']
                ├── ['predictions']  # str: 予測結果
                └── ['rank']  # int: 予測を断定した時の順位。断定できなかった場合は0。
"""
```

## src

### config.py
`python src/graph_association.py`を実行する際の設定を決めるファイル。必要な引数やデフォルト値が記されている。

### graph_association.py
連想を行うメインのファイル。


### utils.comparators.py
埋め込み表現同士を比較する際に使用する。`arg_option.py`の`--comparator`から距離関数(`dot`や`cos`等)を指定できる。

### utils.file_handlers.py
ファイルの読み込みや書き出し、ディレクトリの作成等といったファイル操作を行う。


# 参考文献 (References)
- [facebookresearch/PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) (公式のgithub)
- [Welcome to PyTorch-BigGraph’s documentation!](https://torchbiggraph.readthedocs.io/en/latest/index.html) (公式のリファレンス)
- ["Hello World!" in PyTorch BigGraph](http://nadbordrozd.github.io/blog/2020/08/04/hello-world-in-pytorch-biggraph/) (Pytorch-Biggaphのチュートリアル記事)
- [Pytorch-BigGraphによるWikipedia日本語記事のグラフ埋め込み](https://buildersbox.corp-sansan.com/entry/2019/09/26/110000)  (Pytorch-Biggraphの紹介記事)