# NIR画像のラベルなし乳牛個体識別(DSLN + 疑似ラベル)

## 概要

牛舎カメラの夜間近赤外(NIR)画像で、ラベルなしで乳牛を個体識別する手法の実装です。
ラベル付きの昼間RGB画像で学習したモデルを、疑似ラベルとドメイン特化正規化層(DSLN)によってNIR画像に適応させます。
学部研究のコードから、手法の中核部分を抜粋して整理したものです。

## 背景

24時間の個体モニタリングには夜間画像での識別が必要ですが、NIR画像にはラベルがありません。

## 手法

学習は3段階です。このリポジトリには2と3を収めています。

1. RGB画像で、MegaDescriptor事前学習重みのSwin TransformerをArcFace Lossでファインチューニングし、ベースモデルを作る(本リポジトリには含まない)
2. ベースモデルでNIR画像に疑似ラベルを付ける
3. RGB画像と疑似ラベル付きNIR画像で、DSLNに置換したモデルを追加学習する

### DSLN(Domain-Specific LayerNorm)

RGBとNIRでは特徴量の統計が異なります。そこでSwin Transformerの全LayerNormを、ドメインごとに重みを持つ層へ置換します。パラメータの大半は両ドメインで共有し、正規化層だけをドメインごとに分けます。

| 処理 | 場所 |
|---|---|
| DSLN層 | `nir_cattle_id/dsln.py` の `DSLN` |
| 全LayerNormの置換(元の重みを両ドメインへコピー) | `nir_cattle_id/dsln.py` の `convert_to_dsln` |
| ドメインの一括切り替え | `nir_cattle_id/dsln.py` の `set_model_domain` |
| Swinへの組み込みと凍結設定 | `nir_cattle_id/model.py` の `build_dsln_swin` |

### 疑似ラベル生成

NIR画像を個体数より多いクラスタに分割し(Over-clustering)、RGBデータベースを参照した投票でクラスタ単位にラベルを決めます。

1. NIR特徴量をK-Meansでクラスタリングする
2. 各NIR画像に、コサイン類似度が最大のRGB画像の個体ID(k=1の最近傍)を票として割り当てる
3. クラスタごとに多数決を取り、多数派の割合(純度)が閾値未満のクラスタは棄却する

| 処理 | 場所 |
|---|---|
| 全体の流れ | `nir_cattle_id/pseudo_label.py` の `assign_pseudo_labels` |
| 最近傍の票 | `nir_cattle_id/pseudo_label.py` の `nearest_db_labels` |
| クラスタ内投票と純度の計算 | `nir_cattle_id/pseudo_label.py` の `vote_in_clusters` |

### 学習ループ

1ステップでRGBとNIRを1バッチずつ、対応するDSLNに切り替えて流します。ArcFace Lossのクラス重みは両ドメインで共有します。実装は `nir_cattle_id/trainer.py` の `DualDomainTrainer.train_step` です。

## 結果

### Top-1精度(3データセット平均、提案手法は各5回平均)

| | ベースモデル | 提案手法 |
|---|---|---|
| NIR | 0.850 | 0.902 |
| RGB | 0.932 | 0.949 |

### 疑似ラベルの精度(1データセットでの値、K=100、閾値60%)

| Precision | Recall |
|---|---|
| 0.937 | 0.843 |

Precisionは採用した画像のうちラベルが正しい割合、Recallは全画像のうち正しいラベルで採用された割合です。評価には正解ラベルを使いますが、生成処理では使いません。

## 自分が実装した範囲

学習・推論パイプライン全体、モデル構築、疑似ラベル処理、DSLN組み込み、実験・評価を担当しました(個人研究)。

## ディレクトリ構成

```
configs/          疑似ラベル生成と学習の設定(YAML)
nir_cattle_id/
  dsln.py         DSLN層とモデルへの組み込み
  model.py        Swinの構築と凍結設定
  pseudo_label.py 疑似ラベル生成
  data.py         前処理とPKサンプリング
  trainer.py      学習ループと検証
scripts/          疑似ラベル生成と学習の実行スクリプト
tests/            実データなしで動くテスト
```

## テストの実行方法

テストの実行には、実データ、学習済み重み、GPU、ネットワーク接続のいずれも不要です。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

テストでは計算量を抑えるため、ランダム初期化した `swin_tiny_patch4_window7_224` を使います。

## 実行方法

設定ファイルのパスを書き換えてから、リポジトリのルートで実行します。データは `<root>/<個体ID>/<画像>` の形式を想定しています。

```bash
python -m scripts.generate_pseudo_labels --config configs/pseudo_label.yaml
python -m scripts.train --config configs/train_dsln.yaml --seed 42
```

## 注意

研究データは提供元との関係で公開していません。学習済み重みも含みません。

## 発表

画像電子学会 第317回研究会、2026年6月22日(口頭発表)

発表者: 柴野桜次朗

共著者: 谷口行信教授、山本洋太助教、中村和晃准教授
