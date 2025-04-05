import itertools
import sys

import lightgbm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """学習用データを読み込み、特徴量と目的変数に分けて返します。"""
    train = pd.read_csv("data/train.csv", index_col="id")
    y_train = train.pop("rainfall")
    return train, y_train


def load_test() -> pd.DataFrame:
    """テスト用データを読み込みます。"""
    return pd.read_csv("data/test.csv", index_col="id")


def validate(pipeline: Pipeline, name: str, fold: int = 5) -> None:
    """交差検証によるモデルの評価を行います。"""
    x_train, y_train = load_train()

    metrics = cross_validate(
        estimator=pipeline,
        X=x_train,
        y=y_train,
        cv=fold,
        scoring="roc_auc",
        n_jobs=-1,
    )
    # Print Root Mean Squared Error
    score = metrics["test_score"]
    print(f"[{name.ljust(10)}] Score (roc_auc): {np.mean(score).round(5)}")


def fit_and_predict(pipeline: Pipeline, name: str) -> None:
    """モデルの学習とテストデータの予測を行い、提出用のファイルを作成します。"""
    outfile = f"submission-{name}.csv"
    print(f"[{name.ljust(10)}] Creating {outfile}")

    x_train, y_train = load_train()
    pipeline.fit(x_train, y_train)

    # 答えは 0, 1 の予測値（ラベル）ではなく、1 になる確率を 0.0〜1.0 で出力する。
    # predict ではなく predict_proba を使うことで、0、1 の確率を浮動小数点数で得られる。
    # 確率は round で丸めない方が得点は高くなる
    test = load_test()
    pred = pipeline.predict_proba(test)[:, 1]
    pred_df = pd.DataFrame({"id": test.index, "rainfall": pred})
    pred_df.to_csv(outfile, index=False)


class WindDirectionTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    winddirection には 0〜359 の値が入っているので、
    それを X, Y 成分に分解して連続値として扱えるようにする。
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_["wind_x"] = np.cos(np.radians(X_["winddirection"]))
        X_["wind_y"] = np.sin(np.radians(X_["winddirection"]))
        X_.drop(columns=["winddirection"], inplace=True)
        return X_


class HumidTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_["humid1"] = X_["sunshine"] * X_["humidity"]
        return X_


class LagFeatureTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """n日前のデータを特徴量として追加する Transformer。"""

    def __init__(self, lag=1):
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        # すべての特徴量に対して、lag 日前のものを追加する。
        # ただし、lag 日前のデータがない最初の lag 日は自分自身の値を使う。
        for col in X.columns:
            X_[f"{col}_lag{self.lag}"] = X_[col].shift(self.lag).fillna(X_[col])
        return X_


class StandardTransformer(BaseEstimator, TransformerMixin):
    """各カラムの数値を正規化する Transformer。"""

    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[X.columns] = self.scaler.transform(X)
        return X_


class FeatureInteractionTransformer(
    BaseEstimator, TransformerMixin, OneToOneFeatureMixin
):
    """数値カラムの組み合わせを生成する Transformer。"""

    def __init__(self):
        self.combinations = None

    def fit(self, X, y=None):
        # 数値カラムの2要素組み合わせを生成
        # cols = X.select_dtypes(include="number").columns
        cols = [
            "pressure",
            "maxtemp",
            "temparature",
            "mintemp",
            "dewpoint",
            "humidity",
            "cloud",
            "sunshine",
            "windspeed",
        ]
        self.combinations = list(itertools.combinations(cols, 2))
        return self

    def transform(self, X):
        X_ = X.copy()
        for col1, col2 in self.combinations:
            X_[f"{col1}_x_{col2}"] = X_[col1] * X_[col2]
        return X_


class DebugTransformer(BaseEstimator, TransformerMixin):
    """カラムの一覧を表示するためのデバッグ用 Transformer。"""

    def fit(self, X, y=None):
        print(X.columns)
        return self

    def transform(self, X):
        return X


def create_base_classifier_pipeline() -> Pipeline:
    """
    基準のスコアを求めるためのベースモデル。
    何も処理せずに LGBMClassifier で予測するだけ。
    """
    estimator = lightgbm.LGBMClassifier(
        random_state=0, force_col_wise=True, n_jobs=-1, verbose=-1
    )
    return make_pipeline(estimator)


def create_lgbm_classifier_pipeline() -> Pipeline:
    estimator = lightgbm.LGBMClassifier(
        random_state=1,
        force_col_wise=True,
        n_jobs=-1,
        verbose=-1,
    )
    # Scikit-learn の Transformer である SimpleImputer や StandardScaler は
    # デフォルトで Numpy 配列を返すが、set_output() で pandas.DataFrame を返すように設定できる。
    # 独自の Transformer を作成する場合は、OneToOneFeatureMixin を継承することで
    # デフォルトの set_output() を提供してくれる。
    return make_pipeline(
        # 欠損値を適当に埋める
        SimpleImputer(strategy="most_frequent"),
        # 数値カラムの乗算での組み合わせを特徴量として追加するので、先に値を正規化しておく。
        StandardScaler(),
        # 数値カラムの乗算での組み合わせを特徴量として追加する。
        FeatureInteractionTransformer(),
        # DebugTransformer(),
        # 1 日前のデータを特徴量として追加する。
        LagFeatureTransformer(lag=1),
        estimator,
    ).set_output(transform="pandas")


def create_kneighbors_classifier_pipeline() -> Pipeline:
    """KNeighborsClassifier を使った予測モデル。"""
    # 予測モデル
    estimator = KNeighborsClassifier(n_neighbors=100, p=1, n_jobs=-1)
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        WindDirectionTransformer(),
        StandardScaler(),
        FeatureInteractionTransformer(),
        LagFeatureTransformer(lag=1),
        estimator,
    ).set_output(transform="pandas")


if __name__ == "__main__":
    pipes = [
        ("lgbm", create_lgbm_classifier_pipeline()),  # 0.8858
        ("knn", create_kneighbors_classifier_pipeline()),  # 0.87228
    ]

    # スタッキングモデルを作成
    stacking = StackingClassifier(
        estimators=pipes,
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    if len(sys.argv) > 1 and sys.argv[1].startswith("--val"):
        print("Cross validation...")
        for name, pipe in pipes:
            validate(pipe, name)
        validate(stacking, "stacking")
    else:
        print("Fitting and predicting...")
        for name, pipe in pipes:
            fit_and_predict(pipe, name)
        fit_and_predict(stacking, "stacking")
