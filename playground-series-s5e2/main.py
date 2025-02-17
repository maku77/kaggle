from abc import abstractmethod

import lightgbm
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class BaseModel:
    def __init__(self, *, extra=True, small_test=False) -> None:
        """
        学習用データを読み込みます。
        - extra: 追加の学習用データをマージする場合は True
        - small_test: 全体の流れを確認するために少量のデータで動かす場合 True
        """
        train = pd.read_csv("data/train.csv", index_col="id")
        if extra:
            ext = pd.read_csv("data/training_extra.csv", index_col="id")
            # データを結合（インデックスの再割り当てはしない）
            train = pd.concat([train, ext], ignore_index=False)

        if small_test:
            train = train[:10000]

        self.x_train = train.drop("Price", axis=1)
        self.y_train = train["Price"]

    def validate(self, fold=5) -> None:
        """
        交差検証によるモデルの評価を行います。
        """
        print("Cross validating...")
        pipeline = self._create_pipeline()
        metrics = cross_validate(
            estimator=pipeline,
            X=self.x_train,
            y=self.y_train,
            cv=fold,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        rmse = (metrics["test_score"].mean() * -1) ** 0.5
        print(f"RMSE: {rmse.round(3)}\n")  # Root Mean Squared Error

    def fit_and_predict(self, outfile="submission.csv") -> None:
        """
        モデルの学習とテストデータの予測を行い、提出用のファイルを作成します。
        """
        pipeline = self._create_pipeline()

        print("Fitting the model...")
        pipeline.fit(self.x_train, self.y_train)

        print("Predicting...")
        test = pd.read_csv("data/test.csv", index_col="id")
        pred = pipeline.predict(test)

        pred_df = pd.DataFrame({"id": test.index, "Price": pred.round(3)})
        pred_df.to_csv(outfile, index=False)
        print(f"Submission file is created: {outfile}\n")

    def _get_categorical_columns(self):
        """
        カテゴリ変数のカラム一覧を取得します。
        """
        return self.x_train.select_dtypes(include=["object"]).columns.to_list()

    @abstractmethod
    def _create_pipeline(self) -> Pipeline:
        """
        前処理とモデルを組み合わせたパイプラインを作成します。
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class RandomForestModel(BaseModel):
    def __init__(self, *, small_test=False) -> None:
        super().__init__(small_test=small_test)

    def _create_pipeline(self) -> Pipeline:
        # 各カラムの変換器を作成（カテゴリ変数は One-hot エーコーディング、それ以外はそのまま残す）
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "OH",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self._get_categorical_columns(),
                )
            ],
            remainder="passthrough",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=0, n_jobs=-1)),
            ],
            verbose=False,
        )


class LgbmModel(BaseModel):
    def __init__(self, *, small_test=False) -> None:
        super().__init__(small_test=small_test)

    def _create_pipeline(self) -> Pipeline:
        # 各カラムの変換器を作成（カテゴリ変数は One-hot エーコーディング、それ以外はそのまま残す）
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "OH",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self._get_categorical_columns(),
                )
            ],
            remainder="passthrough",
        )
        model = lightgbm.LGBMRegressor(random_state=0, force_col_wise=True, n_jobs=-1)
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)],
            verbose=False,
        )


if __name__ == "__main__":
    m1 = LgbmModel(small_test=False)
    m1.fit_and_predict("submission-lgbm.csv")
    # m1.validate()

    # m2 = RandomForestModel(small_test=False)
    # m2.fit_and_predict("submission-random-forest.csv")
    # m2.validate()
