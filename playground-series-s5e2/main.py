import sys
from abc import abstractmethod

import lightgbm
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


class BaseModel:
    def __init__(
        self, *, extra=True, small_test=False, outfile="submission.csv"
    ) -> None:
        """
        学習用データを読み込みます。
        - extra: 追加の学習用データをマージする場合は True
        - small_test: 全体の流れを確認するために少量のデータで動かす場合 True
        """
        # train = pd.read_csv("data/training_extra.csv", index_col="id")
        train = pd.read_csv("data/train.csv", index_col="id")
        if extra:
            ext = pd.read_csv("data/training_extra.csv", index_col="id")
            # データを結合（インデックスの再割り当てはしない）
            train = pd.concat([train, ext], ignore_index=False)

        if small_test:
            train = train[:10000]

        self.x_train = train.drop("Price", axis=1)
        self.y_train = train["Price"]
        self.outfile = outfile

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

        # Print Root Mean Squared Error
        rmse = (metrics["test_score"].mean() * -1) ** 0.5
        print(f"RMSE: {rmse.round(3)}\n")

    def fit_and_predict(self) -> None:
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
        pred_df.to_csv(self.outfile, index=False)
        print(f"Submission file is created: {self.outfile}\n")

    def get_cat_cols(self):
        """Get the list of categorical columns."""
        return self.x_train.select_dtypes(include=["object"]).columns.to_list()

    def get_num_cols(self):
        """Get the list of numerical columns."""
        return self.x_train.select_dtypes(include=["number"]).columns.to_list()

    @abstractmethod
    def _create_pipeline(self) -> Pipeline:
        """
        前処理とモデルを組み合わせたパイプラインを作成します。
        """
        raise NotImplementedError("This method must be implemented in a subclass.")


class RandomForestModel(BaseModel):
    def __init__(self, *, small_test=False) -> None:
        super().__init__(small_test=small_test, outfile="submission-random-forest.csv")

    def _create_pipeline(self) -> Pipeline:
        # カテゴリ変数用の変換器
        cat_transformer = Pipeline(
            steps=[
                ("OneHot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
        )
        # カラムの種類に応じて変換器を割り当て
        transformer = ColumnTransformer(
            transformers=[("categorical", cat_transformer, self.get_cat_cols())],
            remainder="passthrough",
        )
        # 予測モデル
        estimator = RandomForestRegressor(random_state=0, n_jobs=-1)
        return make_pipeline(transformer, estimator)


class LgbmModel(BaseModel):
    def __init__(self, *, small_test=False) -> None:
        super().__init__(small_test=small_test, outfile="submission-lgbm.csv")

    def _create_pipeline(self) -> Pipeline:
        # カテゴリ変数用の変換器
        cat_transformer = Pipeline(
            steps=[
                ("OneHot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
        )
        # 数値変数用の変換器
        num_transformer = Pipeline(
            steps=[
                ("SimpleImputer", SimpleImputer(strategy="mean")),
                ("Polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ]
        )
        # カラムの種類に応じて変換器を割り当て
        transformer = ColumnTransformer(
            transformers=[
                ("categorical", cat_transformer, self.get_cat_cols()),
                ("numerical", num_transformer, self.get_num_cols()),
            ],
            remainder="passthrough",
        )
        # 予測モデル
        estimator = lightgbm.LGBMRegressor(
            random_state=0, force_col_wise=True, n_jobs=-1
        )
        return make_pipeline(transformer, estimator)


if __name__ == "__main__":
    # m = RandomForestModel(small_test=False)
    m = LgbmModel(small_test=False)
    if len(sys.argv) > 1 and sys.argv[1].startswith("--val"):
        m.validate()
    else:
        m.fit_and_predict()
