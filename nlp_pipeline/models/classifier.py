from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

VALID_PARAMS = {    
    "logreg": {"C", "max_iter"},
    "svm": {"C"},
    "nb": {"alpha"},
    "rf": {"n_estimators", "max_depth"},
    "lgbm": {"n_estimators", "learning_rate"}
}

class Classifier:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]
        self.search_type = config.get("search", "manual")  # manual | grid | random
        self.params = config.get("params", {})
        self.param_grid = config.get("param_grid", {})
        self.n_iter = config.get("n_iter", 10)
        self.cv = config.get("cv", 3)
        self.model = None

    # ---------------------------
    # MODEL FACTORY
    # ---------------------------
    def _get_base_model(self):
        if self.model_name == "nb":
            return MultinomialNB()

        elif self.model_name == "bernoulli_nb":
            return BernoulliNB()

        elif self.model_name == "logreg":
            return LogisticRegression(max_iter=1000)

        elif self.model_name == "svm":
            return LinearSVC()

        elif self.model_name == "rf":
            return RandomForestClassifier()

        elif self.model_name == "lgbm":
            return LGBMClassifier()

        else:
            raise ValueError(f"Modelo não suportado: {self.model_name}")

    # ---------------------------
    # FIT
    # ---------------------------
    def fit(self, X, y):
        base_model = self._get_base_model()

        if self.search_type == "manual":     
            model_name = self.model_name
            valid_keys = VALID_PARAMS.get(model_name, set())
            filtered_params = {
                k: v for k, v in self.params.items()
                if k in valid_keys
            }
            if filtered_params:
                base_model.set_params(**filtered_params)

            self.model = base_model
            self.model.fit(X, y)

        # 🔹 GRID SEARCH
        elif self.search_type == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grid,
                scoring="f1_macro",
                cv=self.cv,
                n_jobs=-1,
                verbose=1
            )
            search.fit(X, y)
            self.model = search.best_estimator_

            print("🔎 Best params (Grid):", search.best_params_)
            print("📊 Best score:", search.best_score_)

        # 🔹 RANDOM SEARCH
        elif self.search_type == "random":
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring="f1_macro",
                cv=self.cv,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            search.fit(X, y)
            self.model = search.best_estimator_

            print("Best params (Random):", search.best_params_)
            print("Best score:", search.best_score_)

        else:
            raise ValueError(f"Search inválido: {self.search_type}")

    # ---------------------------
    # PREDICT
    # ---------------------------
    def predict(self, X):
        return self.model.predict(X)
    
    