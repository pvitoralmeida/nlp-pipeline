from copy import deepcopy

from nlp_pipeline.pipeline.pipeline import NLPPipeline
from nlp_pipeline.experiments.results import ResultsManager
from nlp_pipeline.experiments.strategies import (
    generate_all_combinations,
    sample_random_configs
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class ExperimentRunner:
    def __init__(self, base_config, config_space, strategy="grid", n_samples=5):
        self.base_config = base_config
        self.config_space = config_space
        self.strategy = strategy
        self.n_samples = n_samples

    def _split(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _evaluate(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    def _get_configs(self):
        if self.strategy == "grid":
            return list(generate_all_combinations(self.config_space))

        elif self.strategy == "random":
            return sample_random_configs(self.config_space, self.n_samples)

        else:
            raise ValueError("Strategy inválida")

    def run(self, X, y):
        X_train, X_val, X_test, y_train, y_val, y_test = self._split(X, y)

        results = ResultsManager()
        configs = self._get_configs()

        print(f"\nTotal de experimentos: {len(configs)}")

        for i, cfg in enumerate(configs):
            print(f"\nExperimento {i+1}/{len(configs)}")
            print(cfg)

            config = deepcopy(self.base_config)

            config["preprocessing"].update(cfg.get("preprocessing", {}))
            config["vectorization"].update(cfg.get("vectorization", {}))
            config["model"].update(cfg.get("model", {}))

            pipeline = NLPPipeline(config)

            # treino
            pipeline.fit(X_train, y_train)

            # validação
            val_pred = pipeline.predict(X_val)
            val_f1 = self._evaluate(y_val, val_pred)

            # teste
            test_pred = pipeline.predict(X_test)
            test_f1 = self._evaluate(y_test, test_pred)

            results.add(
                cfg.get("preprocessing"),
                cfg.get("vectorization"),
                cfg.get("model"),
                val_f1,
                test_f1
            )

        df = results.summary()

        return df