import pandas as pd


class ResultsManager:
    def __init__(self):
        self.results = []

    def add(self, preproc, vect, model, val_f1, test_f1):
        self.results.append({
            "preprocessing": str(preproc),
            "vectorizer": str(vect),
            "model": str(model),
            "val_f1": val_f1,
            "test_f1": test_f1
        })

    def to_dataframe(self):
        return pd.DataFrame(self.results)

    def sort(self, df, metric="val_f1"):
        return df.sort_values(by=metric, ascending=False)

    def summary(self, top_n=5):
        df = self.to_dataframe()
        df = self.sort(df)

        print("\nTOP RESULTADOS:\n")
        print(df.head(top_n))

        return df