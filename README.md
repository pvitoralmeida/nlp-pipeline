# NLP Pipeline Dinâmico para Modelos Clássicos

Pipeline modular e configurável para análise de sentimentos utilizando técnicas de PLN + Machine Learning clássico, com suporte a experimentação automatizada de diferentes estratégias.

---

## Visão Geral

Este projeto permite:

* Aplicar diferentes estratégias de pré-processamento;
* Testar múltiplas abordagens de vetorização;
* Treinar diversos modelos clássicos;
* Comparar resultados de forma estruturada;
* Selecionar automaticamente as melhores configurações.

---

## Instalação

Clone o repositório e instale em modo desenvolvimento:

```bash
pip install -e .
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## Uso Rápido (Pipeline Básico)

```python
import pandas as pd
from nlp_pipeline.pipeline.pipeline import NLPPipeline
from config.configs import default_config

# carregar dataset
df = pd.read_csv("data/IMDB-Dataset.csv")

# caso deseje testar rapidamente, utilize um recorte do dataframe original
# df = df.sample(500)

X = df["review"].fillna("").astype(str)
y = df["sentiment"].map({"positive": 1, "negative": 0})

# instanciar pipeline
pipeline = NLPPipeline(default_config)

# treino
pipeline.fit(X, y)

# predição
preds = pipeline.predict(X[:5])

print(preds)
```

---

## Uso Avançado (Experimentação Automática)

```python
import pandas as pd
from nlp_pipeline.experiments import ExperimentRunner
from config.configs import default_config

df = pd.read_csv("data/IMDB-Dataset.csv")

# caso deseje testar rapidamente, utilize um recorte do dataframe original
# df = df.sample(500)

X = df["review"].fillna("").astype(str)
y = df["sentiment"].map({"positive": 1, "negative": 0})

config_space = {
    "preprocessing": [
        {"remove_stopwords": False},
        {"remove_stopwords": True},
        {"remove_stopwords": "keep_negations", "handle_negations": True}
    ],
    "vectorization": [
        {"method": "tfidf", "ngram_range": (1, 1)},
        {"method": "tfidf", "ngram_range": (1, 2)}
    ],
    "model": [
        {"model": "logreg", "search": "manual"},
        {"model": "svm", "search": "manual"}
    ]
}

runner = ExperimentRunner(
    base_config=default_config,
    config_space=config_space,
    strategy="random",  # "grid" ou "random"
    n_samples=3
)

results = runner.run(X, y)
```

---

## Configuração

### `default_config`

Define uma execução padrão da pipeline:

```python
default_config = {
    "preprocessing": {...},
    "vectorization": {...},
    "model": {...}
}
```

### `config_space`

Define o espaço de busca para experimentos:

```python
config_space = {
    "preprocessing": [...],
    "vectorization": [...],
    "model": [...]
}
```

---

## Componentes do Pipeline

### Pré-processamento

* Lowercase;
* Remoção de stopwords (com opção de preservar negações);
* Lematização / stemming;
* Tratamento de negações;
* Remoção de URLs;
* Normalização de emojis.

---

### Vetorização

* Bag-of-Words (BoW);
* TF-IDF;
* TF-IDF + Truncated SVD;
* Word2Vec (média dos embeddings).

---

### Modelos

* Naive Bayes;
* Regressão Logística;
* Linear SVM;
* Random Forest;
* LightGBM.

---

### Seleção de Hiperparâmetros

* Manual;
* Grid Search;
* Random Search;

---

## Saída dos Experimentos

A execução gera:

* Tabela ordenada por desempenho (F1-macro)

Exemplo:

```
preprocessing | vectorizer | model | val_f1 | test_f1
----------------------------------------------------
...           | ...        | ...   | 0.89   | 0.88
```

---

## Estrutura do Projeto

```
nlp_pipeline/
│
├── pipeline/
├── preprocessing/
├── vectorization/
├── models/
├── experiments/
│
config/
data/
```

---

## Autor

Projeto desenvolvido como parte de estudos em PLN e Machine Learning por Paulo Vitor.
