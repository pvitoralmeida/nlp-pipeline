# NLP Pipeline Dinâmico para Modelos Clássicos

Pipeline modular e configurável para análise de sentimentos utilizando técnicas de PLN + Machine Learning clássico, com suporte a experimentação automatizada de diferentes estratégias.

---

## Visão Geral

Este projeto permite:

* Aplicar diferentes estratégias de pré-processamento;
* Testar múltiplas abordagens de vetorização;
* Treinar diversos modelos clássicos;
* Comparar resultados de forma estruturada;
* Selecionar automaticamente as melhores configurações;
* Evitar combinações de configuração inválidas ou pouco adequadas;;

--- 

## Justificativa das escolhas de pré-processamento e vetorização

### Pré-processamento

* Tokenização e lematização — spaCy: integração entre etapas linguísticas (tokenização, lematização e stopwords). A lematização foi escolhida por preservar melhor o significado das palavras em comparação ao stemming.

* Remoção de stopwords — spaCy: reduzir ruído (foi incluída a opção keep_negations para preservar palavras como “não” e “nunca”, relevantes para a polaridade).

* Tratamento de negações (regra customizada): adição do sufixo _NEG em tokens após negações para diferenciar termos como “gostei” e “gostei_NEG”.

* Normalização de emojis — emoji: converter emojis em texto para preservar informação semântica em dados informais.

* Limpeza básica (lowercase e remoção de URLs): padronizar o texto e remover ruídos não relevantes.

### Vetorização

* TF-IDF — scikit-learn: capturar a importância relativa dos termos.

* N-grams (especialmente (1,2)): capturar contexto local.

* Sublinear TF: reduzir o impacto de termos muito frequentes, melhorando a distribuição dos pesos.

* TF-IDF + TruncatedSVD — scikit-learn: reduzir a dimensionalidade do espaço vetorial, mantendo a maior parte da variância dos dados.

* Word2Vec — Gensim: capturar relações semânticas entre palavras por meio de embeddings densos.

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
requirements.txt
README.md
```

## Análise dos Resultados

Para comprovar o funcionamento da pipeline, bem como comparar o desempenho das combinações de modelos de machine learning clássicos, técnicas de pré-processamento e vetorização em tarefas de PLN, 10 configurações aleatórias definidas a partir de um espaço de busca pré-estabelecido foram avaliadas. A partir disso, os resultados descritos abaixo puderam ser observados.

Os experimentos demonstraram que modelos lineares, como SVM e Regressão Logística, apresentaram desempenho superior em relação a modelos baseados em árvores, como Random Forest e LightGBM. Isso pode ser explicado pela natureza dos dados textuais, caracterizados por alta dimensionalidade e esparsidade, favorecendo modelos lineares.

Em relação à vetorização, o uso de TF-IDF com n-gramas de tamanho (1,2) apresentou os melhores resultados, indicando que a inclusão de bigramas contribui para capturar contexto relevante sem introduzir ruído excessivo. Já o uso de trigramas mostrou-se instável, possivelmente devido ao aumento significativo da dimensionalidade.

A aplicação de sublinear TF também contribuiu positivamente, reduzindo o impacto de termos muito frequentes e melhorando a capacidade de generalização do modelo.

No contexto deste experimento, observou-se que estratégias completas de pré-processamento, isto é, as que incluíram lematização, remoção de stopwords, tratamento de negações e normalização de emojis, resultaram em melhor desempenho médio. Isso sugere que, para este dataset e conjunto de modelos, a qualidade da representação textual teve impacto significativo nos resultados.

Por fim, foi possível identificar casos de overfitting, especialmente em algumas configurações envolvendo modelos baseados em árvores, nos quais houve discrepância significativa entre os resultados de validação e teste. Isso reforça a importância de avaliar modelos em dados não vistos para garantir robustez.

As configurações e métricas dos modelos analisados podem ser encontradas em `experiments/result.csv` e o experimento pode ser replicado de maneira análoga por meio do script `experiments/model_analysis.py`.

---

## Autor

Projeto desenvolvido como parte de estudos em PLN e Machine Learning por Paulo Vitor.
