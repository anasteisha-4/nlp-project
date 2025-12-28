# Two-Stage Information Retrieval System

Двухэтапная поисковая система: **Retrieve → Rerank** для польского языка.

## Задача

Построить IR-систему с двумя этапами:

1. **Retriever (bi-encoder)** — быстрый поиск кандидатов по эмбеддингам
2. **Reranker (cross-encoder)** — точное переранжирование top-k документов

## Датасет

**[WebFAQ Retrieval](https://huggingface.co/datasets/PaDaS-Lab/webfaq-retrieval)** — польский язык (`pol`)

| Компонент | Описание |
|-----------|----------|
| `corpus.jsonl` | Уникальные документы (ответы) |
| `queries.jsonl` | Вопросы |
| `train.jsonl` / `test.jsonl` | Аннотации релевантности |

## Архитектура проекта

```
nlp-project/
├── models/                  # Обученные модели
├── src/
│   ├── retriever/           # Скрипты для Retriever
│   │   ├── train_retriever.py
│   │   └── prepare_train_data_retriever.py
│   ├── reranker/            # Скрипты для Reranker
│   │   ├── train_reranker.py
│   │   ├── prepare_train_data_reranker.py
│   │   └── retrieve_data.py
│   ├── pipeline/            # Пайплайн и оценка
│   │   ├── retrieve_rerank.py      # Основной пайплайн
│   │   └── evaluate_reranker.py    # Скрипт оценки
│   ├── evaluate_retriever.py
│   └── metrics.py
├── images/                  # Графики результатов
└── requirements.txt         # Зависимости
```

## Модели

### Retriever
- **Модель**: `intfloat/multilingual-e5-base`
- **Метод**: Bi-encoder (Fine-tuned)
- **Использование**: Первичный поиск top-100 кандидатов

### Reranker
- **Модель**: `sdadas/polish-reranker-bge-v2`
- **Метод**: Cross-encoder (Fine-tuned)
- **Использование**: Переранжирование кандидатов для получения финального top-10

## Метрики

| Модель | Recall@5 | Recall@10 | MRR | nDCG@10 |
|--------|----------|-----------|-----|---------|
| Baseline Retriever | 0.7975 | 0.8452 | 0.7046 | 0.7368 |
| Trained Retriever | 0.8470 | 0.8893 | 0.7542 | 0.7869 |
| Retriever + Reranker | 0.8692 | 0.9086 | 0.7808 | 0.8118 |

## Установка

```bash
pip install -r requirements.txt
```

## Использование

Для запуска полного пайплайна поиска и реранкинга:

```bash
# 1. Запуск поиска и реранкинга (генерация результатов)
python src/pipeline/retrieve_rerank.py

# 2. Оценка результатов (подсчет метрик)
python src/evaluate_reranker.py
```

## Результаты

### Recall@k Comparison
![Recall@k](images/recall_at_k.png)

### MRR Comparison
![MRR](images/mrr.png)

### nDCG@10 Comparison
![nDCG@10](images/ndcg.png)

## Выводы

1.  **Эффективность обучения Retriever**: Дообучение Bi-encoder модели (`intfloat/multilingual-e5-base`) на целевом датасете позволило улучшить качество первичного поиска. Recall@10 вырос с 0.8452 (Baseline) до 0.8893. Это подтверждает, что адаптация модели под домен WebFAQ повышает полноту поиска.
2.  **Роль Reranker**: Внедрение второго этапа (Cross-encoder `sdadas/polish-reranker-bge-v2`) дало значительный прирост качества ранжирования. MRR вырос с 0.7542 до 0.7808, а nDCG@10 с 0.7869 до 0.8118. Это доказывает, что Cross-encoder эффективно уточняет выдачу, поднимая наиболее релевантные ответы в топ.
3.  **Итоговый результат**: Реализованный двухэтапный пайплайн (Retrieve + Rerank) достиг Recall@10 ~91% и nDCG@10 ~0.81, что является отличным показателем для вопросно-ответной системы.

## Команда

| Участник | Роль |
|------|-----------------|
| Анастасия Бронина | Архитектура проекта,интеграция компонентов, подготовка отчета |
| Софья Князева | Dense Retriever, FAISS, метрики |
| Александр Гусев | Cross-encoder Reranker, обучение |
