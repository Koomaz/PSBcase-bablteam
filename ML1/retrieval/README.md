# Retrieval для базы знаний ПСБ

Модуль работает на Python 3.10+ и использует стандартную библиотеку. В репозитории можно импортировать `retrieval.Retriever`. Для автопроверки используйте корневой модуль `retrieval_runtime.py`.

```bash
python -m scripts.build_index
python -m scripts.test_retrieval
python -m scripts.evaluate_retrieval
python -m scripts.export_retrieval
```

Готовые локальные индексы находятся в `artifacts/`. Для пересборки с обязательными эмбеддингами организаторов добавьте `--require-dense`. Для полностью локальной сборки добавьте `--offline`. Поддерживаются каталог базы и ZIP через `--source`. При наличии BGE-векторов загрузка индекса не вызывает API. Поиск считает только вектор запроса.

Примеры подключения и контракт для генерации приведены в [отчёте для ML2](../reports/retrieval_report_for_ML2.md). Результаты по всем демо-вопросам доступны в [отчёте о проверке](../reports/evaluation_report.md).
