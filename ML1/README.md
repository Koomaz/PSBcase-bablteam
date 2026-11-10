# ML1 Retrieval

Модуль поиска по базе знаний кейса ПСБ. В комплекте исходный пакет, готовые локальные индексы, скрипты проверки, результаты оценки и корневой модуль для сабмита.

## Быстрый запуск

Из корня репозитория выполните:

```powershell
cd ML1
python -X utf8 -m scripts.test_retrieval "п. 4.2 договора ПТ-05/2023" --explain
python -X utf8 -m scripts.evaluate_retrieval
```

Для полной проверки и пересборки поместите исходную базу организаторов в каталог `ML1/knowledge-base/`. Альтернативный источник для сборки передаётся через `--source` и может быть ZIP-архивом.

```powershell
python -X utf8 -m scripts.build_index
python -X utf8 -m scripts.test_retrieval
```

- [Подключение к генерации](reports/retrieval_report_for_ML2.md)
- [Анализ базы знаний](reports/kb_analysis.md)
- [Оценка качества поиска](reports/evaluation_report.md)
- [Манифест поставляемого индекса](artifacts/index_manifest.json)

Для автоматической проверки скопируйте `retrieval_runtime.py` в корень сабмита рядом с `solution.py`. Вызов `Retriever.from_data_dir()` загружает базу из каталога платформы. Для обновления корневого модуля запустите `python -m scripts.export_retrieval`.
