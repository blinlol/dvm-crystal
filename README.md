# пайплайн сбора данных

1) в auto test
- разбиваем программу `all/x.fdv` на программы `data/x/x<i>.fdv`
- проверяем, что все работает с помощью `check_all_programs.sh` и `check_programs.sh`
2) `cp -r dvm-auto-test/data dvm-crystal-data`
3) в dvm-crystal-data
- задаем параметры в param_values.py
- `python run.py --max-cores <x> --max-threads <y>`
4) в dvm-crystal
- проверяем конфиг в `config/data_config.yaml`, значения data_collection.default_source_dir, data_aggregation.results_directory
- `python main.py collect`
- `python main.py aggregate`
- `python main.py features`


1) запуск в crystal-data
2) python main.py collect
3) python main.py aggregate
4) python main.py features


# DVMH Performance Predictor

Система машинного обучения для предсказания эффективности параллельных программ с использованием технологии DVMH (Distributed Virtual Memory Hierarchy).

## 📋 Описание проекта

DVMH Performance Predictor - это комплексная система, которая анализирует статические характеристики DVMH программ и предсказывает их производительность при различных конфигурациях параллельного запуска. Система использует современные методы машинного обучения, включая нейронные сети с механизмом внимания.

### 🎯 Основные возможности

- **Автоматический сбор данных** из Fortran исходных файлов
- **Извлечение признаков** из статистики покрытия кода
- **Обучение моделей** (MLP и Attention-based архитектуры)
- **Предсказание ускорения** для новых программ
- **Дообучение моделей** на новых данных
- **Анализ важности признаков** и визуализация результатов

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## Основные компоненты
```bash
# Установите PyTorch для обучения моделей
pip install torch torchvision

# Установите дополнительные зависимости
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml
```

## Базовое использование
```bash
# 1. Сбор данных из исходных программ
python main.py collect --source ./sources --results ./data/raw

# 2. Полный пайплайн: агрегация + создание признаков
python main.py pipeline --config config/data_config.yaml

# 3. Обучение моделей
python scripts/train.py --dataset data/processed/feature_dataset.csv

# 4. Предсказание для новой программы
python scripts/predict.py program.f --grid "2,2" --threads 4 --time 1.5
```

## 📁 Структура проекта
```bash
dvmh-performance-predictor/
├── config/
│   └── data_config.yaml          # Конфигурационный файл
├── src/
│   ├── data_collector.py         # Сбор данных покрытия
│   ├── stats_aggregator.py       # Агрегация статистики
│   ├── feature_extractor.py      # Извлечение признаков
│   ├── feature_preprocessor.py   # Предобработка данных
│   ├── model_trainer.py          # Обучение моделей
│   ├── model_finetuner.py        # Дообучение моделей
│   ├── predictor.py              # Система предсказаний
│   └── utils.py                  # Утилиты
├── scripts/
│   ├── train.py                  # Скрипт обучения
│   ├── predict.py                # Скрипт предсказания
│   └── finetune.py               # Скрипт дообучения
├── main.py                       # Главный модуль
└── README.md
```

Пример папки с данными по параллельным запускам
```bash
parallel_results/
├── 1d/
│   ├── program_name
│       ├── p1_t1_1
│           ├── stat.txt
├── 2d/
│   ├── program_name
│       ├── p1_t1_1x1
│           ├── stat.txt
├── 3d/
│   ├── program_name
│       ├── p1_t1_1x1x1
│           ├── stat.txt
├── 4d/
│   ├── program_name
│       ├── p1_t1_1x1x1x1
│           ├── stat.txt
```

## 🔧 Детальное использование
### 1. Сбор данных
Собирает статистику покрытия из Fortran файлов:
```bash
python main.py collect --source ./sources --results ./data/raw
```
#### Параметры:
- --source - директория с исходными программами
- --results - директория для сохранения результатов

```bash
sources/
├── 1d/
│   └── program.f
├── 2d/
│   ├── program.f
├── 3d/
│   ├── program.f
├── 4d/
│   ├── program.f
```

### 2. Агрегация и создание признакового пространства
```bash
# Только агрегация
python main.py aggregate --config config/data_config.yaml

# Только создание признаков
python main.py features --input aggregated_data.json --output dataset.csv

# Полный пайплайн
python main.py pipeline --config config/data_config.yaml
```

### 3. Обучение моделей
```bash
python scripts/train.py \
    --dataset data/processed/feature_dataset.csv \
    --epochs 150 \
    --batch-size 64 \
    --hidden-dims 512 256 128 \
    --test-programs 5
```
#### Основные параметры:

- --dataset - путь к датасету
- --epochs - количество эпох обучения
- --batch-size - размер батча
- --hidden-dims - архитектура скрытых слоев
- --test-programs - количество программ для тестирования

### 4. Предсказание эффективности
```bash
# Из Fortran файла
python scripts/predict.py program.f \
    --grid "2,2" \
    --threads 4 \
    --time 1.5 \
    --model dvmh_attention_model

# Из JSON файла статистики
python scripts/predict.py info.json \
    --grid "1,2,3" \
    --threads 8 \
    --time 2.1 \
    --output-format detailed
```

#### Параметры:

- --grid - сетка процессоров (формат: "x,y,z" или "xyz")
- --threads - количество нитей
- --time - время параллельного выполнения
- --model - имя модели для использования
- --output-format - формат вывода (simple/json/detailed)

### 5. Дообучение моделей
```bash
python scripts/finetune.py \
    --base-model dvmh_mlp_model \
    --new-data new_programs.csv \
    --strategies "full_model,last_layers,gradual_unfreezing" \
    --lr 0.0001 \
    --epochs 50
```

#### Стратегии дообучения:

- full_model - дообучение всей модели
- last_layers - дообучение последних слоев
- adapter - добавление адаптерных слоев
- gradual_unfreezing - постепенное размораживание слоев

## 🧠 Архитектура моделей
### MLP модель (DVMHEfficiencyMLP)
```python
# Многослойный перцептрон с BatchNorm и Dropout
Hidden layers: [256, 128, 64]
Activation: ReLU
Regularization: Dropout (0.3), BatchNorm
Output: Single value (speedup prediction)
```
### Attention модель (DVMHAttentionModel)
```python
# Модель с механизмом внимания
Embedding: Linear(input_dim, hidden_dims[0])
Attention: Query-Key-Value mechanism
Hidden layers: [128, 64]
Output: Speedup + attention weights
```
