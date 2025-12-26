============================================AI Generated===============================

# Intent Classification Project

Проект по классификации намерений пользователя в диалоговых системах с использованием различных подходов машинного обучения.

## Структура проекта

```
ml_proj/
├── project/                       # Весь код проекта
│   ├── README.md                  # Этот файл
│   ├── requirements.txt           # Зависимости проекта
│   ├── ml/                        # ML код
│   │   ├── data/                  # Работа с данными
│   │   │   ├── preprocessing.py   # Предобработка текстов
│   │   │   ├── load_banking_datasets.py  # Загрузка банковских датасетов
│   │   │   └── banking_class_mapping.py  # Маппинг классов
│   │   ├── models/                # Модели машинного обучения
│   │   │   └── classical_models.py  # Классические ML модели
│   │   ├── scripts/               # Исполняемые скрипты
│   │   │   ├── train_with_cv.py   # Обучение моделей
│   │   │   ├── predict.py         # Предсказание намерений
│   │   │   ├── preprocess_data.py # Скрипт предобработки
│   │   │   └── create_visualizations.py  # Создание визуализаций
│   │   └── utils/                 # Утилиты
│   │       └── visualizations.py # Функции визуализации
│   └── backend/                   # Backend код (если есть)
├── models/                        # Сохраненные обученные модели
├── data/                          # Данные
│   └── processed/                 # Обработанные датасеты
├── results/                       # Результаты экспериментов
├── presentations/                 # Презентация проекта
└── notebooks/                     # Jupyter notebooks для анализа
```

## Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас установлен Python 3.8 или выше.

## Использование

### Загрузка и объединение датасетов

Для загрузки и объединения датасетов выполните:

```bash
cd project
python ml/data/load_banking_datasets.py
```

Скрипт выполнит следующие действия:
1. Загрузит датасеты из Hugging Face:
   - `mteb/banking77` - банковские запросы (77 классов)
   - `pilarllera/BANKING49` - банковские запросы (49 классов)
2. Приведет данные из banking77 к 49 классам BANKING49
3. Объединит датасеты в единый формат
4. Сохранит объединенный датасет в `../data/processed/banking49_*.csv`
5. Создаст маппинг меток в `../data/processed/banking49_label_mapping.json`

### Датасеты

**Banking77 Dataset (mteb/banking77)**
- Банковские запросы на английском языке
- 77 классов намерений
- Примеры: activate_my_card, card_arrival, balance_not_updated

**BANKING49 Dataset (pilarllera/BANKING49)**
- Банковские запросы на английском языке
- 49 классов намерений
- Примеры: activate_my_card, card_arrival_delivery_estimate, fee_charged

### Обучение моделей

Для обучения моделей с 5-fold cross-validation и калибровкой вероятностей используйте:

```bash
cd project
python ml/scripts/train_with_cv.py --model lr    # Logistic Regression
python ml/scripts/train_with_cv.py --model rf    # Random Forest
python ml/scripts/train_with_cv.py --model both   # Обе модели
```

**Процесс обучения:**
1. **5-Fold Stratified Cross-Validation** для оценки модели
   - Данные разбиваются на 5 частей с сохранением распределения классов
   - Модель обучается на 4 частях, тестируется на 1
   - Процесс повторяется 5 раз
   - Результат: OOF (Out-Of-Fold) предсказания для анализа
2. **Обучение финальной модели** на 80% train+val данных
3. **Калибровка вероятностей** методом Isotonic Regression на 20% train+val данных
   - Повышает уверенность модели в предсказаниях
   - Улучшает калибровку вероятностей
4. **Финальная оценка** на независимом test set

**Результаты (с калибровкой):**
- Logistic Regression: 85.04% (CV), 84.14% (Test)
  - Уверенность: 72.92% (после калибровки, +16.14 п.п.)
- Random Forest: 65.90% (CV), 67.51% (Test)
  - Уверенность: 64.43% (после калибровки, +52.27 п.п.)

**Созданные файлы:**
- `../results/*_oof_predictions.csv` - OOF-предсказания для анализа
- `../models/*_model.pkl` - обученные модели
- `../models/*_vectorizer.pkl` - векторизаторы

## Разработка

### Модули

**Данные (`ml/data/`):**
- `preprocessing.py` - предобработка текстов
- `load_banking_datasets.py` - загрузка банковских датасетов
- `banking_class_mapping.py` - маппинг классов между датасетами

**Модели (`ml/models/`):**
- `classical_models.py` - классические модели ML

**Скрипты (`ml/scripts/`):**
- `train_with_cv.py` - обучение моделей с cross-validation и калибровкой
- `predict.py` - предсказание намерений
- `preprocess_data.py` - скрипт предобработки данных
- `create_visualizations.py` - создание визуализаций

**Утилиты (`ml/utils/`):**
- `visualizations.py` - функции для визуализации

### Добавление новых датасетов

Для добавления нового датасета:

1. Добавьте функцию загрузки в `ml/data/load_data.py`
2. Добавьте датасет в `ml/data/load_banking_datasets.py`
3. Запустите скрипт объединения


## Требования

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- datasets 2.14+
- pandas, numpy, scikit-learn
