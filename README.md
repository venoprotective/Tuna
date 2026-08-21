# Tuner — полуавтоматический подбор гиперпараметров ML-моделей

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-supported-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-supported-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM-supported-green)
![pandas](https://img.shields.io/badge/pandas-supported-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

`Tuner` — это Python-класс для удобного полуавтоматического подбора гиперпараметров моделей машинного обучения.  
Он позволяет быстро сравнить несколько моделей, запустить поиск параметров через `RandomizedSearchCV` или `GridSearchCV` и получить итоговую таблицу результатов.

---

## Содержание

- [Описание](#описание)
- [Возможности](#возможности)
- [Поддерживаемые модели](#поддерживаемые-модели)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Параметры конструктора](#параметры-конструктора)
- [Методы](#методы)
- [Атрибуты результата](#атрибуты-результата)
- [Примеры использования](#примеры-использования)
- [Пример вывода](#пример-вывода)
- [Сетки гиперпараметров](#сетки-гиперпараметров)
- [Важные замечания](#важные-замечания)
- [Рекомендации](#рекомендации)
- [Структура проекта](#структура-проекта)
- [Лицензия](#лицензия)
- [Вклад в проект](#вклад-в-проект)
- [Обратная связь](#обратная-связь)

---

## Описание

Класс `Tuner` автоматизирует рутину, связанную с:

- выбором моделей;
- определением типа задачи;
- запуском кросс-валидации;
- поиском гиперпараметров;
- сравнением качества моделей;
- сохранением лучших параметров.

Проект хорошо подходит для:

- быстрого прототипирования ML-решений;
- сравнения базовых моделей;
- подбора параметров для регрессии и классификации;
- учебных и исследовательских задач.

---

## Возможности

- Поддержка **регрессии** и **классификации**
- Автоматическое определение типа задачи по целевой переменной
- Работа с популярными моделями:
  - `linear`
  - `rf`
  - `gb`
  - `xgboost`
  - `lgbm`
  - `svm`
  - `knn`
- Использование:
  - `RandomizedSearchCV`
  - `GridSearchCV`
- Интерактивный выбор моделей через консоль
- Готовые предустановленные сетки гиперпараметров
- Возможность задать собственную метрику
- Итоговая таблица результатов в виде `pandas.DataFrame`
- Сортировка моделей по качеству

---

## Поддерживаемые модели

| Ключ | Регрессия | Классификация |
|---|---|---|
| `linear` | `LinearRegression` | `LogisticRegression` |
| `rf` | `RandomForestRegressor` | `RandomForestClassifier` |
| `gb` | `GradientBoostingRegressor` | `GradientBoostingClassifier` |
| `xgboost` | `XGBRegressor` | `XGBClassifier` |
| `lgbm` | `LGBMRegressor` | `LGBMClassifier` |
| `svm` | `SVR` | `SVC` |
| `knn` | `KNeighborsRegressor` | `KNeighborsClassifier` |

---

## Установка

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/ваш-ник/название-репозитория.git
cd название-репозитория
```

### 2. Установите зависимости

```bash
pip install scikit-learn xgboost lightgbm pandas
```

Или создайте файл `requirements.txt`:

```txt
scikit-learn>=1.3.0
pandas>=2.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

Затем выполните:

```bash
pip install -r requirements.txt
```

---

## Быстрый старт

Предположим, класс `Tuner` находится в файле `tuner.py`.

### Пример для регрессии

```python
from tuner import Tuner
from sklearn.datasets import load_diabetes

# Загрузка данных
X, y = load_diabetes(return_X_y=True)

# Создание тюнера
tuner = Tuner(
    model=["rf", "xgboost", "lgbm"],
    X=X,
    y=y,
    type_model="regression",
    searcher="RandomizedSearchCV",
    scoring="auto",
    cv=5,
    n_iter=20,
    random_state=42
)

# Запуск подбора параметров
tuner.run()

# Получение результатов
results = tuner.get_results()
print(results)
```

---

## Параметры конструктора

| Параметр | Тип | По умолчанию | Описание |
|---|---|---:|---|
| `model` | `str`, `list`, `None` | `None` | Модели для обучения: `'linear'`, `'rf'`, `'gb'`, `'xgboost'`, `'lgbm'`, `'svm'`, `'knn'`. Можно передать список моделей или `None` для интерактивного выбора. |
| `X` | `pd.DataFrame`, `np.ndarray` | `None` | Матрица признаков. |
| `y` | `pd.Series`, `np.ndarray` | `None` | Целевая переменная. |
| `type_model` | `str` | `'auto'` | Тип задачи: `'regression'`, `'classification'`, `'auto'`. |
| `searcher` | `str`, `None` | `'auto'` | Метод поиска: `'RandomizedSearchCV'`, `'GridSearchCV'`, `None`. |
| `scoring` | `str`, `callable` | `'auto'` | Метрика оценки. При `'auto'` для классификации используется `accuracy_score`, для регрессии — `r2_score`. |
| `cv` | `int`, `'auto'`, `None` | `'auto'` | Количество фолдов кросс-валидации. При `'auto'` используется 5. |
| `n_iter` | `int` | `20` | Количество итераций для `RandomizedSearchCV`. |
| `random_state` | `int` | `42` | Для воспроизводимости результатов. |

---

## Методы

### `run()`

Запускает процесс подбора гиперпараметров для всех выбранных моделей.

Что происходит внутри:

1. Для каждой модели определяется сетка гиперпараметров.
2. Если поиск разрешен, запускается `RandomizedSearchCV` или `GridSearchCV`.
3. Если поиск отключен, модель обучается с параметрами по умолчанию.
4. Сохраняются лучшие модели, параметры и оценки.
5. Выводится итоговая таблица результатов.

Пример:

```python
tuner.run()
```

---

### `get_results()`

Возвращает `pandas.DataFrame` с результатами сравнения моделей.

Результат отсортирован по убыванию метрики `score`.

```python
results = tuner.get_results()
```

Структура DataFrame:

| Колонка | Описание |
|---|---|
| `model` | Название модели |
| `score` | Лучшая оценка на кросс-валидации |
| `params` | Лучшие найденные гиперпараметры |

Если `run()` еще не был вызван, метод вернет `None` и выведет сообщение.

---

## Атрибуты результата

После вызова `run()` доступны:

| Атрибут | Описание |
|---|---|
| `best_models_` | Список лучших обученных моделей |
| `best_params_` | Список лучших найденных параметров |
| `best_scores_` | Список лучших оценок кросс-валидации |
| `results_df_` | Итоговая таблица результатов |

Пример:

```python
print(tuner.best_models_)
print(tuner.best_params_)
print(tuner.best_scores_)
print(tuner.results_df_)
```

---

## Примеры использования

### 1. Интерактивный выбор моделей

Если передать `model=None`, в консоли появится меню выбора.

```python
tuner = Tuner(
    model=None,
    X=X,
    y=y,
    type_model="auto"
)

tuner.run()
```

Пример приглашения:

```text
models:
linear, rf, gb, xgboost, lgbm, svm, knn
 "all" - pick all models
 "boost" - comparing boosting models(lgbm, gb, xgboost)
 "linear, rf, etc" - pick your models like list

Your choice:
```

Можно ввести, например:

```text
all
```

или:

```text
boost
```

или:

```text
linear, rf, xgboost
```

---

### 2. Классификация с GridSearchCV

```python
from tuner import Tuner
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

tuner = Tuner(
    model=["linear", "svm", "rf"],
    X=X,
    y=y,
    type_model="classification",
    searcher="GridSearchCV",
    cv=3,
    scoring="roc_auc",
    random_state=42
)

tuner.run()
```

---

### 3. Обучение без поиска гиперпараметров

Если не нужен подбор параметров, можно отключить поиск.

```python
tuner = Tuner(
    model=["linear", "knn"],
    X=X,
    y=y,
    searcher=None,
    cv=None
)

tuner.run()
```

В этом случае модели обучаются с параметрами по умолчанию.

---

### 4. Пользовательская метрика для регрессии

```python
from tuner import Tuner
from sklearn.metrics import make_scorer, mean_absolute_error

custom_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

tuner = Tuner(
    model=["rf", "gb"],
    X=X,
    y=y,
    type_model="regression",
    scoring=custom_scorer,
    cv=5,
    random_state=42
)

tuner.run()
```

> Для метрик, где меньшее значение лучше, например `MAE` или `MSE`, используйте `greater_is_better=False`.

---

### 5. Сравнение всех бустингов

```python
tuner = Tuner(
    model=["gb", "xgboost", "lgbm"],
    X=X,
    y=y,
    type_model="auto",
    searcher="RandomizedSearchCV",
    n_iter=30,
    cv=5,
    random_state=42
)

tuner.run()
```

---

## Пример вывода

```text
type: regression
amount models: 3
searcher: RandomizedSearchCV
cross-validation: 5 folds

[1/3] Handling RandomForestRegressor...
find hyperparams for RandomForestRegressor...
	Score: 0.8234
	 Params: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5}

[2/3] Handling XGBRegressor...
find hyperparams for XGBRegressor...
	Score: 0.8451
	 Params: {'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.9}

[3/3] Handling LGBMRegressor...
find hyperparams for LGBMRegressor...
	Score: 0.8523
	 Params: {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 200}

results
model                 score  params
LGBMRegressor         0.8523 {'num_leaves': 31, 'learning_rate': 0.1, ...}
XGBRegressor          0.8451 {'learning_rate': 0.05, 'max_depth': 5, ...}
RandomForestRegressor 0.8234 {'n_estimators': 200, 'max_depth': 10, ...}
```

---

## Сетки гиперпараметров

Внутри класса уже заданы базовые сетки параметров для основных моделей.

### Регрессия

| Модель | Основные подбираемые параметры |
|---|---|
| `LinearRegression` | `fit_intercept` |
| `RandomForestRegressor` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `GradientBoostingRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample` |
| `XGBRegressor` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `LGBMRegressor` | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `subsample`, `colsample_bytree` |
| `SVR` | `C`, `epsilon`, `kernel`, `gamma` |
| `KNeighborsRegressor` | `n_neighbors`, `weights`, `p` |

### Классификация

| Модель | Основные подбираемые параметры |
|---|---|
| `LogisticRegression` | `C`, `l1_ratio`, `solver`, `max_iter` |
| `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `subsample` |
| `XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `LGBMClassifier` | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `subsample`, `colsample_bytree` |
| `SVC` | `C`, `kernel`, `gamma`, `probability` |
| `KNeighborsClassifier` | `n_neighbors`, `weights`, `p` |

---

## Важные замечания

### Автоматическое определение типа задачи

Если указано `type_model='auto'`, тип задачи определяется автоматически по целевой переменной `y`:

- если уникальных значений `y` **меньше или равно 10** — задача считается **классификацией**;
- если уникальных значений `y` **больше 10** — задача считается **регрессией**.

Пример:

```python
tuner = Tuner(
    model=["rf", "lgbm"],
    X=X,
    y=y,
    type_model="auto"
)
```

---

### Автоматический выбор поисковика

Если указано `searcher='auto'`:

- при наличии `cv` будет использован `RandomizedSearchCV`;
- при `cv=None` поиск гиперпараметров выполнен не будет.

---

### Отключение поиска параметров

Если передать:

```python
searcher=None
cv=None
```

модели будут обучены без поиска гиперпараметров.

---

### Производительность

`GridSearchCV` выполняет полный перебор всех комбинаций параметров и может работать очень медленно на больших сетках.

Для быстрого результата рекомендуется использовать:

```python
searcher="RandomizedSearchCV"
```

и ограничивать количество итераций:

```python
n_iter=20
```

---

### Обработка ошибок

Если во время обучения модели возникает ошибка, `Tuner` не прерывает весь процесс.  
В этом случае модель может быть добавлена с параметрами по умолчанию, а оценка может быть `None`.

---

### Метрики

По умолчанию:

- для классификации используется `accuracy_score`;
- для регрессии используется `r2_score`.

Можно передать другую метрику в `scoring`, например:

```python
scoring="roc_auc"
```

или собственный scorer.

---

## Рекомендации

Для получения более стабильных и качественных результатов:

1. **Масштабируйте признаки** для:
   - `svm`
   - `knn`
   - `linear`

2. **Используйте RandomizedSearchCV** вместо GridSearchCV, если сетка параметров большая.

3. **Уменьшайте количество фолдов**, если данных много:
   ```python
   cv=3
   ```

4. **Ограничивайте `n_iter`** для быстрого прототипирования:
   ```python
   n_iter=10
   ```

5. **Проверяйте не только одну метрику**, особенно в задачах классификации с дисбалансом классов.

6. **Для больших датасетов** сначала проверяйте 2–3 модели, а затем расширяйте поиск.

---

## Структура проекта

Пример структуры репозитория:

```text
ваш-проект/
│
├── tuner.py
├── README.md
├── requirements.txt
└── examples/
    └── example_usage.py
```

---

## Лицензия

Проект распространяется под лицензией **MIT**.

---

## Вклад в проект

Если вы нашли ошибку или хотите предложить улучшение:

1. Создайте **Issue** с описанием проблемы.
2. Сделайте **Fork** проекта.
3. Внесите изменения.
4. Отправьте **Pull Request**.

Приветствуются:

- исправление багов;
- добавление новых моделей;
- улучшение документации;
- добавление тестов;
- оптимизация производительности.

---

## Обратная связь

Если у вас есть вопросы или предложения, создайте Issue в репозитории.

Если проект оказался полезным — поставьте звезду.
