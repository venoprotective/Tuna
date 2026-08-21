from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from tuner import Tuner

print("="*60)
print("ТЕСТ TUNER")
print("="*60)

# ============================================
# ТЕСТ 1: КЛАССИФИКАЦИЯ
# ============================================
print("\n" + "="*60)
print("ТЕСТ 1: КЛАССИФИКАЦИЯ")
print("="*60)

X_clf, y_clf = make_classification(
    n_samples=500,
    n_features=10,           
    n_informative=8,         
    n_redundant=2,           
    n_repeated=0,            
    n_classes=2,
    random_state=42
)

X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(X_clf.shape[1])])
y_clf = pd.Series(y_clf, name='target')

print(f"Данные: {X_clf.shape}, классов: {len(np.unique(y_clf))}")

tuner_clf = Tuner(
    model=['rf', 'xgboost', 'lgbm'],
    X=X_clf,
    y=y_clf,
    type_model='classification',
    searcher='RandomizedSearchCV',
    cv=3,
    n_iter=10,
    random_state=42
)

tuner_clf.run()

# Получаем результаты
print("\n" + "-"*40)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
results_clf = tuner_clf.get_results()
print(results_clf)

# ============================================
# ТЕСТ 2: РЕГРЕССИЯ
# ============================================
print("\n" + "="*60)
print("ТЕСТ 2: РЕГРЕССИЯ")
print("="*60)

X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,            
    n_informative=8,          
    noise=0.1,
    random_state=42
)

X_reg = pd.DataFrame(X_reg, columns=[f'f{i}' for i in range(X_reg.shape[1])])
y_reg = pd.Series(y_reg, name='target')

print(f"Данные: {X_reg.shape}, диапазон target: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

tuner_reg = Tuner(
    model=['rf', 'xgboost', 'lgbm'],
    X=X_reg,
    y=y_reg,
    type_model='regression',
    searcher='RandomizedSearchCV',
    cv=3,
    n_iter=10,
    random_state=42
)

tuner_reg.run()

print("\n" + "-"*40)
print("РЕЗУЛЬТАТЫ РЕГРЕССИИ:")
results_reg = tuner_reg.get_results()
print(results_reg)

# ============================================
# ТЕСТ 3: БЕЗ ПОИСКА (searcher=None)
# ============================================
print("\n" + "="*60)
print("ТЕСТ 3: БЕЗ ПОИСКА (searcher=None)")
print("="*60)

tuner_no_search = Tuner(
    model=['rf'],
    X=X_clf,
    y=y_clf,
    type_model='classification',
    searcher=None,
    cv=None,
    random_state=42
)

tuner_no_search.run()
results_no_search = tuner_no_search.get_results()
print("\nРезультаты без поиска:")
print(results_no_search)


# ============================================
# ТЕСТ 4: GridSearchCV (простая сетка)
# ============================================
print("\n" + "="*60)
print("ТЕСТ 4: GridSearchCV")
print("="*60)

# Берем маленькие данные для GridSearch
X_small, y_small = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

X_small = pd.DataFrame(X_small)
y_small = pd.Series(y_small)

tuner_grid = Tuner(
    model=['rf'],
    X=X_small,
    y=y_small,
    type_model='classification',
    searcher='GridSearchCV',
    cv=3,
    random_state=42
)

tuner_grid.run()
results_grid = tuner_grid.get_results()
print("\nРезультаты GridSearchCV:")
print(results_grid)

# ============================================
# ТЕСТ 5: ИНТЕРАКТИВНЫЙ РЕЖИМ (раскомментируй если хочешь)
# ============================================
print("\n" + "="*60)
print("ТЕСТ 5: ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("="*60)

print("Сейчас будет интерактивный выбор моделей...")

X_inter, y_inter = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

X_inter = pd.DataFrame(X_inter)
y_inter = pd.Series(y_inter)

tuner_inter = Tuner(
    model=None,  
    X=X_inter,
    y=y_inter,
    type_model='classification',
    searcher='RandomizedSearchCV',
    cv=3,
    n_iter=10,
    random_state=42
)

tuner_inter.run()
results_inter = tuner_inter.get_results()
print(results_inter)

# ============================================
# ИТОГ
# ============================================
print("\n" + "="*60)
print("ИТОГ ТЕСТИРОВАНИЯ")
print("="*60)

print(f"""
Тесты завершены:
✅ Классификация - выполнена
✅ Регрессия - выполнена
✅ Без поиска - выполнена
✅ GridSearchCV - выполнена
✅ Интерактивный - готов к запуску

ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!
""")