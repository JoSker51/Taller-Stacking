# Genera el notebook del taller C3 Taller 3: Stacking aplicado a Yomi Level.
# Uso: python build_nb.py  ->  Taller_Stacking_Yomi_Level.ipynb (sin ejecutar)
import nbformat as nbf
import textos as T

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(T.INTRO)

code(r'''
# ============================================================
# Librerías y estilo de las gráficas
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report)
warnings.filterwarnings('ignore')
os.makedirs('figuras', exist_ok=True)
sns.set_theme(style='whitegrid', font_scale=0.95)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
ORDEN = ['Fácil', 'Intermedio', 'Difícil']
COLOR = {'KNN': '#9AA3AD', 'SVM': '#7C93B0', 'Random Forest': '#2B4A6F',
         'Voting Soft': '#F2A900', 'Stacking': '#D9480F'}
''')

md(T.FASE1)
code(r'''
# ============================================================
# 1 · Cargar y preparar el dataset (mismo preprocesamiento de todos los talleres)
# ============================================================
# El CSV se lee junto al notebook; en Colab, si no se subió, se descarga del repositorio
ruta = ('moe_db_anilist_dataset.csv' if os.path.exists('moe_db_anilist_dataset.csv')
        else 'https://raw.githubusercontent.com/JoSker51/Taller-Stacking/main/moe_db_anilist_dataset.csv')
df = pd.read_csv(ruta)
df = df[df['entry_type'].notna()].copy()                      # quitamos títulos sin metadatos

# Objetivo: dificultad (1-100) en terciles -> 3 clases del mismo tamaño
df['nivel'], cortes = pd.qcut(df['Overall Difficulty (1-100)'], 3, labels=ORDEN, retbins=True)

# Faltantes con la mediana de su mismo formato; formato y géneros como columnas 0/1
for c in ['episodes', 'start_year']:
    df[c] = df[c].fillna(df.groupby('entry_type')[c].transform('median'))
df['genero_desconocido'] = df['genres'].isna().astype(int)
df['es_anime'] = df['entry_type'].str.startswith('anime').astype(int)
df['es_pelicula'] = df['entry_type'].str.endswith('movie').astype(int)
g = (df['genres'].fillna('')
     .str.replace('Sci-Fi & Fantasy', 'Sci-Fi, Fantasy', regex=False)
     .str.replace('Action & Adventure', 'Action, Adventure', regex=False)
     .str.replace('War & Politics', 'War', regex=False)
     .str.replace('Science Fiction', 'Sci-Fi', regex=False))
gen = g.str.get_dummies(sep=', ')
gen = gen.loc[:, gen.sum() >= 100]
gen.columns = ['g_' + c for c in gen.columns]
df = pd.concat([df, gen], axis=1)
for c in ['Avg. Words/Episode', 'Unique Words', 'episodes']:
    df['log_' + c] = np.log1p(df[c])

# 28 variables. Sin Kanji/Vocab Difficulty ni Vocab Density: la etiqueta se calcula con ellas (fuga de datos)
features = (['log_Avg. Words/Episode', 'log_Unique Words', 'log_episodes', 'start_year',
             'es_anime', 'es_pelicula', 'genero_desconocido'] + list(gen.columns))
X, y = df[features], df['nivel'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Títulos: {len(df):,} | variables: {len(features)} | train: {len(X_train):,} | test: {len(X_test):,}".replace(',', '.'))
print("Balance:", y.value_counts().reindex(ORDEN).to_dict(), "| cortes:", np.round(cortes[1:3], 1))
''')

md(T.FASE2)
code(r'''
# ============================================================
# 2 · Modelos base (nivel 0): los parámetros que encontramos en cada taller
# ============================================================
modelos_base = {
    # Taller KNN: GridSearchCV (5 folds) -> K=39, Manhattan, pesos por distancia
    'KNN': make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=39, metric='manhattan', weights='distance')),
    # Taller SVM: GridSearch de 20 combinaciones -> RBF, C=100, gamma='scale'
    'SVM': make_pipeline(StandardScaler(),
                         SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42)),
    # Proyecto final: 300 árboles, profundidad 15, 40% de variables por división
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, max_features=0.4,
                                            random_state=42, n_jobs=-1),
}

def errores_graves(real, pred):
    """% de títulos donde se confunde un extremo con el otro (Fácil <-> Difícil)."""
    real, pred = np.asarray(real), np.asarray(pred)
    return np.mean(((real == 'Fácil') & (pred == 'Difícil')) | ((real == 'Difícil') & (pred == 'Fácil')))

def evaluar(modelo):
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, [list(modelo.classes_).index(c) for c in ORDEN]]
    return pred, {'Accuracy': accuracy_score(y_test, pred),
                  'F1-macro': f1_score(y_test, pred, average='macro'),
                  'AUC': roc_auc_score(label_binarize(y_test, classes=ORDEN), proba, average='macro'),
                  'Recall Difícil': recall_score(y_test, pred, labels=['Difícil'], average='macro'),
                  'Errores graves': errores_graves(y_test, pred)}

pred, tabla = {}, {}
for nombre, modelo in modelos_base.items():
    modelo.fit(X_train, y_train)
    pred[nombre], tabla[nombre] = evaluar(modelo)
pd.DataFrame(tabla).T.round(3)
''')

md(T.FASE3)
code(r'''
# ============================================================
# 3 · Stacking: los 3 modelos base + un meta-modelo (nivel 1) que aprende a combinarlos
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stacking = StackingClassifier(
    estimators=list(modelos_base.items()),            # nivel 0: los "expertos"
    final_estimator=LogisticRegression(max_iter=2000), # nivel 1: el "juez", simple a propósito
    cv=cv,                                             # el juez aprende con predicciones out-of-fold
    stack_method='predict_proba',                      # le pasamos probabilidades, no solo etiquetas
    n_jobs=-1)
stacking.fit(X_train, y_train)
pred['Stacking'], tabla['Stacking'] = evaluar(stacking)

# Voting Soft del taller anterior, para comparar "democracia" contra "meritocracia"
voting = VotingClassifier(list(modelos_base.items()), voting='soft', n_jobs=-1).fit(X_train, y_train)
pred['Voting Soft'], tabla['Voting Soft'] = evaluar(voting)

orden_m = ['KNN', 'SVM', 'Random Forest', 'Voting Soft', 'Stacking']
resultados = pd.DataFrame(tabla).T.loc[orden_m]
resultados.round(3)
''')

code(r'''
# Reporte por clase del Stacking
print(classification_report(y_test, pred['Stacking'], labels=ORDEN, digits=3))
''')

md(T.FASE4)
code(r'''
# ============================================================
# 4 · ¿A quién le cree el juez? Pesos aprendidos por el meta-modelo
# ============================================================
meta = stacking.final_estimator_
# Dentro del stacking las clases se codifican como 0, 1, 2; stacking.classes_ traduce el número al nombre
clases_base = [stacking.classes_[i] for i in stacking.estimators_[0].classes_]  # orden de las probabilidades de nivel 0
clases_meta = list(stacking.classes_)                                           # filas de coeficientes del meta-modelo
columnas = [f"{m}\nP({c})" for m in modelos_base for c in clases_base]
pesos = pd.DataFrame(meta.coef_, index=clases_meta, columns=columnas).loc[ORDEN]
pesos = pesos[[f"{m}\nP({c})" for m in modelos_base for c in ORDEN]]   # columnas en orden Fácil, Intermedio, Difícil

fig, ax = plt.subplots(figsize=(12, 3.6))
sns.heatmap(pesos, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'peso'}, ax=ax)
ax.set_title('Pesos del meta-modelo: cuánto cuenta cada probabilidad de cada experto al decidir cada nivel')
ax.set_ylabel('Nivel que decide el juez'); ax.set_xlabel('')
plt.tight_layout(); plt.savefig('figuras/3_pesos_meta.png'); plt.show()

# Peso que cada experto le da a "su propio voto" (P(clase) para decidir esa misma clase)
confianza = pd.DataFrame({m: [pesos.loc[c, f"{m}\nP({c})"] for c in ORDEN] for m in modelos_base}, index=ORDEN)
confianza.round(2)
''')

md(T.FASE5)
code(r'''
# ============================================================
# 5 · Comparación visual: modelos base vs Voting vs Stacking
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
lims = {'Accuracy': (0.50, 0.62), 'F1-macro': (0.50, 0.62), 'AUC': (0.70, 0.79), 'Errores graves': (0, 0.09)}
for ax, (met, lim) in zip(axes, lims.items()):
    v = resultados[met]
    barras = ax.bar(v.index, v.values, color=[COLOR[m] for m in v.index])
    for b in barras:
        txt = f"{b.get_height():.1%}" if met == 'Errores graves' else f"{b.get_height():.3f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (lim[1] - lim[0]) * 0.015, txt, ha='center', fontsize=8.5)
    ax.set_ylim(*lim); ax.set_title(met + (' (menos es mejor)' if met == 'Errores graves' else ''))
    ax.tick_params(axis='x', rotation=35)
plt.tight_layout(); plt.savefig('figuras/4_comparacion.png'); plt.show()
''')

code(r'''
# Matrices de confusión: mejor modelo solo vs Voting vs Stacking
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
for ax, nombre, cmap in zip(axes, ['Random Forest', 'Voting Soft', 'Stacking'], ['Blues', 'YlOrBr', 'Oranges']):
    cm = confusion_matrix(y_test, pred[nombre], labels=ORDEN)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax, xticklabels=ORDEN, yticklabels=ORDEN)
    ax.set_title(f"{nombre} · F1 {resultados.loc[nombre, 'F1-macro']:.3f}")
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
plt.tight_layout(); plt.savefig('figuras/5_matrices.png'); plt.show()

graves = pd.DataFrame({n: {'Fácil -> Difícil': confusion_matrix(y_test, pred[n], labels=ORDEN)[0, 2],
                           'Difícil -> Fácil': confusion_matrix(y_test, pred[n], labels=ORDEN)[2, 0]}
                       for n in orden_m}).T
graves
''')

md(T.FASE6)
code(r'''
# ============================================================
# 6 · ¿Qué meta-modelo usar? Probamos 3 "jueces" y la opción passthrough
# ============================================================
candidatos = {
    'Regresión Logística (elegido)': dict(final_estimator=LogisticRegression(max_iter=2000)),
    'Regresión Logística C=0,1 (más estricta)': dict(final_estimator=LogisticRegression(C=0.1, max_iter=2000)),
    'Random Forest pequeño (profundidad 3)': dict(final_estimator=RandomForestClassifier(
        n_estimators=200, max_depth=3, random_state=42, n_jobs=-1)),
    'Regresión Logística + passthrough': dict(final_estimator=LogisticRegression(max_iter=2000), passthrough=True),
}
filas = {}
for nombre, params in candidatos.items():
    if nombre.endswith('(elegido)'):
        _, filas[nombre] = pred['Stacking'], tabla['Stacking']
        continue
    s = StackingClassifier(list(modelos_base.items()), cv=cv, n_jobs=-1, **params).fit(X_train, y_train)
    _, filas[nombre] = evaluar(s)
metas = pd.DataFrame(filas).T
metas.round(4)
''')

md(T.FASE7)
code(r'''
# ============================================================
# 7 · ¿Por qué la validación cruzada es obligatoria? Stacking SIN out-of-fold
# ============================================================
# Entrenamos el juez con las predicciones que cada modelo hace sobre sus PROPIOS datos de entrenamiento
P_train, P_test, acierto_train = [], [], {}
for nombre, modelo in modelos_base.items():
    m = clone(modelo).fit(X_train, y_train)
    P_train.append(m.predict_proba(X_train)); P_test.append(m.predict_proba(X_test))
    acierto_train[nombre] = accuracy_score(y_train, m.predict(X_train))
juez_tramposo = LogisticRegression(max_iter=2000).fit(np.hstack(P_train), y_train)
pred_sin_cv = juez_tramposo.predict(np.hstack(P_test))

print("Acierto de cada modelo sobre sus propios datos de entrenamiento:")
for k, v in acierto_train.items():
    print(f"  {k:14s} {v:.1%}")
pesos_sin_cv = pd.DataFrame(juez_tramposo.coef_, index=juez_tramposo.classes_, columns=columnas).loc[ORDEN]
print("\nPeso de cada experto en su propio voto, SIN validación cruzada:")
print(pd.DataFrame({m: [pesos_sin_cv.loc[c, f'{m}\nP({c})'] for c in ORDEN] for m in modelos_base}, index=ORDEN).round(2))
comp_cv = pd.DataFrame({'Sin validación cruzada': {'Accuracy': accuracy_score(y_test, pred_sin_cv),
                                                  'F1-macro': f1_score(y_test, pred_sin_cv, average='macro')},
                        'Con validación cruzada (cv=5)': {'Accuracy': tabla['Stacking']['Accuracy'],
                                                         'F1-macro': tabla['Stacking']['F1-macro']}}).T
comp_cv.round(4)
''')

md(T.FASE8)
code(r'''
# ============================================================
# 8 · ¿Qué pasa si quitamos un experto del stacking?
# ============================================================
ablacion = {'Los 3 modelos': tabla['Stacking']['F1-macro']}
for quitar in modelos_base:
    est = [(k, m) for k, m in modelos_base.items() if k != quitar]
    s = StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000), cv=cv, n_jobs=-1).fit(X_train, y_train)
    ablacion[f'Sin {quitar}'] = f1_score(y_test, s.predict(X_test), average='macro')
ablacion = pd.Series(ablacion)

fig, ax = plt.subplots(figsize=(9, 3.8))
cols = ['#D9480F'] + ['#B9C3D0'] * 3
barras = ax.barh(ablacion.index[::-1], ablacion.values[::-1], color=cols[::-1])
for b in barras:
    ax.text(b.get_width() + 0.0007, b.get_y() + b.get_height() / 2, f"{b.get_width():.3f}", va='center', fontsize=9)
ax.axvline(tabla['Random Forest']['F1-macro'], color='#2B4A6F', linestyle='--', label='Random Forest solo')
ax.set_xlim(0.57, 0.605); ax.set_xlabel('F1-macro'); ax.legend(loc='lower right')
ax.set_title('Stacking al quitar un modelo base')
plt.tight_layout(); plt.savefig('figuras/6_ablacion.png'); plt.show()
ablacion.round(4)
''')

md(T.HALLAZGOS)
md(T.CONCLUSION)

nb = nbf.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'Taller_Stacking_Yomi_Level.ipynb')
print('notebook generado')
