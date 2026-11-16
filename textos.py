# Textos (markdown) del notebook del taller de Stacking. Todos los números salen de la ejecución del notebook.

INTRO = r'''
# Taller C3 · Stacking aplicado a Yomi Level
**Aprendizaje Supervisado · Universidad Sergio Arboleda**
José Santiago González · Oscar Mateo Arrubla · Leonardo Diasgranados · Docente: Hernán Cruz Bueno

**Sector:** aprendizaje de japonés con contenido audiovisual (anime, doramas y cine japonés).
**Tarea:** clasificación. Predecir si el japonés de un título será **Fácil, Intermedio o Difícil** para un estudiante.

**La idea del taller:** en Voting, los 3 modelos votan y todos los votos valen lo mismo (una *democracia*). En **Stacking** agregamos un **meta-modelo**, un "juez", que **aprende a quién creerle** según el caso (una *meritocracia*). El juez no ve los datos del título: solo ve lo que dicen los 3 expertos.

```
Datos del título (28 variables)
        │
        ├──► KNN ─────────────┐
        ├──► SVM ─────────────┼──► probabilidades ──► META-MODELO (Regresión Logística) ──► Fácil / Intermedio / Difícil
        └──► Random Forest ───┘        (nivel 0)                     (nivel 1)
```
'''

FASE1 = r'''
## 1 · El dataset

| Aspecto | Detalle |
|---|---|
| Fuente | Moe-DB + AniList + TMDB: 9.977 títulos; quedan **9.034** tras quitar los que no tienen metadatos |
| Variables | **28**: palabras por episodio, palabras distintas y episodios (en logaritmo), año, anime/live-action, película/serie, género desconocido y 21 géneros |
| Objetivo | Dificultad (1-100) en terciles: **Fácil** < 19,9 · **Intermedio** 19,9-25,4 · **Difícil** > 25,4 |
| Tipo | **Clasificación multiclase** (3 clases balanceadas; el azar acierta 33%) |
| División | 80/20 estratificada, semilla 42: 7.227 de entrenamiento y 1.807 de prueba (la misma de todos los talleres) |

**¿Por qué clasificación y no regresión?** La app decide por niveles (semáforo verde, amarillo y rojo), y así podemos comparar directamente con los modelos y el Voting de los talleres anteriores.
'''

FASE2 = r'''
## 2 · Modelos base (nivel 0)

Como indica la clase: *"cada modelo base mantiene sus propios hiperparámetros tal como en los ejercicios individuales"*. Usamos los 3 modelos de nuestros talleres, de **familias distintas**, que es la "regla de oro" del stacking (no tres árboles iguales):

| Experto | Familia | Parámetros (de cada taller) | En qué se equivoca |
|---|---|---|---|
| **KNN** | Distancias | K=39, Manhattan, pesos por distancia | Cuando no hay títulos parecidos cerca |
| **SVM** | Márgenes | Kernel RBF, C=100, gamma='scale' | Cuando la frontera entre niveles es muy irregular |
| **Random Forest** | Árboles | 300 árboles, profundidad 15, 40% de variables | Con valores fuera de lo que vio al entrenar |
'''

FASE3 = r'''
## 3 · Stacking: el meta-modelo

- **Meta-modelo:** `LogisticRegression`. Un "jefe simple y con criterio", como recomienda la clase: los modelos base ya hicieron el trabajo pesado y un juez complejo se sobreajustaría.
- **cv=5 (out-of-fold):** el juez aprende con predicciones que cada experto hizo sobre títulos que **no vio** al entrenar. Si no, el juez creería en el modelo que más memorizó (lo comprobamos en la sección 7).
- **stack_method='predict_proba':** el juez recibe **probabilidades** (9 números: 3 por experto) y no solo la etiqueta, así sabe qué tan seguro está cada uno.

También entrenamos el **Voting Soft** del taller anterior para comparar.
'''

FASE4 = r'''
## 4 · ¿A quién le cree el juez?

Cada casilla es el peso que el meta-modelo le da a una probabilidad de un experto cuando decide un nivel. **Rojo** = esa opinión empuja hacia ese nivel; **azul** = lo aleja.

**Lo que aprendió el juez:**
- **Le cree mucho más a Random Forest:** para decidir *Fácil* le da peso **1,73** a lo que dice Random Forest, frente a 0,77 del SVM y 0,31 de KNN. Para *Difícil*, **1,66** frente a 0,67 y 0,51.
- **El SVM es el segundo experto** en todos los niveles.
- **KNN casi no cuenta para Intermedio (0,08):** el juez aprendió que KNN no sabe distinguir ese nivel y lo ignora ahí.
- **Usa también los "no":** si Random Forest da alta probabilidad de *Fácil*, eso resta mucho para *Difícil* (**−1,31**). El juez no solo suma votos, también descarta.

Esta es la diferencia con Voting: allí los 3 pesaban igual (1/3 cada uno); aquí los pesos se **aprendieron** según quién suele acertar.
'''

FASE5 = r'''
## 5 · Resultados: modelos base vs Voting vs Stacking

| Modelo | Accuracy | F1-macro | AUC | Recall Difícil | Errores graves |
|---|---|---|---|---|---|
| KNN | 0,546 | 0,544 | 0,737 | 0,602 | 7,9% |
| SVM | 0,570 | 0,568 | 0,746 | 0,662 | 6,5% |
| Random Forest | 0,581 | 0,580 | 0,770 | 0,650 | 6,1% |
| Voting Soft | 0,593 | 0,591 | 0,769 | 0,663 | 5,6% |
| **Stacking** | **0,600** | **0,600** | **0,779** | **0,672** | **5,0%** |

**El Stacking es el mejor en las 5 métricas.** Acierta **60 de cada 100** títulos (el azar, 33), detecta el **67%** de los títulos difíciles y comete el menor número de **errores graves**:
'''

FASE6 = r'''
## 6 · ¿Qué meta-modelo elegir?

La clase sugiere probar varios "jueces". Probamos 4 opciones:

**Resultado:** las 4 quedan prácticamente empatadas (F1 entre 0,597 y 0,601). Elegimos la **Regresión Logística** por el principio de parsimonia: con el mismo resultado, el juez más simple es más estable, se entrena más rápido y sus pesos se pueden **explicar** (sección 4). El Random Forest pequeño logra un poco menos de errores graves (4,6%), pero es una caja negra y su ventaja en F1 es de solo 0,002.

- **C=0,1** hace al juez más estricto (más regularización). No mejora: no hay sobreajuste que corregir.
- **passthrough=True** le pasa al juez también los datos originales, además de las predicciones. Tampoco mejora: los expertos ya extrajeron lo útil de las 28 variables.
'''

FASE7 = r'''
## 7 · "PERO CUIDADO": ¿por qué la validación cruzada es obligatoria?

Repetimos el stacking **sin** validación cruzada: el juez aprende con lo que cada experto predice sobre sus **propios datos de entrenamiento**.

**Lo que pasa:**
- **KNN acierta el 100% en entrenamiento**, porque con pesos por distancia cada título es su propio vecino más cercano y se lo sabe de memoria.
- El juez cae en la trampa: le da a KNN un peso de **3,85**, más del doble que a Random Forest (1,46), justo al revés de lo correcto.
- En títulos nuevos, KNN es el **peor** modelo. Resultado: F1 **0,569**, peor que Random Forest solo (0,580) y que el stacking bien hecho (**0,600**).

Es exactamente lo que advierte la clase: *"si no usas esto, el meta-modelo elegirá siempre al modelo base que más se sobreajustó"*.
'''

FASE8 = r'''
## 8 · ¿Qué pasa si quitamos un experto?
'''

HALLAZGOS = r'''
## Hallazgos

1. **El Stacking es nuestro mejor modelo hasta ahora:** F1 **0,600**, frente a 0,591 del Voting Soft y 0,580 de Random Forest solo. Mejora también el AUC (0,779) y baja los errores graves a **5,0%** (Random Forest: 6,1%).
2. **El juez aprendió a quién creerle:** confía sobre todo en Random Forest, luego en el SVM, y casi ignora a KNN para el nivel Intermedio. Es la "meritocracia" de la clase frente a la "democracia" del Voting.
3. **Los 3 expertos aportan**, aunque KNN sea el más débil: al quitarlo el F1 baja a 0,595; sin SVM, a 0,589; sin Random Forest, a **0,582**. La diversidad de familias es lo que da la ventaja.
4. **La validación cruzada no es opcional:** sin ella el juez confía en el modelo que memoriza (KNN) y el stacking queda **peor que un solo modelo** (F1 0,569).
5. **El meta-modelo simple basta:** cambiar el juez por otro más complejo no mejora el resultado.
6. **Donde más mejora es en los errores graves:** *Fácil → Difícil* baja de 48 (RF) a **42** y *Difícil → Fácil* de 62 a **48**. Para Yomi Level es lo más importante, porque es el error que hace que un principiante reciba algo que no puede entender.
'''

CONCLUSION = r'''
## Conclusión

**Elegimos el Stacking (KNN + SVM + Random Forest, con Regresión Logística como juez)** como el modelo más preciso de Yomi Level: es el mejor en todas las métricas y el que menos confunde un título fácil con uno difícil.

**Limitaciones:**
- **La mejora es real pero moderada** (+2 puntos de F1 sobre Random Forest). El techo lo pone la información disponible: con 28 variables de ficha y conteos de palabras, el nivel **Intermedio** sigue siendo difícil de separar (F1 0,472).
- **Es más pesado:** entrena 3 modelos más 5 rondas de validación cruzada para el juez, y el SVM con probabilidades es lo más lento. Conviene en un servidor; para la app que corre en el navegador seguiríamos usando Random Forest solo.
'''
