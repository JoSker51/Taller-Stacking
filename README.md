# C3 Taller 3 · Stacking — Yomi Level

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoSker51/Taller-Stacking/blob/main/Taller_Stacking_Yomi_Level.ipynb)

**Aprendizaje Supervisado · Universidad Sergio Arboleda**
José Santiago González · Oscar Mateo Arrubla · Leonardo Diasgranados · Docente: Hernán Cruz Bueno

**Tarea:** clasificar el japonés de cada título (anime, doramas, cine) en *Fácil*, *Intermedio* o *Difícil*. Aplicamos **Stacking**: KNN, SVM y Random Forest (nivel 0) + una Regresión Logística que aprende a combinarlos (nivel 1).

## Entregables

| Archivo | Contenido |
|---|---|
| [`Taller_Stacking_Yomi_Level.ipynb`](Taller_Stacking_Yomi_Level.ipynb) | Notebook ejecutado y comentado |
| [`Informe_Taller_Stacking_Yomi_Level.pdf`](Informe_Taller_Stacking_Yomi_Level.pdf) | Informe de resultados |
| [`presentacion_stacking.html`](presentacion_stacking.html) | Presentación (descargar y abrir en el navegador; flechas para avanzar) |
| [`figuras/`](figuras/) | Gráficas del notebook |

## Resultados (1.807 títulos de prueba)

| Modelo | Accuracy | F1-macro | AUC | Errores graves |
|---|---|---|---|---|
| KNN | 0,546 | 0,544 | 0,737 | 7,9% |
| SVM | 0,570 | 0,568 | 0,746 | 6,5% |
| Random Forest | 0,581 | 0,580 | 0,770 | 6,1% |
| Voting Soft | 0,593 | 0,591 | 0,769 | 5,6% |
| **Stacking** | **0,600** | **0,600** | **0,779** | **5,0%** |

Sin validación cruzada el stacking cae a F1 0,569: el meta-modelo le cree a KNN, que memoriza el entrenamiento (100%).

## Cómo ejecutar

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Taller_Stacking_Yomi_Level.ipynb
```

En Colab, el notebook descarga el dataset solo si no está en la carpeta.
