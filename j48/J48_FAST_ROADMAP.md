# J48 Fast Roadmap

La línea `j48.fast` nace separada del baseline `strict`.

Principios:
- `strict` sigue siendo la referencia de fidelidad contra WEKA J48.
- `fast` puede usar representaciones internas distintas, pero debe validar
  equivalencia práctica antes de reemplazar nada.
- toda optimización nueva se mide contra `strict` y contra WEKA.

Estado actual:
- backend inicial: `numpy_fast`
- intervención implementada: codificación densa de atributos nominales a
  `float64`, preservando faltantes como `NaN`
- API pública disponible:
  - `J48Classifier(backend="numpy_fast", fidelity="equivalent")`
  - `J48FastClassifier(...)`

Orden recomendado:
1. medir `strict` vs `fast` en `Adult`, `UNSW-NB15`, `CIC-IDS2017`
2. perfilar `fit` y `predict`
3. optimizar candidate generation numérica
4. optimizar nominales de alta aridad
5. aplanar inferencia
6. evaluar `numba` o `cython` sobre los hot paths confirmados

Gate mínimo por cambio:
- tests unitarios
- `Layer A` sintético final
- `Layer B base`
- `Layer C base`

Backlog operativo:

- [J48_FAST_OPTIMIZATION_BACKLOG.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_OPTIMIZATION_BACKLOG.md)
