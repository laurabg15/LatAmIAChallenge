# LatAmAIChallenge: reconocimiento de señas LSC5 con landmarks

Informe técnico del pipeline de clasificación de cinco señas aisladas de la Lengua de Señas Colombiana (LSC5) para el reto **LatAm AI Challenge Chile × Colombia 2026**. Documenta los datos, las decisiones de diseño (con la evidencia que las respalda), los resultados y qué se tomó o no de la bibliografía revisada.

> **Reporte técnico entregable:** documento Word entregado por separado (`Informe_tecnico_LSC5.docx`). **Pesos del modelo final:** [`models/best_model.joblib`](models/best_model.joblib).

## Contenido

1. [Problema y criterios de evaluación](#1-problema-y-criterios-de-evaluación)
2. [Datos](#2-datos)
3. [Hallazgos de la exploración](#3-hallazgos-de-la-exploración)
4. [Validación: Leave-One-Signer-Out](#4-validación-leave-one-signer-out)
5. [Preprocesamiento](#5-preprocesamiento)
6. [Aumentación de datos](#6-aumentación-de-datos)
7. [Modelos](#7-modelos)
8. [Métricas, complejidad y selección del modelo final](#8-métricas-complejidad-y-selección-del-modelo-final)
9. [Experimentos y resultados](#9-experimentos-y-resultados)
10. [Bibliografía: qué se aplica y qué no](#10-bibliografía-qué-se-aplica-y-qué-no)
11. [Estructura del repositorio y ejecución](#11-estructura-del-repositorio-y-ejecución)
12. [Limitaciones y trabajo pendiente](#12-limitaciones-y-trabajo-pendiente)

---

## 1. Problema y criterios de evaluación

| | |
|---|---|
| Tarea | Clasificación supervisada multiclase (5 señas) a partir de landmarks de MediaPipe Holistic |
| Evaluación oficial | Un firmante **no visto** (D = M01) |
| Métricas | Accuracy top-1 y F1 macro |
| Desempate | 1) menos GFLOPs por muestra (batch 1, **incluido el preprocesamiento**); 2) menos parámetros |

| Código | Seña |
|---|---|
| `sign_01` | Café |
| `sign_02` | Aromática |
| `sign_03` | Hola |
| `sign_04` | Qué necesitas |
| `sign_05` | Con gusto |

Como el test es un firmante desconocido, toda decisión se evalúa con **validación por firmante** (sección 4), nunca con particiones aleatorias.

## 2. Datos

- **Entrenamiento:** 1125 archivos `.npy` en `data/train/landmarks/sign_XX/`, de 3 firmantes (M02, M03, M04) y 3 vistas de cámara (V01, V02, V03). Hay 75 muestras por seña y firmante.
- **Formato de cada `.npy`:** `(frames, 1662)`, con:

  | Bloque | Columnas | Contenido |
  |---|---|---|
  | Pose | 0–131 | 33 puntos × (x, y, z, visibilidad) |
  | Rostro | 132–1535 | 468 puntos × (x, y, z) |
  | Mano izquierda | 1536–1598 | 21 puntos × (x, y, z) |
  | Mano derecha | 1599–1661 | 21 puntos × (x, y, z) |

  Un punto en (0, 0, 0) significa que MediaPipe no lo detectó.
- **Nombre de archivo:** `S{seña}_M{firmante}_V{vista}_C{cámara}_Clip{n}.npy`, por ejemplo `S01_M02_V01_C01_Clip076.npy`.
- **Longitud:** entre 33 y 97 frames (mediana 55).
- **Test:** 375 muestras del firmante M01, con nombres anónimos.

## 3. Hallazgos de la exploración

Fuente: `notebooks/01_exploracion_dataset.ipynb` y `reports/figures/paso*.png`.

1. **Datos completos y balanceados:** no hay NaN, cada landmark tiene su video y las clases están perfectamente balanceadas.
2. **Las manos faltan sobre todo en el reposo.** Los frames sin mano se concentran en el primer ~25 % y el último ~25–30 % del video (`paso2_perfil_temporal.png`). Durante la seña (30–70 %) se detectan casi siempre. El tramo útil tiene una mediana de 39 frames, contra 55 del video completo.
3. **Hay un firmante zurdo.** M02 firma con la mano izquierda en el 100 % de sus videos; M03 y M04 son diestros.
4. **Qué distingue a las señas:** la altura de la mano respecto a los hombros y si participa la segunda mano. Los pares con más riesgo de confusión son `sign_01`/`sign_03` y `sign_04`/`sign_05`.
5. **Cada firmante tiene su ritmo:** M03 es el más rápido (~49 frames) y M04 el más lento (~62). La duración del video no debe ser una pista para el modelo.
6. **Caso anómalo:** en la vista **V03 de M04**, MediaPipe detecta las dos manos incluso en reposo (95 % de los frames). Eso afecta cualquier criterio basado en "hay mano detectada" (sección 5.2).

## 4. Validación: Leave-One-Signer-Out

- **Decisión:** el firmante se extrae del bloque `M` del nombre de archivo.
- **Por qué:** la primera versión usaba el bloque `S`, que es la **seña**. Eso generaba 5 "folds" donde la seña de validación nunca aparecía en el entrenamiento.
- **Cómo funciona:** 3 folds (validan M02, M03 y M04). Cada uno entrena con 750 muestras y valida con 375.
- **Comprobaciones:** un `assert` en `src/validation.py` verifica que ningún firmante quede en entrenamiento y validación a la vez. Las 5 señas están presentes en ambos lados de cada fold.
- **Sin fugas entre vistas:** al separar por firmante, las tres vistas de una misma ejecución quedan siempre del mismo lado. Las estadísticas de normalización de los modelos (BatchNorm, StandardScaler) se ajustan solo con el entrenamiento de cada fold.
- **Supuesto:** que `M` identifica al firmante. Los datos lo respaldan, porque es el único bloque que se repite en las 5 señas.

## 5. Preprocesamiento

Implementado en `src/dataset.py`. Todos los pasos son **deterministas y usan solo la propia muestra**, así que se aplican igual al test, sin fuga de información.

### 5.1 Selección y normalización (`load_and_normalize`)

| Paso | Decisión | Motivo |
|---|---|---|
| Selección | 55 puntos: 13 de la pose (nariz, hombros, codos, muñecas, dedos de la pose) + 21 de cada mano. Se descarta el rostro. | La seña está en manos y brazos; el rostro es el 86 % de los datos y casi no aporta. |
| Coordenadas | Solo x, y, más una máscara de faltantes | MediaPipe estima `z` de forma poco fiable. Un 0 no es una coordenada real. |
| Tipo de dato | `float32` | La mitad de memoria y el tipo nativo de PyTorch. Los `.npy` originales (`float64`) no se modifican. |
| Normalización | Centro en el punto medio de los hombros, escala por el ancho de hombros | Elimina el tamaño de la persona y la distancia a la cámara. |

### 5.2 Recorte del reposo

El reposo es casi un tercio de cada video y es igual en todas las señas, así que se recorta.

- **Versión 1 (descartada): primera y última mano detectada**, sostenida 3 frames, con 2 frames de margen. **Falla en `S04_M04_V03`:** como ahí las manos se detectan en reposo, el recorte no corta nada (mediana de 61 frames, contra 43 en V01).
- **Versión 2 (actual): subida de la muñeca, `REST_RISE = 0.3`.** Un frame es activo si alguna **muñeca de la pose** (puntos 15 y 16, detectados en el 100 % de los frames) subió más de 0,3 anchos de hombro respecto a su propio reposo. El reposo es el percentil 90 de su altura en ese video. Se recorta entre el primer y el último tramo de 3 frames activos seguidos, con 2 frames de margen.

**Por qué no un umbral de altura fijo:** la mano en reposo detectada está entre 1,3 y 2,0 anchos de hombro bajo los hombros, y `sign_04`/`sign_05` llegan a 1,7–1,9 durante la seña. Un umbral fijo cortaría las señas bajas. Medir la **subida relativa** se adapta a cada video.

**Elección del umbral** (mediana de frames tras el recorte en `S04_M04`, y porcentaje de videos de `sign_05` que pierden frames de la seña):

| Criterio | V01 | V02 | V03 | `sign_05` con parte de la seña cortada |
|---|---|---|---|---|
| Manos detectadas | 43 | 45 | **61** | — |
| Subida > 0,3 | 45 | 50 | **51** | 2,2 % |
| Subida > 0,5 | 43 | 47 | 47 | 6,7 % |
| Subida > 0,8 | 39 | 43 | 42 | 7,1 % |

Se eligió 0,3 porque iguala las tres vistas y casi no corta señas. Tras el recorte, los videos quedan entre 21 y 91 frames (mediana 42).

**Limitación:** el criterio supone que el video empieza y termina en reposo. Todos los videos de entrenamiento siguen ese ciclo.

### 5.3 Representación para los modelos (`to_features`)

1. **Forma de cada mano:** dedos relativos a la muñeca de la mano. Separa la forma de la mano de su posición.
2. **190 características por frame:** 110 posiciones (55 puntos × 2) + 80 de forma de mano (2 manos × 20 × 2). Los faltantes se anulan con la máscara.
3. **Dos salidas, según el tipo de modelo:**

| Salida | Usada por | Construcción | Tamaño |
|---|---|---|---|
| `vector` | Dummy, Linear, Random Forest | Media en **4 tramos iguales** del recorte + desviación estándar global + presencia de cada mano por tramo (ignorando faltantes) | 958 |
| `sequence` | CNN 1D, GRU, GRU2, LSTM, TCN | Remuestreo a **`SEQ_LEN = 20`** frames (vecino más cercano) + presencia de cada mano | 20 × 192 |

**Por qué 4 tramos** (experimento con Linear y una semilla):

| Tramos | Características | M02 | M03 | M04 | Media |
|---|---|---|---|---|---|
| 2 | 574 | 1,000 | 0,939 | 0,973 | 0,971 |
| **4** | **958** | 1,000 | 0,976 | 0,941 | **0,972** |
| 6 | 1342 | 0,997 | 0,979 | 0,899 | 0,958 |
| 8 | 1726 | 1,000 | 0,936 | 0,880 | 0,939 |

Más tramos empeoran el resultado, sobre todo en M04: con pocos frames por tramo, el vector memoriza el ritmo de cada firmante. Por la misma razón **se descartaron los cortes por perfil de manos** (10/26/36/64/74/88 % del video): dejaban un solo tramo sobre la seña, dependían del ritmo de cada firmante y perdían sentido tras el recorte. Quedan documentados como constante comentada en `src/dataset.py`.

**Por qué remuestrear la secuencia:** los modelos necesitan un largo fijo. Los videos cortos repiten frames y los largos saltan algunos; siempre se conservan el primer y el último frame. Además, el modelo no puede usar la duración como atajo. `SEQ_LEN` pasó de 40 a 20 siguiendo a Mindlin (2021) (sección 10), lo que reduce a la mitad los GFLOPs de los modelos temporales.

## 6. Aumentación de datos

Solo se aplica en entrenamiento, siempre activa, cada vez que se lee una muestra (`augment` en `src/dataset.py`). Validación y test nunca se aumentan, y la aumentación no cuenta en los GFLOPs.

| Técnica | Parámetro | Decisión y motivo |
|---|---|---|
| **Reflejo zurdo ↔ diestro** | p = 0,5 | `x → -x`, e intercambio de las manos y de los pares izquierda/derecha de la pose. Se eligió **como aumentación** (el modelo aprende la seña con las dos manos) y no como conversión de todo a diestro, porque no sabemos si M01 es zurdo. |
| Rotación | ±10° alrededor de los hombros | Imita la inclinación de la cámara o del cuerpo. Es pequeña porque la altura de la mano distingue señas. |
| Jitter | σ = 0,02 anchos de hombro | Ruido de MediaPipe; solo sobre puntos detectados. |
| Eliminación de frames | 10 % | Robustez a videos más cortos. |
| Eliminación de puntos | 5 %, y a veces una mano en el 30 % de los frames (p = 0,2) | Imita fallos de detección; no sabemos cómo detecta MediaPipe a M01. |
| **Muestreo temporal aleatorio** | ±½ intervalo por posición | Solo para modelos de secuencia: varía qué frames se toman al remuestrear, sin cambiar la duración (Mindlin, 2021). |
| ~~Escalado~~ | — | **Descartado:** la normalización por hombros ya lo anula. |
| ~~Cambio de velocidad~~ | — | **Descartado:** el remuestreo a largo fijo lo anula. |

Para **Random Forest** (sin épocas), el entrenamiento se recorre 3 veces con aumentación (750 → 2250 muestras).

## 7. Modelos

Definidos en `src/models.py`. Todos los de PyTorch normalizan la entrada con `BatchNorm1d`, ajustado solo con el entrenamiento de cada fold.

| Modelo | Entrada | Arquitectura | Para qué está |
|---|---|---|---|
| `dummy` | vector | Siempre predice la clase más frecuente (sklearn) | **Piso de referencia** (accuracy 0,20) |
| `linear` | vector | BatchNorm → Linear(958 → 5). Equivale a una regresión logística. | Modelo simple, barato y fuerte |
| `random_forest` | vector | 200 árboles, profundidad 12 (sklearn) | Comparación no lineal |
| `cnn1d` | secuencia | 2 × Conv1d(64, k = 5) + BN + ReLU, MaxPool, AvgPool, Dropout → Linear | Trayectoria temporal |
| `gru` | secuencia | GRU de 1 capa (64) → último estado → Linear | Recurrente básica |
| `gru2` | secuencia | GRU 64 → GRU 128 → Dense 128 + ELU + Dropout → Linear | Arquitectura de Mindlin (2021) adaptada a landmarks |
| `lstm` | secuencia | LSTM de 1 capa (64) → último estado → Linear | Recurrente con memoria |
| `tcn` | secuencia | Conv 1×1 → 3 bloques residuales Conv1d(k = 3, dilatación 1/2/4) → AvgPool → Linear | CNN temporal más profunda (equivalente a Conv3D sobre landmarks) |

**Descartados:**
- **Transformer:** demasiados parámetros para 1125 muestras.
- **VGG16, MobileNet y Conv3D sobre video:** trabajan con píxeles; MediaPipe ya hace ese trabajo de extraer características. Ver sección 10.
- **Regresión logística de sklearn:** es redundante con `linear`.

**Entrenamiento:** Adam (lr 1e-3, weight decay 1e-4), 50 épocas, batch 32, entropía cruzada.

## 8. Métricas, complejidad y selección del modelo final

- **Métricas:** accuracy y F1 macro (`zero_division = 0`) sobre el firmante de validación, promediadas sobre **3 folds × 3 semillas (42, 43, 44)**.
- **GFLOPs (batch 1):**
  - Modelos de PyTorch: `torch.utils.flop_counter.FlopCounterMode`. La LSTM se cuenta a mano (`4 × 2 × (entrada + oculta) × oculta × frames`), porque el contador le asigna 0.
  - Random Forest: nodos recorridos más votos.
  - En todos los casos se suma una estimación del preprocesamiento para 55 frames, la mediana del video crudo.
- **Parámetros:** pesos entrenables en PyTorch; para Random Forest, 2 por nodo interno y 1 por clase en cada hoja.
- **Modelo final:** el de mayor accuracy, luego mayor F1 y luego menos GFLOPs. Se reentrena con los 3 firmantes y se guarda en `models/best_model.joblib`.

## 9. Experimentos y resultados

### 9.1 Corrida A: primera versión del pipeline (1 semilla, recorte por manos, `SEQ_LEN = 40`)

| Modelo | Accuracy | F1 macro | GFLOPs | Parámetros |
|---|---|---|---|---|
| linear | 0,972 | 0,972 | 0,000104 | 6.711 |
| random_forest | 0,947 | 0,946 | 0,000097 | 64.959 |
| cnn1d | 0,957 | 0,956 | 0,005829 | 83.013 |

**El reflejo resolvió el problema del zurdo:** el fold de M02 pasó de 0,49 (referencia de otro equipo, regresión logística sin reflejo) a **1,000** en los tres modelos.

### 9.2 Corrida B: 7 modelos, 3 semillas, excluyendo `S04_M04_V03` del entrenamiento

| Modelo | Accuracy | F1 macro | GFLOPs | Parámetros |
|---|---|---|---|---|
| dummy | 0,200 | 0,067 | 0,000094 | 0 |
| **linear** | **0,971** | **0,971** | 0,000104 | 6.711 |
| random_forest | 0,943 | 0,943 | 0,000097 | 67.479 |
| cnn1d | 0,943 | 0,942 | 0,005829 | 83.013 |
| gru | 0,939 | 0,939 | 0,004027 | 50.245 |
| lstm | 0,947 | 0,946 | 0,005338 | 66.757 |
| tcn | 0,947 | 0,947 | 0,004027 | 50.501 |

Conclusiones:
- **Linear es el mejor, el más estable entre semillas y de los más baratos.** Con 750 muestras, los modelos de secuencia no aprovechan la trayectoria.
- **Excluir `S04_M04_V03` no cambió el resultado** (Linear con semilla 42: M02 1,000 → 0,997; M03 0,976 → 0,976). Se revirtió (`EXCLUDE_FROM_TRAIN = []`) y el problema se atacó con el recorte de la sección 5.2.
- **El punto débil era M04** (0,932): `sign_05` se confundía con `sign_02` y `sign_04` con `sign_02`. Por vista, V01 tenía 0 errores, V02 tenía 12 y V03 tenía 10. Se resolvió en la corrida C.

### 9.3 Corrida C: recorte por subida de muñeca, `SEQ_LEN = 20`, muestreo temporal aleatorio y `gru2`

Media de 3 folds × 3 semillas; sin videos excluidos.

| Modelo | Accuracy | F1 macro | GFLOPs | Parámetros | Accuracy en B | Cambio |
|---|---|---|---|---|---|---|
| dummy | 0,200 | 0,067 | 0,000094 | 0 | 0,200 | = |
| **linear** | **0,991** | **0,991** | 0,000104 | 6.711 | 0,971 | **+2,0** |
| random_forest | 0,960 | 0,960 | 0,000097 | 73.730 | 0,943 | +1,7 |
| cnn1d | 0,970 | 0,970 | 0,002962 | 83.013 | 0,943 | +2,7 |
| gru | 0,934 | 0,934 | 0,002061 | 50.245 | 0,939 | −0,5 |
| gru2 | 0,957 | 0,956 | 0,005044 | 141.573 | — | nuevo |
| lstm | 0,937 | 0,936 | 0,002716 | 66.757 | 0,947 | −1,0 |
| tcn | 0,951 | 0,951 | 0,002061 | 50.501 | 0,947 | +0,4 |

Accuracy por firmante de validación:

| Fold | linear | random_forest | cnn1d | gru | gru2 | lstm | tcn |
|---|---|---|---|---|---|---|---|
| M02 | 0,999 | 0,970 | 1,000 | 0,998 | 1,000 | 0,992 | 1,000 |
| M03 | 0,985 | 0,976 | 0,977 | 0,917 | 0,965 | 0,921 | 0,925 |
| M04 | **0,989** | 0,934 | 0,933 | 0,887 | 0,905 | 0,899 | 0,929 |

Conclusiones:

- **El recorte por subida de la muñeca resolvió el punto débil de M04.** En los modelos de vector, el preprocesamiento es lo único que cambió frente a la corrida B (`SEQ_LEN` y el muestreo temporal no los afectan). Linear sube en M04 de 0,932 a **0,989**, y la confusión de `sign_05` con `sign_02` en M04 baja de 10 casos a 1. El reposo detectado en V03 era el que confundía al modelo.
- **Linear sigue siendo el modelo final:** 0,991 de accuracy, estable entre semillas (0,973–1,000 por fold) y de los más baratos (0,0001 GFLOPs, 6.711 parámetros).
- **`SEQ_LEN = 20` redujo a la mitad los GFLOPs de los modelos de secuencia** sin perder rendimiento: `cnn1d` sube a 0,970 y `tcn` a 0,951. En estos modelos los efectos están mezclados (recorte, largo de secuencia y muestreo aleatorio), así que no se puede atribuir la mejora a uno solo.
- **`gru2` mejora a `gru`** (0,957 contra 0,934), pero cuesta 2,4 veces más GFLOPs y 2,8 veces más parámetros, y queda lejos de Linear. Con 750 muestras, la recurrencia no aporta frente al resumen por tramos.
- **Los modelos de secuencia son inestables entre semillas:** por ejemplo, `cnn1d` en M04 varía entre 0,888 y 0,981.
- **Error restante más relevante (Linear):** en M03, `sign_05` (Con gusto) se predice como `sign_01` (Café) en 8 de 75 casos.

## 10. Bibliografía: qué se aplica y qué no

**[1] Muñoz-Galindez, J. A. y Vargas-Cañas, R. (2023).** _Modelo de interpretación de lengua de señas colombiano usando inteligencia artificial._ Revista de Investigación, Desarrollo e Innovación, 13(2), 357–366. https://doi.org/10.19053/20278306.v13.n2.2023.16840

- **Qué hacen:** 70 firmantes, 10 palabras de LSC, 6 imágenes RGB por seña. Comparan **VGG16** estático (66,5 %) contra **VGG16 congelada + LSTM 128** (76 %, F1 77,3), con ~27 M de parámetros. Separan por persona (50/10/10). Usan espejo como aumentación en el modelo estático, L2 y early stopping.

**[2] Mindlin, I. (2021).** _Reconocimiento de Lengua de Señas con Redes Neuronales Recurrentes._ Tesina de Licenciatura, Facultad de Informática, Universidad Nacional de La Plata.

- **Qué hacen:** LSA64 (64 señas, 10 personas, guantes de color), sobre video RGB. Comparan **ConvLSTM**, **MobileNet + 2 GRU (64, 128) + Dense 128** y **Conv3D**; el mejor es Conv3D con manos segmentadas (99,4 %). Encuentran que el muestreo temporal aleatorio mejora todos los modelos, que 16 frames rinden más que 32, que Conv3D falla si cambia la longitud y que el cropping espacial empeora. **Su validación no separa por persona**, así que sus cifras son optimistas.

| Idea | Fuente | Decisión | Motivo |
|---|---|---|---|
| Espejo / reflejo por mano dominante | [1], [2] | ✅ **Aplicado** (como aumentación, p = 0,5) | Resuelve el firmante zurdo (M02: 0,49 → 1,00) |
| Validación por firmante | [1] | ✅ **Aplicado** (LOSO) | Es el escenario del test |
| Muestreo temporal aleatorio | [2] | ✅ **Aplicado** (corrida C) | Mejoró todos los modelos en la tesis; no altera la duración |
| Secuencias más cortas (16 en lugar de 32) | [2] | ✅ **Aplicado:** `SEQ_LEN` 40 → 20 | Mejor o igual resultado, con la mitad de GFLOPs |
| GRU de 2 capas + Dense 128 | [2] | ✅ **Aplicado** como `gru2`, conservando `gru` para comparar | Mejor recurrente de la tesis; se mide si aporta sobre landmarks |
| Aislar manos del fondo | [2] | ✅ **Equivalente ya aplicado** | Usar landmarks de manos y brazos sin rostro cumple el mismo papel |
| Regularización L2 | [1], [2] | ✅ **Aplicado** (weight decay 1e-4 + dropout) | — |
| Early stopping con validación interna | [1] | ⏸️ **Pendiente** | Requiere separar una validación dentro de cada fold; agrega complejidad |
| VGG16 / MobileNet preentrenadas | [1], [2] | ❌ **No aplicado** | Trabajan sobre píxeles. MediaPipe ya extrae las características y VGG16 cuesta ~15 GFLOPs por imagen, contra ~0,0001 de nuestro Linear: perderíamos el desempate. |
| Conv3D / ConvLSTM sobre video | [2] | ❌ **No aplicado** (su equivalente sobre landmarks es `cnn1d` / `tcn`) | Los landmarks no forman una imagen; además, Conv3D es frágil ante cambios de longitud. |
| Cropping espacial aleatorio | [2] | ❌ **No aplicado** | Empeoró en la tesis; la normalización por hombros ya elimina la posición. |

## 11. Estructura del repositorio y ejecución

```
data/train/landmarks/sign_0X/*.npy   datos (no se suben al repositorio)
notebooks/01_exploracion_dataset.ipynb
reports/figures/                     figuras de exploración y matrices de confusión
reports/resultados_*.csv             resultados LOSO por fold y resumen
src/dataset.py                       preprocesamiento, aumentación y Dataset
src/models.py                        modelos y build_model
src/validation.py                    LOSO por firmante
src/evaluate.py                      métricas, GFLOPs, parámetros y matrices de confusión
train.py                             entrenamiento LOSO, tabla final y modelo final
generate_submission.py               predicción del test y validación del submission
```

```powershell
# Entorno (una vez; Python 3.12, versiones fijadas en requirements.txt)
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # opcional: PyTorch para CPU, más liviano
.venv\Scripts\python -m pip install -r requirements.txt

# Entrenamiento LOSO de todos los modelos (~20–30 min en CPU)
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python train.py

# Submission (cuando exista data/test/landmarks y data/test/submission_template.csv)
.venv\Scripts\python generate_submission.py
```

**Salidas:**
- `reports/resultados_por_fold.csv` y `reports/resultados_resumen.csv`.
- `reports/figures/confusion_<modelo>.png`, con la semilla 42.
- `models/best_model.joblib`.
- `submission.csv`, con columnas `video_id,predicted_sign` y valores `sign_01`…`sign_05`, en el orden del template. Se valida contra el template y la predicción se hace muestra por muestra (batch 1), sin aumentación.

**Configuración:** constantes al inicio de `src/dataset.py` (preprocesamiento y aumentación) y de `train.py` (modelos, épocas, semillas, exclusiones).

## 12. Limitaciones y trabajo pendiente

- **El firmante de test (M01) es desconocido:** la mano dominante, el ritmo y la cámara pueden diferir. La media LOSO es la estimación honesta.
- **Supuesto de firmante:** `M` identifica al firmante. Falta confirmarlo con la documentación del dataset.
- **Diferencias pequeñas entre modelos** (1 a 3 puntos): se usan 3 semillas, pero sigue habiendo ruido.
- **GFLOPs del preprocesamiento:** son una estimación analítica, no un conteo exacto.
- **Submission:** la lógica está lista, pero no se ha probado de punta a punta porque aún no están `data/test` ni el template.
- **Próximos pasos:** early stopping, revisar la confusión de `sign_05` con `sign_01` en M03 (corrida C), separar el efecto de `SEQ_LEN` y del muestreo temporal en los modelos de secuencia, y medir el tiempo de inferencia.
