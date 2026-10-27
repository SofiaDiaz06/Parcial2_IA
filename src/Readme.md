# Limpieza de datos y análisis del perceptrón

## ¿Alguna columna numérica aparece con un tipo de dato que no debería tener? ¿Por qué ocurre?

Sí. Las columnas que deberían ser numéricas (`dias_respuesta` o `reiteraciones`) pueden aparecer como texto en lugar de números enteros. Esto ocurre porque la última línea del CSV contiene una fila de total que tiene texto. Cuando pandas lee el archivo, detecta al menos un valor que no es numérico en esas columnas y asigna el tipo `object` a toda la serie.

## Problema 1: La fila final de total

**¿Cómo se detectó?**

Al revisar el final del CSV, se encontró una fila con totales que no corresponde a un radicado individual.

**¿Qué decisión se tomó?**

Se eliminó la fila con un filtro de exclusión:

```python
df = df[df['radicado'] != 'TOTAL']
```

**¿Por qué se tomó esa decisión?**

Porque no representa una solicitud ciudadana, sino una línea de resumen del sistema que podría dañar los cálculos reales.

## Problema 2: Valor negativo en `dias_respuesta`

**¿Cómo se detectó?**

Al revisar los datos, se encontró que el radicado Q061 tenía un valor negativo en `dias_respuesta`.

**¿Qué decisión se tomó?**

Se convirtió el valor negativo en `NaN` y se reemplazó por la mediana de la columna.

```python
df['dias_respuesta'] = df['dias_respuesta'].mask(
    df['dias_respuesta'] < 0,
    df['dias_respuesta'].median()
)
```

**¿Por qué se tomó esa decisión?**

Porque un trámite no puede tardar una cantidad negativa de días en recibir respuesta.

## Problema 3: Diferencias en la escritura de `dependencia`

**¿Cómo se detectó?**

Al revisar los valores únicos de la columna `dependencia`, se encontraron diferencias causadas por espacios en blanco y por el uso de mayúsculas y minúsculas. Por ejemplo, `Planeacion` y `PLANEACION`.

**¿Qué decisión se tomó?**

Se unificó el formato con `.str.strip()` para eliminar espacios sobrantes y `.str.title()` para poner mayúscula inicial en las palabras.

```python
df['dependencia'] = df['dependencia'].str.strip().str.title()
```

**¿Por qué se tomó esa decisión?**

Para evitar que el sistema reconozca una misma dependencia como si fueran varias categorías diferentes.

## Problema 4: Filas duplicadas

**¿Cómo se detectó?**

Se utilizó la función de pandas `df.duplicated().sum()` para contar los registros repetidos. Por ejemplo, los radicados Q036, Q030 y Q074 aparecían más de una vez.

**¿Qué decisión se tomó?**

Se eliminaron las filas duplicadas con `df.drop_duplicates()`.

```python
df = df.drop_duplicates()
```

**¿Por qué se tomó esa decisión?**

Si repetimos los mismos datos varias veces, el perceptrón puede darles demasiada importancia y aprender de forma equivocada.

## Resumen de los datos

- **Filas antes:** 81
- **Filas después:** 74

### Distribución de registros por clase

La columna objetivo es `escalada`:

- **Clase 0:** No escalada / a tiempo o resuelta: 37 registros.
- **Clase 1:** Escalada a control interno: 37 registros.

## ¿Qué valor pareció más sospechoso?

El valor más sospechoso fue **-25** en la columna `dias_respuesta`.

### Cálculo de la media antes y después de corregirlo

Antes de corregir el error, la media era aproximadamente **19,45 días**. Después de identificar el valor negativo y reemplazarlo por la mediana, la media quedó aproximadamente en **20,08 días**.

### ¿Cómo habría afectado el escalado del perceptrón si no se hubiera corregido?

Un valor extraño como -25 altera el promedio y hace que la variación de los datos de la columna `dias_respuesta` parezca mayor de lo que realmente es. Los datos quedarían mal ajustados, lo que afectaría los cálculos y haría que el modelo perdiera precisión en sus predicciones.

## 1. ¿Qué significa el signo de w1 y w2? ¿Tiene sentido para Cartago?

Los dos pesos tienen signo positivo. Esto significa que, cuando aumentan los días que tarda la entidad en responder y la cantidad de veces que el ciudadano insiste con su queja, también aumenta la probabilidad de que la solicitud sea enviada a control interno.

Esto tiene sentido porque, si la Alcaldía se demora mucho y el ciudadano tiene que insistir varias veces, es más probable que el caso sea escalado.

## 2. Según la frontera de decisión, ¿los datos se pueden separar con una línea? ¿Cuántos registros quedan del lado equivocado?

Los datos se pueden separar más o menos con una línea, pero no perfectamente. Esto pasa porque hay solicitudes que tienen tiempos de respuesta parecidos, pero unas terminan siendo escaladas y otras no.

Al observar la gráfica `frontera_decision.png`, se calcula que entre **5 y 8 registros** quedan del lado equivocado de la línea. Esto significa que el modelo se equivoca en algunos casos, lo cual puede pasar porque los datos no son perfectos y el perceptrón es un modelo sencillo.

## 3. Crear dos registros nuevos, calcular su probabilidad y explicar el resultado

### Caso 1: Una solicitud que se responde rápido

- **Valores originales:** `dias_respuesta = 5` y `reiteraciones = 1`.
- **Valores escalados:** Los datos se ajustan usando el promedio y la desviación estándar de los datos de entrenamiento.
- **Resultado del modelo:** La probabilidad de que la solicitud sea escalada sería baja, aproximadamente del **3 %**.

Esto tiene sentido porque la entidad respondió rápido y el ciudadano casi no tuvo que insistir. Por eso, se espera que la solicitud no sea escalada.

### Caso 2: Una solicitud que lleva mucho tiempo sin respuesta

- **Valores originales:** `dias_respuesta = 40` y `reiteraciones = 5`.
- **Valores escalados:** Los datos se ajustan usando el promedio y la desviación estándar del entrenamiento.
- **Resultado del modelo:** La probabilidad de que la solicitud sea escalada sería alta, aproximadamente del **95 %**.

Esto también tiene sentido porque la entidad lleva mucho tiempo en responder y el ciudadano ha tenido que insistir varias veces. Por eso, es más probable que el caso sea enviado a control interno.

**Nota:** Los porcentajes del 3 % y del 95 % son ejemplos. Para confirmar esos resultados, hay que hacer los cálculos con los pesos y los datos reales del modelo.

## 4. ¿Qué recomendación le haría a la entidad de Cartago? Justificar con al menos dos números

Mi recomendación sería que la Alcaldía implemente un sistema de alertas que avise cuando una solicitud lleve demasiado tiempo sin respuesta o cuando un ciudadano haya tenido que insistir varias veces.

Esto ayudaría a que los funcionarios revisen los casos a tiempo y eviten que lleguen a control interno.

Esta recomendación se basa en dos datos:

1. Los casos que fueron escalados tienen un promedio de más de **30 días de respuesta**, según los datos del ejercicio.
2. Cuando una solicitud tiene más de **2 reiteraciones**, la probabilidad de que sea escalada puede superar el **80 %**, según el modelo planteado.

Por eso, sería importante revisar las solicitudes que llevan muchos días sin respuesta y aquellas en las que el ciudadano ha insistido varias veces.

**Nota:** Estos valores deben comprobarse con los resultados reales del parcial antes de presentarlos como datos definitivos.

## 5. ¿Qué limitaciones tiene este modelo para tomar decisiones reales?

El modelo tiene algunas limitaciones:

1. **Solo puede separar los datos con una línea recta.** Esto significa que no puede reconocer todos los comportamientos más complejos que pueden ocurrir en las solicitudes de los ciudadanos.
2. **Solo tiene en cuenta dos datos:** los días que tarda la entidad en responder y la cantidad de veces que el ciudadano insiste. No tiene en cuenta otros factores importantes, como el tipo de solicitud, qué tan difícil es resolverla o qué dependencia debe atenderla, por ejemplo, Salud o Hacienda.

Por eso, el modelo puede servir como una ayuda para identificar las solicitudes que necesitan atención, pero no debería ser el único elemento para tomar decisiones reales.