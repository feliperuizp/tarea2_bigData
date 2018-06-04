### Contributors:
- Felipe Rojos @farojos
- Felipe Ruiz @feliperuizp

# **Informe Tarea 2**

## 1. **Análisis de sentimientos en Críticas de películas**

### Actividad 1:

### Actividad 2:

### Actividad 3:

## 2. **Análisis de sentimientos en Twitter**

### Actividad 4:

Primero se procesan los vectores de spanish-word-embeddings del profesor Jorge Pérez de la Universidad de Chile. Se descargan localmente 10.000.000 de palabra para Word2Vector, FastText y Glove.

Los tiempos para descargar los vectores son:
- Word2Vec: 285 segundos
- FastText: 259 segundos
- Glove: 251 segundos

Luego para procesar el dataset de Tweets, se utiliza la librería nltk:
```
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
```

Luego se carga el Stemmer en español, las Stopwords en español para limpiar el contenido de cada tweet y el Tokenizer.

El dataset tiene la variable "polarities" y "polarities type", solo se utilizan las "polarities" ya que en el dataset de test solo se encuentra esa variable.

Polarities puede tomar 6 valores, lo que se representa con un número del 0 al 5.
```
polarities=['NONE', 'NEU', 'P', 'N+', 'P+', 'N']
````

 Se procesan los tweets para obtener una lista tweets, cada tweet queda representado como: 
 ```
 [[Lista de palabras no vectorizadas del contenido del tweet],polaridad]
```
El tiempo de procesamiento para esto es de:
- 6 segundos para el dataset de entrenamiento (7219 tweets). Hay tweets que no tienen contenido en el dataset, por lo que son eliminados.
- 50 segundos para el dataset de test (60789 tweets)

### Actividad 5:

 Luego para cada tweet se vectorizan sus palabras y se calcula el promedio de los vectores del contenido, para obtener los tweets de la siguiente forma:

 ```
 [vector promedio,polaridad]
```

Se realiza esta tarea para los tres métodos Word2Vector, FastText y Glove.

El tiempo de procesamiento para vectorizar los datos es de:
- 1.7 segundos para el dataset de entrenamiento
- 12.8 segundos para el dataset de test

Finalmente se entrena el modelo para los tres métodos con svm.

Los tiempos de entrenamiento de los modelos son de:
- 22.04 segundos para Word2Vector
- 20.58 segundos para FastText
- 22.99 segundos para Glove

Los resultados son:
- Training Set Accuracy Word2Vector: 0.371364653244
- Test Set Accuracy Word2Vector: 0.470426455482
- Training Set Accuracy FastText: 0.479166666667
- Test Set Accuracy FastText: 0.451224361311
- Training Set Accuracy Glove: 0.470078299776
- Test Set Accuracy Glove: 0.425875608867

```
Confussion matrix W2V:
 [[ 9459     0   229    15  4966  6450]
 [  118     0    11     1   379   796]
 [  372     0    36     0   631   446]
 [  673     0    50    21   784  3026]
 [ 4536     0   155     5 11910  4023]
 [ 2223     0   107    11  1957  6968]]

Confussion matrix FastText:
 [[ 9326   657  2272  1063  3709  4092]
 [  100    94   177   142   260   532]
 [  246    44   643    37   296   219]
 [  482   130   327  1684   438  1493]
 [ 4133   540  2888   566 10187  2315]
 [ 1960   490  1135  1179  1201  5301]]

Confussion matrix Glove:
 [[8919  823 2661 1367 3341 4008]
 [ 110  136  183  162  241  473]
 [ 220   69  653   53  273  217]
 [ 538  169  431 1537  482 1397]
 [4146  757 3032  899 9398 2397]
 [1925  559 1201 1263 1256 5062]]
```

### Actividad 6:

Rendimiento?
Comparar matrices de confusión?

## 3. **Entrenar un Modelo de Clasificación de Textos**

### Actividad 7:

Indique el ´area tem´atica de los textos y tama˜no del set de datos utilizado. Tambi´en indique en caso que
haya realizado alg´un tipo de preprocesamiento de los datos.

El texto elegido es el primer libre de la saga Canción de hielo y fuego, titulado Juego de Tronos. El archivo es un .txt de 1.8mb, 25656 líneas, 1772149 carácteres y 316495 palabras.

El preprocesamiento 

### Actividad 8:

### Actividad 9: