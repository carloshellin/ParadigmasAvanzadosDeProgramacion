# Paradigmas Avanzados de Programación

Este repositorio almacena las prácticas realizadas durante el curso 2020/2021 para la asignatura [PARADIGMAS AVANZADOS DE PROGRAMACIÓN (781004)](https://www.uah.es/es/estudios/estudios-oficiales/grados/asignatura/Paradigmas-Avanzados-de-Programacion-781004/) del departamento Ciencias de la Computación de la Universidad de Alcalá.

## Contenido

Las carpetas son las siguientes:

- _CUDA_: incluye la entrega de pequeños ejercicios realizados en CUDA, así como también las entregas de la convolución de matrices en GPU mediante un bloque, múltiples bloques y memoria global/compartida (incluyendo en las carpetas _extra_ la convolución completa).
- _Cloud_: informe y presentación del bloque _Compute y Contenedores_ en Azure.
- _Scala_: ejercicios realizados en Scala y las entregas de la implementación del juego [Bejeweled](http://www.bejeweled.com/).

## Uso

En el caso de la carpeta _CUDA_ es necesario para compilar los códigos tener una [GPU compatible con CUDA](http://developer.nvidia.com/cuda-gpus), instalar el [NVIDIA CUDA Toolkit](http://developer.nvidia.com/cuda-downloads) y para en el caso de Windows instalar [Visual Studio](https://visualstudio.microsoft.com/es/vs/) o gcc para Linux. Se puede compilar creando un proyecto de Visual Studio o usando el comando `nvcc` para compilar los ficheros _.cu_ y ejecutarlo desde la consola.

Para _Scala_ es necesario tener instalado el propio [Scala](https://scala-lang.org/) para poder compilar los ficheros _.scala_ con el comando `scalac` y realizar su ejecución desde la consola usando `scala`. También se puede instalar [Scala IDE](http://scala-ide.org/) y realizar la compilación/ejecución creando un nuevo proyecto.

## Licencia

El Real Decreto 1791/2010, de 30 de diciembre, por el que se aprueba el Estatuto del Estudiante Universitario, dedica su artículo 7 a los derechos comunes de los estudiantes universitarios y se establece lo siguiente:

> Al reconocimiento de la autoría de los trabajos elaborados durante sus estudios y a la protección de la propiedad intelectual de los mismos.

Además, en el artículo 27 sobre los trabajos y memorias de evaluación, se indica que:
> La publicación o reproducción total o parcial de los trabajos (...) o la utilización para cualquier otra finalidad distinta de la estrictamente académica, requerirá la autorización expresa del autor o autores, de acuerdo con la legislación de propiedad intelectual.
