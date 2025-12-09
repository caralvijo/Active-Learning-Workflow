# Active Learning Workflow para Entrenamiento de Potenciales Basados en Nequix

## Resumen
Este repositorio presenta un flujo de trabajo completo para la ejecución de ciclos de *active learning* orientados al entrenamiento de modelos basados en Nequix. El enfoque integra generación de configuraciones, evaluación con ensambles de modelos, detección de alta incertidumbre y retroalimentación mediante cálculos de primeros principios. El objetivo es proporcionar una plataforma reproducible, modular y extensible para ampliar bases de datos atómicas y optimizar modelos de predicción de energías y fuerzas.

## Descripción General del Método
El método sigue un esquema iterativo clásico de *active learning*, estructurado en cuatro componentes principales:

1. **Generación de configuraciones iniciales.**
   Conjuntos de estructuras se obtienen a partir de dinámicas moleculares, distorsiones, muestreos aleatorios o cualquier técnica empleada por el usuario. Estas configuraciones constituyen el punto de partida del ciclo.

2. **Evaluación mediante un ensamble de modelos Nequix.**
   Un conjunto de modelos es empleado para estimar energías y fuerzas en cada configuración candidata. La dispersión entre modelos se interpreta como medida de incertidumbre y se utiliza para decidir qué configuraciones requieren cálculos de referencia.

3. **Selección activa.**
   Configuraciones con incertidumbre elevada son seleccionadas para su evaluación con un método de mayor fidelidad, típicamente DFT. La estrategia permite concentrar el esfuerzo computacional en regiones relevantes del espacio configuracional.

4. **Retroalimentación y reentrenamiento.**
   Los resultados de DFT se incorporan a la base de datos de entrenamiento, tras lo cual los modelos del ensamble se reentrenan. Este proceso mejora progresivamente la capacidad predictiva del sistema.

## Componentes del Repositorio
El repositorio contiene módulos que implementan cada etapa del ciclo:

- **Carga de modelos Nequix.** Mecanismos robustos para cargar checkpoints y reconstruir modelos desde estado guardado.
- **Conversión a calculadores ASE.** Integración transparente para permitir el uso de modelos Nequix dentro de pipelines de ASE.
- **Manejo del ensamble.** Utilidades para evaluar múltiples modelos, computar medias, varianzas e índices de incertidumbre.
- **Selección activa.** Herramientas para filtrar configuraciones mediante criterios cuantitativos de incertidumbre y umbrales definidos por el usuario.
- **Interfaz de expansión de dataset.** Soporte para anexar configuraciones seleccionadas junto con sus cálculos de referencia.

## Dependencias
El flujo de trabajo requiere las siguientes bibliotecas:

- Python 3.9 o superior
- ASE
- Nequix y nequixase
- NumPy
- Torch (para inferencia de modelos)

## Uso General
A continuación se presenta una guía técnica para ejecutar el ciclo completo de active learning.

### 1. Preparación del entorno
```bash
# Crear entorno
python3 -m venv nequix-al
source nequix-al/bin/activate

# Instalar dependencias requerida
pip install ase torch numpy
pip install nequix nequixase   # si se distribuyen por pip
```

### 2. Estructuras iniciales
Coloque sus configuraciones iniciales en un directorio, por ejemplo:
```
data/initial_structures/
```
Los formatos compatibles incluyen `.xyz`, `.traj` o cualquier formato soportado por ASE.

### 3. Evaluación con ensamble de modelos
```bash
python run_ensemble.py \
    --models models/model_1.pt models/model_2.pt models/model_3.pt models/model_4.pt models/model_5.pt \
    --structures data/initial_structures/ \
    --output ensemble_results.json
```
Esto genera un archivo con energías, fuerzas y medida de incertidumbre para cada modelo.

### 4. Selección activa
```bash
python select_high_uncertainty.py \
    --ensemble ensemble_results.json \
    --threshold 0.08 \
    --output selected_structures/
```
El directorio `selected_structures/` contendrá sólo configuraciones con incertidumbre superior al umbral.

### 5. Cálculos DFT de referencia
El usuario debe conectar su propio backend DFT (VASP, Quantum ESPRESSO, CASTEP, etc.). Ejemplo genérico:
```bash
bash launch_dft.sh selected_structures/
```
Los resultados deben guardarse en:
```
data/dft_results/
```

### 6. Expansión del dataset y reentrenamiento
```bash
python update_dataset.py \
    --new data/dft_results/ \
    --dataset data/training_set/

python train_ensemble.py \
    --dataset data/training_set/ \
    --outdir models_new/
```

### 7. Repetición del ciclo
Vuelva al paso 3 usando los modelos actualizados.

## Objetivo del Repositorio
Este flujo de trabajo se propone como un punto de partida para proyectos en los que es necesario explorar eficientemente el espacio configuracional mediante estrategias activas. Está diseñado para facilitar experimentación, auditoría y reproducibilidad.

## Validación del Modelo
La validación del modelo se realiza comparando predicciones del ensamble contra resultados de referencia obtenidos mediante DFT. Se recomiendan las siguientes métricas:

- Error cuadrático medio (RMSE) en energías.
- RMSE en fuerzas atómicas.
- Desviación media absoluta (MAE).
- Distribución del error por especie química.
- Evaluación estructural mediante curvas energía–volumen o relajaciones.

Ejemplo de validación:
```bash
python validate_models.py \
    --models models_new/*.pt \
    --testset data/validation_set/ \
    --output validation_report.json
```

## Ejemplo de launch_dft.sh para VASP
Ejemplo básico para ejecutar DFT en VASP usando un script por estructura seleccionada:

```bash
#!/bin/bash

STRUCT_DIR=$1
OUTDIR=data/dft_results
mkdir -p $OUTDIR

for s in $(ls $STRUCT_DIR/*.vasp); do
    NAME=$(basename ${s%.vasp})
    WORKDIR=dft_runs/$NAME
    mkdir -p $WORKDIR

    # Copiar entrada
    cp $s $WORKDIR/POSCAR
    cp INCAR KPOINTS POTCAR $WORKDIR/

    cd $WORKDIR
    mpirun -np 16 vasp_std > vasp.out
    cd -

    # Copiar resultados
    cp $WORKDIR/OUTCAR $OUTDIR/${NAME}_OUTCAR
    cp $WORKDIR/CONTCAR $OUTDIR/${NAME}_CONTCAR
    cp $WORKDIR/vasp.out $OUTDIR/${NAME}_log

done
```

## Diagrama del Pipeline
A continuación se presenta un diagrama simplificado en formato ASCII:

```
     +---------------------------+
     |  Estructuras iniciales   |
     +------------+--------------+
                  |
                  v
     +---------------------------+
     |  Ensamble Nequix:         |
     |  predicción E/F + var     |
     +------------+--------------+
                  |
                  v
     +---------------------------+
     | Selección activa:         |
     | filtrar por incertidumbre |
     +------------+--------------+
                  |
                  v
     +---------------------------+
     |  Cálculos DFT de alta     |
     |  fidelidad                |
     +------------+--------------+
                  |
                  v
     +---------------------------+
     | Expansión del dataset     |
     +------------+--------------+
                  |
                  v
     +---------------------------+
     | Reentrenamiento del       |
     | ensamble Nequix           |
     +------------+--------------+
                  |
                  v
            (Repetir ciclo)
```

## Contacto
Para dudas, comentarios o sugerencias, el usuario puede modificar o extender libremente los módulos, así como abrir solicitudes de mejora en el repositorio.

