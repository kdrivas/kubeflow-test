# < NOMBRE DEL REPO >

## Objetivo
< Objetivo técnico y de negocio de la iniciativa, qué negocio lo pide? >

## Responsables
< Contacto del Negocio, DS, MLE >

## Modelos
< Qué modelos se entrenan, id_modelo (según nomenclatura de modelos) y definición de su target >

## Insights
< Principales insights durante el EDA (Variables potenciales, validación de hipótesis del negocio, ...) >  
< Principales insights durante el entrenamiento (Variables más importantes, decisión de selección del algoritmo) >

## Training Pipeline
### Target Engineering
< Fuente para construir el/los targets >  
< Entidad (ejm persona, cliente_poliza, vehiculo, ...) >  
< Filtros o limpieza adicional en la construccion del target >  
< Estadísticas sobre el target (% target, ...) >

### Enrich
< FS: Mencionar los grupos de variables que se usan (ejm demográficas, RCC, ...) >  
< Data Entries: Mencionar las variables que se usan y la lógica de su construcción  >  

### Feature Engineering
< Resumen del preprocesamiento y feature engineering sobre la data >

### Training
< Algoritmo, submodelos (ejm stacking/blending de modelos), HPO >

### Predicción
< Mencionar si el modelo es offline u online >

### Postprocesamiento: Scoring
< Si se usa el modelo (posiblemente con otros modelos) para generar un score, resumir la lógica de este scoring >

## Checklist
### Data
- [ ] Target
  - [ ] MDM (Data Estructurada) o GCS ADS (Data No estructurada)
    (No: Data Entry)
- [ ] Features (Data Estructurada)
  - [ ] FS (o MDM)
    (No: Data Entry, al menos 1 variable)
### Modelo
- [ ] Training Pipeline
  - [ ] Nomenclatura (< entidad >_< aplicacion >_< negocio >_< objective >)
  - [ ] Package
  - [ ] Artifacts (model, preprocess) en GCS
- [ ] Re-Training Pipeline
  - [ ] Reentrenamiento parametrizado
- [ ] Prediction
  - [ ] Modelos Batch: Scheduled Prediction Pipeline (predicciones compartido al MDM) o
  - [ ] Modelos Online: predict.py
### Código
- [ ] Documentación
  - [ ] Nomenclatura modelos
  - [ ] README
  - [ ] Data Report
  - [ ] Tags (Data Catalog)
- [ ] CI/CD Training Pipeline
  - [ ] Enviorment Variables
  - [ ] Trigger CloudBuild
  - [ ] Sucsess dev/test/prod

