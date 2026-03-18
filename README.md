# Riesgo de Crédito E2E — Lending Club Showcase

**Plataforma de inteligencia de riesgo crediticio de nivel académico-paper**, construida sobre datos históricos de Lending Club (2007–2020). Implementa un pipeline completo de Machine Learning + Investigación de Operaciones con predicción conformal, optimización robusta de portafolio y análisis causal.

[![Demo en Vivo](https://img.shields.io/badge/Demo%20en%20Vivo-Streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://lending-club-showcase.streamlit.app/)
[![CI](https://img.shields.io/github/actions/workflow/status/EigenCharlie/Lending-Club-End-to-End/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/EigenCharlie/Lending-Club-End-to-End/actions/workflows/ci.yml)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-00A86B?style=flat-square&logo=dvc&logoColor=white)](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End)
[![MLflow](https://img.shields.io/badge/MLflow-Experimentos-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End.mlflow)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow?style=flat-square&logo=yandex&logoColor=white)](https://catboost.ai/)
[![License: MIT](https://img.shields.io/badge/Licencia-MIT-lightgrey?style=flat-square)](LICENSE)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-DA7857?style=flat-square&logo=anthropic&logoColor=white)](https://claude.ai/code)

---

## Demo en Vivo

**[lending-club-showcase.streamlit.app](https://lending-club-showcase.streamlit.app/)**

Dashboard interactivo con **31 páginas** organizadas en 4 secciones: desde fundamentos metodológicos hasta evidencia paper-grade, con análisis causal, optimización robusta, provisiones IFRS9 y material activo para 3 publicaciones académicas.

---

## ¿Qué hace esta plataforma?

Una tesis de maestría que va más allá del modelo puntual. El problema central es que las estimaciones de probabilidad de default (PD) puntuales ignoran la incertidumbre, produciendo portafolios frágiles ante cambios de distribución. La solución implementada:

```text
CatBoost PD → Calibración Auto (Platt/Isotonic/Venn-Abers)
  → Intervalos Conformales Mondrian [PD_low, PD_high]
  → Conjuntos de Incertidumbre Box
  → Optimización Robusta Pyomo/HiGHS
  → Portafolio Óptimo con Garantías Estadísticas
```

**Por qué importa:**
- Los intervalos bootstrap no tienen garantías de cobertura en muestras finitas
- Los intervalos bayesianos requieren supuestos distribucionales
- **La predicción conformal es libre de distribución, con garantías matemáticas de cobertura**

---

## Métricas Clave del Pipeline (OOT 2018–2020)

| Componente | Métrica | Valor |
|---|---|---|
| PD Model (CatBoost) | AUC (OOT) | **0.7130** |
| PD Model | Brier Score | 0.1545 |
| PD Model | ECE (calibración) | 0.0059 |
| Calibración | Método seleccionado | **Venn-Abers** |
| Conformal 90% | Cobertura real | **92.5%** |
| Conformal 95% | Cobertura real | 95.9% |
| Survival RSF | C-index | 0.6715 |
| Fairness | Checks aprobados | **6/6** |
| ECL Base | Pérdida esperada | $1.0B |
| ECL Severo | Escenario adverso | $1.8B |
| Portafolio Robusto | Retorno neto | $183K |
| SPO+ vs Two-stage | Reducción de regret | **49.1%** |

> **Datos**: 1,346,311 préstamos (train) + 237,584 (calibración) + 276,869 (test OOT).
> Splits temporales estrictos: sin data leakage entre conjuntos.

---

## Arquitectura del Pipeline

```text
data/raw/lending_club.csv
  ↓ limpieza y eliminación de fuga temporal
data/processed/{train, calibration, test}.parquet
  ↓ feature engineering (WOE, interacciones, contratos Pandera)
  ├── train_pd_model.py        → PD CatBoost + calibrador
  ├── generate_conformal...py  → Intervalos Mondrian por grade
  ├── estimate_causal...py     → CausalForestDML (EconML)
  ├── run_survival_analysis.py → Cox PH + RSF (scikit-survival)
  ├── forecast_default_rates.py → statsforecast / mlforecast
  ├── run_ifrs9_sensitivity.py  → ECL multi-escenario
  ├── optimize_portfolio.py    → Pyomo + HiGHS (7 políticas)
  └── run_fairness_audit.py    → Paridad demográfica, EO, DIR
```

---

## Páginas del Dashboard

### Libro y Fundamentos

| Página | Descripción |
|---|---|
| Resumen Ejecutivo | KPIs en vivo del pipeline, estado del baseline canónico |
| Glosario y Fundamentos | Conceptos de riesgo crediticio, IFRS9, Basel y ML |

### Pipeline Operativo

| Página | Descripción |
|---|---|
| Visión End-to-End | Flujo completo del pipeline con diagrama interactivo |
| Arquitectura y Linaje de Datos | Linaje dbt, contratos Feast, DuckDB marts |
| Mapa Integrado de Métodos | Conexión entre todos los componentes metodológicos |
| Ingeniería de Features | WOE/IV con OptBinning, importancias SHAP, correlaciones |
| Laboratorio de Modelos | Comparativa LR vs CatBoost, curvas calibración, Venn-Abers |
| Cuantificación de Incertidumbre | Intervalos conformales Mondrian, 6 variantes, backtesting |
| Panorama Temporal | Forecasting de tasas de default (AutoARIMA, NBEATS, NHITS) |
| Análisis de Supervivencia | Cox PH + RSF, curvas KM, Brier score, impacto en ECL |
| Optimizador de Portafolio | 7 políticas de asignación, frontera de eficiencia robusta |
| Provisiones IFRS9 | Staging S1/S2/S3, ECL multi-escenario, SICR conformal |

### Insight Factory

| Página | Descripción |
|---|---|
| Atlas de Evidencia | Galería de figuras de los 13 notebooks del pipeline |
| Historia de Datos | EDA narrativa: distribuciones, tasas de default, tendencias |
| Explicabilidad e Interpretabilidad | SHAP global/local, ALE plots, casos de frontera |
| Inteligencia Causal | CATE por subgrupo, política causal, simulación DoWhy |
| Simulación A/B | Robust vs non-robust: retorno, Sharpe, atribución por grade |
| Chat con Datos | Consultas en lenguaje natural sobre los datasets (DuckDB) |
| Benchmark RAPIDS GPU | cuML KMeans, UMAP, HDBSCAN, cuGraph vs CPU |

### Gobernanza y Libro

| Página | Descripción |
|---|---|
| Gobernanza del Modelo | Umbrales operacionales, semántica de thresholds, SR 11-7 |
| Stack Tecnológico | Todas las librerías y herramientas con justificación metodológica |
| Tesis Especialización | Alpha sweep Mondrian vs global, análisis de Pareto cobertura-ancho |
| Contribución de Tesis | Proposición central: CP distribution-free como innovation boundary |
| Panorama de Investigación | ~80 papers del estado del arte, mapa de citas |
| **Paper Estrella** | Predict-then-Optimize: α-CP ↔ Γ-robustez, SPO+, regret comparison |
| Paper 1 (Draft Hist.) | CP + Robust Optimization — borrador histórico (absorbido por Estrella) |
| Paper 2: IFRS9 E2E | Staging conformal, ECL con intervalos, BMA vs CP (JBF) |
| Paper 3: Mondrian CP | Cobertura condicional por subgrupo, backtesting (COPA 2026) |
| Buenas Prácticas | DVC, MLflow, pre-commit, DagsHub, reproducibilidad |

---

## Papers en Preparación

| Paper | Venue objetivo | Estado |
|---|---|---|
| **Paper 3 — Mondrian CP para riesgo crediticio** | COPA 2026 (Mayo) | 70% listo |
| **Paper 2 — IFRS9 E2E con CP** | JBF / JORS (Sep 2026) | 60% listo |
| **Paper Estrella — Predict-then-Optimize** | Management Science / OR / EJOR (Dic 2026) | 40% listo |

**Contribución teórica central (Paper Estrella)**: Un bound que conecta formalmente el nivel de confianza conformal $\alpha$ con el parámetro de robustez $\Gamma$ de Bertsimas & Sim, convirtiendo intervalos CP en conjuntos de incertidumbre con garantías teóricas para optimización robusta de portafolio.

---

## Stack Tecnológico

| Categoría | Herramientas |
|---|---|
| **ML / PD** | CatBoost 1.2, scikit-learn 1.6, LightGBM, Optuna (320 trials HPO) |
| **Conformal** | MAPIE 1.3 (SplitConformalRegressor + Mondrian), crepes |
| **Series Temporales** | statsforecast 2.0, mlforecast 0.13, hierarchicalforecast 1.0 |
| **Supervivencia** | lifelines 0.30, scikit-survival 0.24 |
| **Causal** | EconML 0.16 (CausalForestDML), DoWhy 0.12 |
| **Optimización** | Pyomo 6.8, HiGHS (open-source MIP), CVXPY 1.6 |
| **MLOps** | DVC 3.56, MLflow 3.9, DagsHub, pandera 0.22 |
| **GPU (research)** | RAPIDS cuDF, cuML, cuGraph (NVIDIA) |
| **Dashboard** | Streamlit 1.54, DuckDB 1.4, dbt-duckdb 1.10, Feast 0.59 |
| **Reproducibilidad** | uv, ruff, pre-commit, nbstripout, pytest (681 tests) |

---

## Cómo navegar el dashboard

> [!TIP]
> Empieza por **Resumen Ejecutivo** para ver los KPIs en vivo del baseline canónico, luego ve a **Visión End-to-End** para entender el flujo completo antes de explorar las páginas temáticas.

> [!NOTE]
> Las páginas de **Gobernanza y Libro** contienen el material de investigación activo: los 3 papers en preparación, el alpha sweep de Mondrian, y el análisis BMA vs CP.

> [!IMPORTANT]
> El showcase despliega **artefactos pre-computados** del baseline `champion-2026-03-12-mega-definitive`. Las métricas en vivo se leen desde los archivos `.json` / `.parquet` del bundle de despliegue.

---

## Reproducibilidad

El código fuente completo, scripts de entrenamiento, y pipeline DVC están en el repositorio principal:

**[EigenCharlie/Lending-Club-End-to-End](https://github.com/EigenCharlie/Lending-Club-End-to-End)**

```bash
# Instalar dependencias
uv sync --extra dev

# Reconstruir pipeline canónico (DVC)
uv run dvc repro

# Correr rebuild operacional congelado
uv run python scripts/run_canonical_rebuild.py --run-tag canonical-local

# Lanzar dashboard localmente
uv run streamlit run streamlit_app/app.py
```

Experimentos registrados en MLflow → DagsHub:
[dagshub.com/EigenCharlie94/Lending-Club-End-to-End.mlflow](https://dagshub.com/EigenCharlie94/Lending-Club-End-to-End.mlflow)

---

## Sobre el Proyecto

Tesis de maestría en gestión de riesgo crediticio — **Carlos Vergara**.
Pipeline de investigación reproducible con Streamlit como capa interactiva y Quarto como destino editorial del libro de tesis.

Fuente de datos: [Lending Club 2007–2020Q3](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1/data) (Kaggle).
