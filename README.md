# Wild honey bee habitat suitability — eastern High Atlas (upper Ziz), Morocco

Data and code for the paper *Benchmarking four machine-learning classifiers with spatial
cross-validation for wild honey bee habitat mapping in Morocco*.

## Contents

| Folder | Content |
|---|---|
| `data/predictors_300m.csv` | 13 environmental predictors on the 300 m grid (48,499 cells; UTM 30N, EPSG:32630) |
| `data/training_data_generalised_1km.csv` | 150 presences + 150 verified absences with all model features. Coordinates are rounded to 1 km (`x_km_1km`, `y_km_1km`) to protect nest locations |
| `data/*.csv`, `best_params.json` | Tuned hyper-parameters, test metrics, spatial-CV results, permutation importance, SHAP importance, area per suitability class |
| `rasters/` | Suitability maps (GeoTIFF, 300 m, UTM 30N) for RF, XGBoost, SVM, MLP and the ensemble; ensemble classes; eastness layer |
| `scripts/` | Python scripts (data preparation, training, evaluation, SHAP, maps) |
| `figures/` | Figures of the paper |

## Grid
300 m cells, 246 rows × 415 columns, lower-left corner (266900, 3535978), EPSG:32630.

## Field data
Presences and absences were verified between February and July 2026 by field survey and
interviews with 12 shepherds. Exact nest coordinates are not published; they can be
requested from the authors for research purposes.

## Software
Python 3, scikit-learn, XGBoost, SHAP, rasterio, pandas, numpy, matplotlib.

## Order of the scripts
`build_original.py` → `build_v2.py` → `train.py` → `evaluate.py` → `shap_run.py` →
`shap_figs.py` → `curves_maps2.py` → `maps_figs.py` → `maps_single_nopts.py` → `panel.py` → `maps_extra.py`.
Note: `build_v2.py` and the map scripts use the exact coordinates, which are not included;
`train.py` and `evaluate.py` can be run on the generalised training file (features are complete).

## Licence
[TODO: choose a licence, e.g. CC BY 4.0 for data and MIT for code]

## Citation
[TODO: add the paper citation and DOI once published]
