# Isolation Forest — hiperparametros CMS-80

Espacio: 7 descriptores. Score util: -decision_function. predict solo es un corte.

Barrido (fondo sintetico 400 + 20 outliers plantados):
- n_estimators 50/100/200/400: separacion 0.200/0.192/0.185/0.185. 200 basta.
- contamination 0.01 y 0.02: FP=0. 0.05+: FP=0.05. El score no cambia.
- max_samples 64: sep 0.128. 256/auto: 0.185.
- AUC plantado=1.0 (caso facil). std entre semillas ~0.01.

Defaults:
- global: n_estimators=200, contamination=0.02, max_samples=256, random_state=42
- tiles: n_estimators=120, contamination=0.08, max_samples=auto
