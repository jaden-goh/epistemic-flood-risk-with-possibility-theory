
# Possibilistic POT Hazard — README

## File structure

```
your-project/
├─ data/
│  └─ ardieres.csv         # discharge time series
├─ poss_predict.py          # build posterior, compute 2 numbers
├─ potanalysis.py           # declustering, exceedances, MRL/TC
├─ run_poss_value.py        # minimal driver
└─ README.md
```

---

## Data format

ardieres.csv with:
- `obs`: discharge
- `date` (or `time`): datetime or decimal year

Example:
```csv
timestamp,obs
2002.78367579909,0.124
2002.78630136986,0.104
2002.78816400304,0.114
```

---

## Quick start

1. Set a threshold `u`, declustering gap, and event set `B`
2. Run:

```bash
python possibility.py
```

Output:
```
Upper probability  P̄(X ∈ B | D) 
Necessity (lower)  P_(X ∈ B | D) 

---

## Theory

### 1. GPD exceedances

- Threshold $(u)$, exceedances $(z = x - u > 0)$
- $(\theta = (\xi, \sigma))$ via GPD

### 2. Possibility distribution

- $(\pi(\theta|D) = \exp(\ell(\theta) - \max \ell))$

### 3. Predictive bounds

- $(\overline{P}(B|D) = \sup_\theta T(\pi(\theta), P_\theta(B)))$
- $(\underline{P}(B|D) = 1 - \sup_\theta T(\pi(\theta), 1 - P_\theta(B)))$

- We also calculate a per-year rate $(\lambda_0)$ via Poisson.

---

## Potential Extensions

- Use Lisflood/GLUE for depth maps
- Use α-cuts on $(\pi(\theta|D))$ to produce hazard bands
