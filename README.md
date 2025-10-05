Possibilistic POT Hazard — README
File structure
your-project/
├─ data/
│  └─ ardieres.csv         # discharge time series
├─ poss_predict.py          # build posterior, compute 2 numbers
├─ potanalysis.py           # declustering, exceedances, MRL/TC
├─ run_poss_value.py        # minimal driver
└─ README.md


Data format
CSV with:

obs: discharge

date (or time): datetime or decimal year

Example:

date,obs
1990-01-01,42.7
1990-01-02,44.1


Quick start
Set a threshold $$ u $$, declustering gap, and event set $$ B $$

Run:

python run_poss_value.py


Output:

Upper probability  P̄(X ∈ B | D) = 0.31
Necessity (lower)  P_(X ∈ B | D) = 0.15


Theory summary
1. GPD exceedances
Threshold $$ u $$, exceedances $$ z = x - u > 0 $$

$$ \theta = (\xi, \sigma) $$ via GPD

2. Possibility distribution
$$ \pi(\theta|D) = \exp(\ell(\theta) - \max \ell) $$

3. Predictive bounds
$$ \overline{P}(B|D) = \sup_\theta T(\pi(\theta), P_\theta(B)) $$

$$ \underline{P}(B|D) = 1 - \sup_\theta T(\pi(\theta), 1 - P_\theta(B)) $$

Optional per-year form includes $$ \lambda_0 $$ via Poisson rate.

Extend
Use Lisflood/GLUE for depth maps

Use $$ \alpha $$-cuts on $$ \pi(\theta|D) $$ to produce hazard bands