"""Monte Carlo simulation for a long-horizon savings plan.

The question it answers: given a starting balance, a yearly contribution, and a
view on returns, what is the distribution of outcomes after N years, and how
likely is the plan to hit its target?

Modules:

    models      ways to generate future returns (parametric GBM, or bootstrap)
    simulation  roll the returns forward into balance paths, with contributions
    analysis    percentiles, probability of hitting the target, VaR and CVaR
    plotting    the fan chart of paths and the histogram of final balances
    data        load price history (for the bootstrap model) behind a loader
"""

__version__ = "0.1.0"
