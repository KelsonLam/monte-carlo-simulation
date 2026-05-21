"""Command line entry point for the Monte Carlo simulation.

Examples
--------
Run with everything from config.yaml::

    python scripts/run_simulation.py

Use the bootstrap model on real history instead of parametric GBM::

    python scripts/run_simulation.py --model bootstrap

Save the charts::

    python scripts/run_simulation.py --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monte_carlo.models import GBMModel, BootstrapModel
from monte_carlo.simulation import PlanConfig, simulate
from monte_carlo import analysis, plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Monte Carlo savings simulation.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    p.add_argument("--model", choices=["gbm", "bootstrap"], help="Return model.")
    p.add_argument("--paths", type=int, help="Number of simulated futures.")
    p.add_argument("--years", type=int, help="Horizon in years.")
    p.add_argument("--no-cache", action="store_true", help="Force a fresh download.")
    p.add_argument("--save-plots", action="store_true", help="Write charts to results/.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_model(cfg: dict, model_type: str, use_cache: bool):
    mcfg = cfg["model"]
    steps = mcfg["steps_per_year"]
    if model_type == "gbm":
        return GBMModel(
            annual_return=mcfg["annual_return"],
            annual_volatility=mcfg["annual_volatility"],
            steps_per_year=steps,
        )
    # bootstrap: pull history and resample monthly returns
    from monte_carlo.data import YFinanceLoader, monthly_returns

    print(f"Loading {mcfg['ticker']} history for the bootstrap ...")
    loader = YFinanceLoader(use_cache=use_cache)
    prices = loader.load(mcfg["ticker"], mcfg["history_start"], mcfg["history_end"])
    return BootstrapModel(monthly_returns(prices))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_type = args.model or cfg["model"]["type"]
    plan = PlanConfig(
        initial_investment=cfg["plan"]["initial_investment"],
        annual_contribution=cfg["plan"]["annual_contribution"],
        horizon_years=args.years or cfg["plan"]["horizon_years"],
        steps_per_year=cfg["model"]["steps_per_year"],
    )
    target = cfg["plan"]["target_value"]
    n_paths = args.paths or cfg["simulation"]["n_paths"]
    rng = np.random.default_rng(cfg["simulation"]["seed"])

    model = build_model(cfg, model_type, use_cache=not args.no_cache)

    print(f"Simulating {n_paths:,} futures over {plan.horizon_years} years "
          f"with the {model_type} model ...")
    result = simulate(model, plan, n_paths, rng)

    stats = analysis.summarize(result.terminal_values, target, result.total_contributed)
    print(f"\nTotal paid in over {plan.horizon_years} years: "
          f"{result.total_contributed:,.0f}")
    print(f"Target: {target:,.0f}\n")
    print(analysis.format_summary(stats))

    if args.save_plots:
        f1 = plotting.plot_fan_chart(result.paths, plan.steps_per_year)
        f2 = plotting.plot_terminal_histogram(
            result.terminal_values, target=target,
            break_even=result.total_contributed,
        )
        out1 = plotting.save_figure(f1, "results/fan_chart.png")
        out2 = plotting.save_figure(f2, "results/terminal_histogram.png")
        print(f"\nSaved charts to {out1} and {out2}")


if __name__ == "__main__":
    main()
