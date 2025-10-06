#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms

from inspect import signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ===========================
# Dataset configuration (simplified)
# ===========================
DATASETS = [
    {
        "name": "heart",
        "path": "./dataset/heart.csv",
        "sensitive_attrs": ["age", "sex", "cp"],   # columns to privatize
        "categorical_attrs": ["sex", "cp","fbs","restecg","exang","slope","ca","thal"],        # categoricals among them (and possibly others)
        "target": "target"
    },
    {
        "name": "diabetes",
        "path": "./dataset/Diabetes.csv",
        "sensitive_attrs": ["Age", "Gender", "BMI"],
        "categorical_attrs": ["Gender"],
        "target": "Diagnosis"
    }
]

# ===========================
# Experiment settings
# ===========================
RUNS        = 30
SEEDS       = list(range(1, RUNS + 1))
BUDGETS     = list(range(1, 11, 1))       # total per-record epsilon (e.g., 1,5,9)
POP_SIZE    = 40
GENERATIONS = 50
CX_PROB     = 0.7
MUT_PROB    = 0.3

# Shared noise replicates for fairness (common random numbers)
NOISE_REPLICATES = (0, 7, 13)


# ===========================
# Helpers for config
# ===========================
def is_sensitive(col: str, ds_cfg: dict) -> bool:
    return col in ds_cfg["sensitive_attrs"]

def is_categorical(col: str, ds_cfg: dict) -> bool:
    return col in ds_cfg["categorical_attrs"]

def feature_columns(df: pd.DataFrame, ds_cfg: dict) -> list:
    """All feature columns = every column except the target."""
    return [c for c in df.columns if c != ds_cfg["target"]]


# ===========================
# LDP mechanisms
# ===========================
def laplace_numeric(series: pd.Series, eps: float, rng: np.random.Generator,
                    lo: float, hi: float) -> np.ndarray:
    """
    Bounded numeric LDP via Laplace; bounds provided by caller (computed on TRAIN split).
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    lo2 = float(lo); hi2 = float(hi)
    if not np.isfinite(lo2): lo2 = np.nanmin(x)
    if not np.isfinite(hi2): hi2 = np.nanmax(x)
    if not np.isfinite(lo2): lo2 = 0.0
    if not np.isfinite(hi2) or hi2 <= lo2: hi2 = lo2 + 1.0

    x = np.clip(x, lo2, hi2)
    span = hi2 - lo2
    xn = (x - lo2) / span
    b = 1.0 / max(eps, 1e-8)
    yn = np.clip(xn + rng.laplace(0.0, b, size=xn.shape), 0.0, 1.0)
    return yn * span + lo2


def exp_mech_categorical(series, eps, rng, categories):
    cats = np.asarray(categories)
    k = len(cats)
    if k == 1:
        return np.repeat(cats[0], len(series))  # nothing to randomize

    cat = pd.Categorical(series, categories=cats, ordered=False)
    true_idx = cat.codes  # -1 if unseen

    keep_w = np.exp(eps)
    p_keep = keep_w / (keep_w + (k - 1))

    out_idx = np.empty(len(true_idx), dtype=int)
    for i, t in enumerate(true_idx):
        u = rng.random()
        if t >= 0 and u < p_keep:
            out_idx[i] = t
        else:
            # sample uniformly from all k categories (if t<0) or from the other k-1 (if t>=0)
            if t < 0:
                out_idx[i] = rng.integers(0, k)
            else:
                r = rng.integers(0, k - 1)
                out_idx[i] = r if r < t else r + 1
    return cats[out_idx]


# ===========================
# Bounds/domains from TRAIN split
# ===========================
def compute_bounds_and_domains(df_train: pd.DataFrame, ds_cfg: dict) -> dict:
    """
    Compute per-column metadata from CLEAN training features:
      - numeric: {'type':'num','lo':..., 'hi':...}
      - categorical: {'type':'cat','cats': np.array([...])}
    """
    meta = {}
    for col in df_train.columns:
        if is_categorical(col, ds_cfg):
            cats = pd.Categorical(df_train[col]).categories.to_numpy()
            if len(cats) == 0:
                cats = np.array([0, 1], dtype=int)  # fallback
            meta[col] = {"type": "cat", "cats": cats}
        else:
            x = pd.to_numeric(df_train[col], errors="coerce")
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
            meta[col] = {"type": "num", "lo": lo, "hi": hi}
    return meta


# ===========================
# Anonymizer (only sensitive columns)
# ===========================
def anonymize_df(df_in: pd.DataFrame, ds_cfg: dict, budgets: list,
                 bounds_meta: dict, noise_seed: int = None) -> pd.DataFrame:
    """
    Noise only ds_cfg['sensitive_attrs'] with per-attribute budgets (same order).
    Non-sensitive columns are copied as-is. Domains/bounds come from TRAIN split.
    """
    df = df_in.copy()
    sens_cols = ds_cfg["sensitive_attrs"]
    assert len(sens_cols) == len(budgets), "Budgets must match # sensitive attributes"

    base = np.random.SeedSequence(noise_seed) if noise_seed is not None else np.random.SeedSequence()
    kids = base.spawn(len(sens_cols))

    for i, col in enumerate(sens_cols):
        rng = np.random.default_rng(kids[i])
        eps = float(budgets[i])
        meta = bounds_meta[col]
        if is_categorical(col, ds_cfg):
            df[col] = exp_mech_categorical(df[col], eps, rng, categories=np.asarray(meta["cats"]))
        else:
            df[col] = laplace_numeric(df[col], eps, rng, meta["lo"], meta["hi"])
    return df


# ===========================
# Utility evaluation (train-noisy / test-clean)
# ===========================
def coerce_target_to_int(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        return y.to_numpy()
    return pd.Categorical(y).codes

def make_downstream_pipeline(df: pd.DataFrame, ds_cfg: dict, seed: int) -> Pipeline:
    """
    Build a sklearn pipeline:
      - Standardize numeric
      - One-hot encode categorical
      - Logistic Regression
    """
    cols = list(df.columns)
    num_cols = [c for c in cols if not is_categorical(c, ds_cfg)]
    cat_cols = [c for c in cols if is_categorical(c, ds_cfg)]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        enc_kwargs = {"handle_unknown": "ignore"}
        if "sparse_output" in signature(OneHotEncoder).parameters:
            enc_kwargs["sparse_output"] = False   # sklearn >=1.2
        else:
            enc_kwargs["sparse"] = False          # sklearn <1.2
        enc = OneHotEncoder(**enc_kwargs)
        transformers.append(("cat", enc, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(random_state=seed)
    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return pipe

def utility_scores_publish(df_clean: pd.DataFrame, ds_cfg: dict, budgets: list,
                           seed: int, reps=NOISE_REPLICATES) -> dict:
    """
    Train on NOISY TRAIN features (only sensitive columns noised), evaluate on CLEAN TEST.
    Uses all columns for training. Bounds/domains computed once from TRAIN.
    """
    feats = feature_columns(df_clean, ds_cfg)
    X = df_clean[feats].copy()
    y = coerce_target_to_int(df_clean[ds_cfg["target"]])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    bounds_meta = compute_bounds_and_domains(X_tr, ds_cfg)

    accs, f1s, precs = [], [], []
    for r in reps:
        X_tr_noisy = anonymize_df(X_tr, ds_cfg, budgets, bounds_meta, noise_seed=seed * 10_000 + r)
        pipe = make_downstream_pipeline(X_tr_noisy, ds_cfg, seed)
        pipe.fit(X_tr_noisy, y_tr)
        y_hat = pipe.predict(X_te)
        accs.append(accuracy_score(y_te, y_hat))
        f1s.append(f1_score(y_te, y_hat, average="weighted", zero_division=0))
        precs.append(precision_score(y_te, y_hat, average="weighted", zero_division=0))

    return {"accuracy": float(np.mean(accs)),
            "f1": float(np.mean(f1s)),
            "precision": float(np.mean(precs))}


# ===========================
# Info-loss (sensitive columns only)
# ===========================
def info_loss_components(df_orig: pd.DataFrame, df_noisy: pd.DataFrame,
                         ds_cfg: dict, sensitive_only=True) -> dict:
    num_losses, cat_losses = [], []
    cols = df_orig.columns
    for col in cols:
        if sensitive_only and not is_sensitive(col, ds_cfg):
            continue
        if is_categorical(col, ds_cfg):
            a = df_orig[col].to_numpy()
            b = df_noisy[col].to_numpy()
            cat_losses.append(np.mean(a != b))
        else:
            a = pd.to_numeric(df_orig[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(df_noisy[col], errors="coerce").to_numpy(dtype=float)
            rng_span = max(np.nanmax(a) - np.nanmin(a), 1e-12)
            mae = np.nanmean(np.abs(a - b))
            num_losses.append(mae / rng_span)
    parts = []
    if num_losses: parts.append(float(np.mean(num_losses)))
    if cat_losses: parts.append(float(np.mean(cat_losses)))
    composite = float(np.mean(parts)) if parts else 0.0
    return {
        "info_loss_num": float(np.mean(num_losses)) if num_losses else np.nan,
        "info_loss_cat": float(np.mean(cat_losses)) if cat_losses else np.nan,
        "info_loss": composite
    }


# ===========================
# Budget allocation helpers
# ===========================
def ensure_positive_and_normalized(budgets, total, min_eps=1e-3):
    b = np.clip(np.array(budgets, dtype=float), min_eps, None)
    b = total * b / b.sum()
    b = np.clip(b, min_eps, None)
    b = total * b / b.sum()
    return b

def das_split(k: int) -> np.ndarray:
    """
    Heuristic DAS: give half to the first sensitive attribute, split the rest equally.
    Works for any k >= 1 and gets normalized later.
    """
    if k == 1:
        return np.array([1.0])
    base = np.full(k, (0.5) / (k - 1))
    base[0] = 0.5
    return base

def ssas_split(k: int) -> np.ndarray:
    """
    Heuristic SSAS: geometric decay 1, 1/3, 1/6, ... normalized later.
    """
    vals = np.array([1.0 / max(1, i + 1) for i in range(k)], dtype=float)
    return vals


# ===========================
# GA setup (optimize per-sensitive-attr eps split)
# ===========================
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
current_n_sensitive = 3  # overwritten per dataset

def random_budget_init():
    return np.random.dirichlet(np.ones(current_n_sensitive) * 5.0).tolist()

toolbox.register("individual", tools.initIterate, creator.Individual, random_budget_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA globals
df_global = None
DS_CFG = None
TOTAL_BUDGET = None
SEED_FOR_GA = None

def setup_ga(df: pd.DataFrame, ds_cfg: dict, total_budget: float, seed_for_ga: int):
    """
    Prepare GA to evaluate per-sensitive-attribute budget splits using publish-style utility.
    """
    global df_global, DS_CFG, TOTAL_BUDGET, SEED_FOR_GA, current_n_sensitive
    df_global = df
    DS_CFG = ds_cfg
    TOTAL_BUDGET = float(total_budget)
    SEED_FOR_GA = int(seed_for_ga)
    current_n_sensitive = len(ds_cfg["sensitive_attrs"])

    # (Re)register evaluate to close over the above globals
    if "evaluate" in toolbox.__dict__:
        try:
            toolbox.unregister("evaluate")
        except Exception:
            pass

    def evaluate(individual):
        b = ensure_positive_and_normalized(individual, TOTAL_BUDGET)
        us = utility_scores_publish(df_global, DS_CFG, b, seed=SEED_FOR_GA, reps=NOISE_REPLICATES)
        return (us["accuracy"],)  # maximize accuracy (could combine metrics)

    toolbox.register("evaluate", evaluate)

def run_ga(seed, patience=10, min_delta=1e-4):
    random.seed(seed); np.random.seed(seed)
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    best = -np.inf
    stale = 0

    for gen in range(GENERATIONS):
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        # Early stop check
        gen_best = max(ind.fitness.values[0] for ind in pop)
        if gen_best > best + min_delta:
            best = gen_best
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

        # Select, clone, and vary
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        pop = offspring

    best_vec = hof[0] if len(hof) else max(pop, key=lambda ind: ind.fitness.values[0])
    return ensure_positive_and_normalized(best_vec, TOTAL_BUDGET).tolist()


# ===========================
# Per-task worker
# ===========================
def load_and_prepare(ds_cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ds_cfg["path"]).dropna().reset_index(drop=True)

    target = ds_cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' missing in {ds_cfg['name']}.")

    # if sensitive_attrs not defined or empty, infer automatically
    sens = ds_cfg.get("sensitive_attrs", [])
    if not sens:  
        sens = [c for c in df.columns if c != target]
        ds_cfg["sensitive_attrs"] = sens

    # warn if any listed categorical attrs don’t exist
    miss_cat = [c for c in ds_cfg.get("categorical_attrs", []) if c not in df.columns]
    if miss_cat:
        print(f"[WARN] Categorical attrs not found in {ds_cfg['name']}: {miss_cat}")

    return df

def eval_method(df_clean: pd.DataFrame, ds_cfg: dict, budgets: list, seed: int) -> tuple[dict, dict]:
    """
    Evaluate utility + info-loss for a given per-sensitive-attribute budget vector.
    Info-loss measured on TRAIN (the part “released for training”), sensitive cols only.
    """
    # Utility
    util = utility_scores_publish(df_clean, ds_cfg, budgets, seed, reps=NOISE_REPLICATES)

    # Info-loss on TRAIN
    feats = feature_columns(df_clean, ds_cfg)
    X = df_clean[feats].copy()
    y = coerce_target_to_int(df_clean[ds_cfg["target"]])
    X_tr, _, _, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    bounds_meta = compute_bounds_and_domains(X_tr, ds_cfg)
    X_tr_noisy = anonymize_df(X_tr, ds_cfg, budgets, bounds_meta, noise_seed=seed * 10_000 + NOISE_REPLICATES[0])
    il = info_loss_components(X_tr, X_tr_noisy, ds_cfg, sensitive_only=True)
    return util, il

def run_for_combination(args):
    """
    One job for (dataset_cfg, seed, total_eps).
    Returns a list of 3 records: GA, DAS, SSAS.
    """
    ds_cfg, seed, tot_eps = args
    recs = []
    df = load_and_prepare(ds_cfg)
    random.seed(seed); np.random.seed(seed)

    k = len(ds_cfg["sensitive_attrs"])

    # ---------- GA ----------
    setup_ga(df, ds_cfg, tot_eps, seed_for_ga=seed)
    best = run_ga(seed)
    util_ga, il_ga = eval_method(df, ds_cfg, best, seed)
    print(f"[{ds_cfg['name']}] seed {seed} – ε={tot_eps} GA: "
          f"acc {util_ga['accuracy']:.5f}, f1 {util_ga['f1']:.5f}, prec {util_ga['precision']:.5f}, "
          f"IL {il_ga['info_loss']:.5f}")
    rec_ga = {"dataset": ds_cfg["name"], "seed": seed, "budget": tot_eps, "method": "GA",
              **util_ga, **il_ga}
    for i, col in enumerate(ds_cfg["sensitive_attrs"]):
        rec_ga[f"eps_{col}"] = best[i]
    recs.append(rec_ga)

    # ---------- DAS ----------
    base_das = das_split(k)
    das = ensure_positive_and_normalized(base_das, tot_eps).tolist()
    util_das, il_das = eval_method(df, ds_cfg, das, seed)
    print(f"[{ds_cfg['name']}] seed {seed} – ε={tot_eps} DAS: "
          f"acc {util_das['accuracy']:.5f}, f1 {util_das['f1']:.5f}, prec {util_das['precision']:.5f}, "
          f"IL {il_das['info_loss']:.5f}")
    rec_das = {"dataset": ds_cfg["name"], "seed": seed, "budget": tot_eps, "method": "DAS",
               **util_das, **il_das}
    for i, col in enumerate(ds_cfg["sensitive_attrs"]):
        rec_das[f"eps_{col}"] = das[i]
    recs.append(rec_das)

    # ---------- SSAS ----------
    base_ss = ssas_split(k)
    ssas = ensure_positive_and_normalized(base_ss, tot_eps).tolist()
    util_ssas, il_ssas = eval_method(df, ds_cfg, ssas, seed)
    print(f"[{ds_cfg['name']}] seed {seed} – ε={tot_eps} SSAS: "
          f"acc {util_ssas['accuracy']:.5f}, f1 {util_ssas['f1']:.5f}, prec {util_ssas['precision']:.5f}, "
          f"IL {il_ssas['info_loss']:.5f}")
    rec_ssas = {"dataset": ds_cfg["name"], "seed": seed, "budget": tot_eps, "method": "SSAS",
                **util_ssas, **il_ssas}
    for i, col in enumerate(ds_cfg["sensitive_attrs"]):
        rec_ssas[f"eps_{col}"] = ssas[i]
    recs.append(rec_ssas)

    return recs


# ===========================
# Orchestrator
# ===========================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_records = []
    start_time = time.time()

    tasks = [(ds, seed, tot_eps) for ds in DATASETS for seed in SEEDS for tot_eps in BUDGETS]

    with Pool(processes=min(len(tasks), cpu_count())) as pool:
        for recs in pool.imap_unordered(run_for_combination, tasks):
            all_records.extend(recs)

    df_out = pd.DataFrame(all_records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = (
        f"results/exp_{timestamp}"
        f"_ldp_v1.5"
        f"_runs{RUNS}"
        f"_pop{POP_SIZE}"
        f"_gen{GENERATIONS}"
        f"_cx{int(CX_PROB*100)}"
        f"_mut{int(MUT_PROB*100)}.csv"
    )
    df_out.to_csv(fname, index=False)

    print("\nSaved per-seed results to:", fname)
    end_time = time.time()
    print(f"Done! Runtime {(end_time - start_time):.2f}s")
