
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

SEED = 42

CHUNK_SIZE = 100_000
GA_SAMPLE_FRACTION = 0.10
MAX_GA_ROWS = 100_000

POPULATION_SIZE = 10
GENERATIONS = 8

ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.02

MIN_FEATURES = 5

WEIGHT_F1 = 0.90
WEIGHT_REDUCTION = 0.10

N_ESTIMATORS = 30
MAX_DEPTH = 16

rng = np.random.default_rng(SEED)

fitness_cache = {}


# ============================================================
# LOAD FEATURE SCHEMA
# ============================================================

def load_features():

    schema_path = DATA_DIR / "feature_schema.json"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return schema["features"]


# ============================================================
# LOAD TRAINING SAMPLE
# ============================================================

def load_training_sample(features):

    train_path = DATA_DIR / "train.csv.gz"

    selected_columns = features + ["target"]

    samples = []

    print("\nLoading GA training sample...")

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            train_path,
            chunksize=CHUNK_SIZE,
            usecols=selected_columns
        )
    ):

        sample = chunk.sample(
            frac=GA_SAMPLE_FRACTION,
            random_state=SEED + chunk_index
        )

        samples.append(sample)

    df = pd.concat(
        samples,
        ignore_index=True
    )

    if len(df) > MAX_GA_ROWS:

        df = df.sample(
            n=MAX_GA_ROWS,
            random_state=SEED
        )

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df["target"].astype("int8")

    print(f"GA dataset size: {len(X):,}")
    print(f"Feature count: {len(features)}")

    print("\nBinary class distribution:")
    print(y.value_counts().sort_index())

    return X.reset_index(drop=True), y.reset_index(drop=True)


# ============================================================
# INTERNAL GROUP-AWARE SPLIT
# ============================================================

def create_internal_split(X, y):

    print("\nCreating internal validation split...")

    # Identical feature vectors are kept in the
    # same partition to reduce duplicate leakage.
    groups = pd.util.hash_pandas_object(
        X,
        index=False
    ).to_numpy(dtype=np.uint64)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    if train_groups.intersection(val_groups):
        raise RuntimeError("Duplicate group leakage detected!")

    print(f"Internal training records: {len(X_train):,}")
    print(f"Internal validation records: {len(X_val):,}")

    print("\nInternal training classes:")
    print(y_train.value_counts().sort_index())

    print("\nInternal validation classes:")
    print(y_val.value_counts().sort_index())

    return X_train, X_val, y_train, y_val


# ============================================================
# BUILD FITNESS CLASSIFIER
# ============================================================

def build_fitness_model():

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "classifier",
            RandomForestClassifier(

                n_estimators=N_ESTIMATORS,

                max_depth=MAX_DEPTH,

                min_samples_leaf=2,

                max_features="sqrt",

                class_weight="balanced_subsample",

                random_state=SEED,

                n_jobs=-1

            )
        )

    ])


# ============================================================
# CHROMOSOME REPAIR
# ============================================================

def repair_chromosome(chromosome):

    chromosome = np.asarray(
        chromosome,
        dtype=np.uint8
    ).copy()

    selected = int(chromosome.sum())

    if selected < MIN_FEATURES:

        missing = MIN_FEATURES - selected

        available = np.flatnonzero(chromosome == 0)

        chosen = rng.choice(
            available,
            size=missing,
            replace=False
        )

        chromosome[chosen] = 1

    return chromosome


# ============================================================
# INITIAL POPULATION
# ============================================================

def initialize_population(n_features):

    population = []

    # Include the full-feature baseline as a candidate.
    population.append(
        np.ones(n_features, dtype=np.uint8)
    )

    for _ in range(POPULATION_SIZE - 1):

        chromosome = rng.integers(
            0,
            2,
            size=n_features,
            dtype=np.uint8
        )

        chromosome = repair_chromosome(chromosome)

        population.append(chromosome)

    return population


# ============================================================
# FITNESS FUNCTION
# ============================================================

def calculate_fitness(
    chromosome,
    X_train,
    X_val,
    y_train,
    y_val
):

    selected_indices = np.flatnonzero(chromosome)

    selected_count = len(selected_indices)

    cache_key = tuple(
        selected_indices.tolist()
    )

    # Avoid training the same chromosome twice.
    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    X_train_selected = X_train.iloc[
        :, selected_indices
    ]

    X_val_selected = X_val.iloc[
        :, selected_indices
    ]

    model = build_fitness_model()

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_val_selected
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    reduction = (
        1 - selected_count / X_train.shape[1]
    )

    fitness = (
        WEIGHT_F1 * macro_f1
        + WEIGHT_REDUCTION * reduction
    )

    result = {
        "fitness": float(fitness),
        "macro_f1": float(macro_f1),
        "selected_features": int(selected_count),
        "reduction": float(reduction)
    }

    fitness_cache[cache_key] = result

    return result


# ============================================================
# TOURNAMENT SELECTION
# ============================================================

def tournament_selection(population, scores):

    candidates = rng.choice(
        len(population),
        size=TOURNAMENT_SIZE,
        replace=False
    )

    winner = max(
        candidates,
        key=lambda idx: scores[idx]["fitness"]
    )

    return population[winner].copy()


# ============================================================
# SINGLE-POINT CROSSOVER
# ============================================================

def crossover(parent1, parent2):

    if rng.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = rng.integers(
        1,
        len(parent1)
    )

    child1 = np.concatenate([
        parent1[:point],
        parent2[point:]
    ])

    child2 = np.concatenate([
        parent2[:point],
        parent1[point:]
    ])

    return (
        repair_chromosome(child1),
        repair_chromosome(child2)
    )


# ============================================================
# BIT-FLIP MUTATION
# ============================================================

def mutate(chromosome):

    mutated = chromosome.copy()

    mutation_mask = (
        rng.random(len(mutated)) < MUTATION_RATE
    )

    mutated[mutation_mask] = (
        1 - mutated[mutation_mask]
    )

    return repair_chromosome(mutated)


# ============================================================
# RUN GENETIC ALGORITHM
# ============================================================

def run_ga(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names
):

    n_features = len(feature_names)

    population = initialize_population(n_features)

    history = []

    best_chromosome = None
    best_result = None

    print("\nStarting Genetic Algorithm...")
    print("=" * 65)

    for generation in range(GENERATIONS):

        generation_start = time.perf_counter()

        scores = [

            calculate_fitness(
                chromosome,
                X_train,
                X_val,
                y_train,
                y_val
            )

            for chromosome in population

        ]

        ranking = sorted(
            range(len(population)),
            key=lambda idx: scores[idx]["fitness"],
            reverse=True
        )

        generation_best_idx = ranking[0]

        generation_best = scores[generation_best_idx]

        if (
            best_result is None
            or generation_best["fitness"] > best_result["fitness"]
        ):

            best_result = generation_best.copy()

            best_chromosome = population[
                generation_best_idx
            ].copy()

        history.append({

            "Generation": generation + 1,

            "Best_Fitness": generation_best["fitness"],

            "Best_Macro_F1": generation_best["macro_f1"],

            "Selected_Features": generation_best["selected_features"],

            "Feature_Reduction_Percentage":
                generation_best["reduction"] * 100,

            "Mean_Fitness": np.mean([
                score["fitness"] for score in scores
            ])

        })

        elapsed = (
            time.perf_counter() - generation_start
        )

        print(
            f"Generation {generation + 1:02d}/{GENERATIONS} | "
            f"Fitness: {generation_best['fitness']:.5f} | "
            f"Macro F1: {generation_best['macro_f1']:.5f} | "
            f"Features: {generation_best['selected_features']}/{n_features} | "
            f"Time: {elapsed:.1f}s"
        )

        # Elitism
        new_population = [
            population[idx].copy()
            for idx in ranking[:ELITE_SIZE]
        ]

        # Create new generation
        while len(new_population) < POPULATION_SIZE:

            parent1 = tournament_selection(
                population,
                scores
            )

            parent2 = tournament_selection(
                population,
                scores
            )

            child1, child2 = crossover(
                parent1,
                parent2
            )

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_chromosome, best_result, history


# ============================================================
# MAIN
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 65)
    print("EVOLUTIONARY FEATURE SELECTION")
    print("=" * 65)

    start_time = time.perf_counter()

    features = load_features()

    X, y = load_training_sample(features)

    X_train, X_val, y_train, y_val = create_internal_split(
        X,
        y
    )

    best_chromosome, best_result, history = run_ga(
        X_train,
        X_val,
        y_train,
        y_val,
        features
    )

    selected_features = [
        features[idx]
        for idx in np.flatnonzero(best_chromosome)
    ]

    total_time = time.perf_counter() - start_time

    reduction = (
        1 - len(selected_features) / len(features)
    ) * 100

    # Save selected feature information.
    result = {

        "algorithm": "Genetic Algorithm",

        "selected_features": selected_features,

        "original_feature_count": len(features),

        "selected_feature_count": len(selected_features),

        "feature_reduction_percentage": reduction,

        "best_internal_macro_f1": best_result["macro_f1"],

        "best_fitness": best_result["fitness"],

        "fitness_formula":
            "0.9 * Macro_F1 + 0.1 * Feature_Reduction",

        "population_size": POPULATION_SIZE,

        "generations": GENERATIONS,

        "random_state": SEED,

        "internal_training_rows": len(X_train),

        "internal_validation_rows": len(X_val),

        "training_sample_rows": len(X),

        "duplicate_grouping":
            "Feature-row hashes with StratifiedGroupKFold",

        "total_execution_time_seconds": total_time

    }

    output_json = MODEL_DIR / "ga_selected_features.json"

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    # Save generation-wise fitness history.
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        REPORT_DIR / "ga_fitness_history.csv",
        index=False
    )

    # Save selected feature list.
    selected_df = pd.DataFrame({

        "Feature": selected_features

    })

    selected_df.to_csv(
        REPORT_DIR / "ga_selected_features.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("GENETIC ALGORITHM COMPLETED")
    print("=" * 65)

    print(f"Original features: {len(features)}")

    print(f"Selected features: {len(selected_features)}")

    print(f"Feature reduction: {reduction:.2f}%")

    print(
        f"Internal macro F1: "
        f"{best_result['macro_f1']:.5f}"
    )

    print(
        f"Best fitness: "
        f"{best_result['fitness']:.5f}"
    )

    print(f"Execution time: {total_time:.2f} seconds")

    print("\nSelected features:")

    for feature in selected_features:
        print(" -", feature)

    print("\nSaved:")
    print(output_json)
    print(REPORT_DIR / "ga_fitness_history.csv")
    print(REPORT_DIR / "ga_selected_features.csv")


if __name__ == "__main__":
    main()
