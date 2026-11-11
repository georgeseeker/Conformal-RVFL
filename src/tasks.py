"""The 25 multiclass UCI tasks used by this project (single source of truth).

Same IDs, names and order as the executed task list of commit 654bdd2 (FGC multiclass list minus
Flags, LED Display and Low Resolution Spectrometer) and as the 25 tasks in results/conformal.csv."""
MULTICLASS_TASKS = (
    (1, "Abalone"),
    (3, "Annealing"),
    (193, "Cardiotocography_10classes"),
    (19, "Car Evaluation"),
    (26, "Connect-4"),
    (33, "Dermatology"),
    (39, "Ecoli"),
    (242, "Energy Efficiency"),
    (42, "Glass Identification"),
    (44, "Hayes-Roth"),
    (45, "Heart_cleveland"),           # native five-class target
    (63, "lymphography"),
    (69, "Molecular Biology"),
    (76, "Nursery"),
    (80, "Optical Recognition of Handwritten Digits"),
    (91, "Soybean"),
    (148, "Statlog (Shuttle)"),
    (146, "Statlog (Landsat Satellite)"),
    (149, "Statlog (Vehicle Silhouettes)"),
    (212, "Vertebral Column"),
    (107, "Waveform"),
    (110, "Yeast"),
    (109, "Wine"),
    (186, "Wine Quality"),
    (111, "Zoo"),
)
GROUP = "multiclass"                 # value of the `group` column in results CSVs
TASK_KEYS = frozenset((GROUP, u, n) for u, n in MULTICLASS_TASKS)
assert len(MULTICLASS_TASKS) == 25 and len(TASK_KEYS) == 25
