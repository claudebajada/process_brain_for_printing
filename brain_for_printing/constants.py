"""
Constant values shared across the Brain‑For‑Printing codebase.
"""

BRAINSTEM_LABELS: list[int] = [
    2, 3, 24, 31, 41, 42, 63, 72, 77, 51, 52, 13, 12,
    43, 50, 4, 11, 26, 58, 49, 10, 17, 18, 53, 54,
    44, 5, 80, 14, 15, 30, 62
]

# Standard FreeSurfer aseg labels for specific structures of interest
BRAINSTEM_LABEL: list[int] = [16]
CEREBELLUM_CORTEX_LABELS: list[int] = [8, 47]
CEREBELLUM_WM_LABELS: list[int] = [7, 46]
CORPUS_CALLOSUM_LABELS: list[int] = [251, 252, 253, 254, 255]
CEREBELLUM_LABELS: list[int] = CEREBELLUM_CORTEX_LABELS + CEREBELLUM_WM_LABELS

