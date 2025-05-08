"""
Constant values shared across the Brain‑For‑Printing codebase.
"""
from typing import List, Dict 

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

# --- Surface Type Constants (Used in CLI) ---
CORTICAL_TYPES: List[str] = ["pial", "white", "mid", "inflated"]
HEMI_CORTICAL_TYPES: List[str] = [f"{h}-{t}" for h in ["lh", "rh"] for t in CORTICAL_TYPES]
# Combined list for parser choices
CORTICAL_CHOICES: List[str] = CORTICAL_TYPES + HEMI_CORTICAL_TYPES

# ASEG-derived structures (Used in CLI)
CBM_BS_CC_CHOICES: List[str] = [
    "brainstem", "cerebellum_wm", "cerebellum_cortex",
    "cerebellum", "corpus_callosum"
]
# Alias often used for fill/smooth options
FILL_SMOOTH_CHOICES: List[str] = CBM_BS_CC_CHOICES

# Keywords for VTK structure requests (Used in CLI)
VTK_KEYWORDS: List[str] = ["all"]

# Example names for VTK-derived subcortical structures (Used in CLI help)
SGM_NAME_EXAMPLES: List[str] = [
    'L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp', 'L_Pall', 'L_Puta', 'L_Thal',
    'R_Accu', 'R_Amyg', 'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 'R_Thal'
]

# Example names for VTK-derived ventricular/vessel structures (Used in CLI help)
VENT_NAME_EXAMPLES: List[str] = [
    '3rd-Ventricle', '4th-Ventricle',
    'Left-Inf-Lat-Vent', 'Left_LatVent_ChorPlex',
    'Right-Inf-Lat-Vent', 'Right_LatVent_ChorPlex',
    'Left-vessel', 'Right-vessel'
]

