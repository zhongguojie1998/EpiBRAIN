import os

PWD = f'{os.environ["workingHOME"]}/BICAN'
OUTPUT_DIR = os.path.join(PWD, 'Analysis/07_eQTL_new/output')

BIOTYPE_TO_KEEP = [
    'protein_coding', 'lncRNA', 'IG_V_gene', 'TR_V_gene', 'IG_C_gene',
    'snoRNA', 'snRNA', 'TR_C_gene', 'miRNA',
]

ORGAN_TISSUE_MAP = {
    'Brain': [
        'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24',
        'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere',
        'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
        'Brain_Hippocampus', 'Brain_Hypothalamus',
        'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia',
        'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
    ],
    'Basal_ganglia': [
        'Brain_Caudate_basal_ganglia',
        'Brain_Nucleus_accumbens_basal_ganglia',
        'Brain_Putamen_basal_ganglia',
    ],
    'Cortex': [
        'Brain_Anterior_cingulate_cortex_BA24',
        'Brain_Cortex',
        'Brain_Frontal_Cortex_BA9',
    ],
}

VARIANT_GROUPS = ['all', '<3k', '3k-12k', '12k-35k', '>35k']

# ---------------------------------------------------------------------------
# Track filter regex patterns (shared by borzoi / alphagenome)
# ---------------------------------------------------------------------------
BRAIN_TRACK_PATTERN = (
    r'brain|cerebr|hippocamp|amygdal|'
    r'frontal.*(lobe|gyrus|cortex|area)|'
    r'neuron|astrocyte|oligodendrocyte|microglia|cerebellum|cerebellar|thalamus|'
    r'parietal.*(lobe|cortex)|temporal.*(lobe|gyrus)|occipital|'
    r'putamen|caudate|substantia|nucleus accumbens|cingulate|'
    r'spinal cord|neurosphere|globus pallidus|medulla oblongata|pons'
)

BASAL_GANGLIA_TRACK_PATTERN = (
    r'putamen|caudate|nucleus accumbens|globus pallidus|striatum|basal.ganglia'
)

CORTEX_TRACK_PATTERN = (
    r'cerebral cortex|frontal cortex|occipital cortex|parietal cortex|'
    r'frontal.*(?:lobe|gyrus)|parietal lobe|temporal.*(?:lobe|gyrus)|occipital.*(?:lobe|pole)|'
    r'cingulate|prefrontal'
)

GTEX_BRAIN_TRACK_PATTERN = r'^RNA:brain'

EMBRYO_FETAL_TRACK_PATTERN = r'embryo|fetal|fetus|embryonic|prenatal|newborn'

DISEASE_TRACK_PATTERN = (
    r'disease|disorder|syndrome|cancer|tumor|tumour|carcinoma|leukemia|lymphoma|'
    r'alzheimer|parkinson|huntington|autism|schizophrenia|epilepsy|atrophy|injury|'
    r'stroke|glioma|glioblastoma'
)

REGION_FILTER_MAP = {
    'brain': BRAIN_TRACK_PATTERN,
    'basal_ganglia': BASAL_GANGLIA_TRACK_PATTERN,
    'cortex': CORTEX_TRACK_PATTERN,
    'gtex_brain': GTEX_BRAIN_TRACK_PATTERN,
}
