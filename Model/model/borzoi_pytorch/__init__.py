import sys
from pathlib import Path
current = Path(__file__)
sys.path.append(str(current.parent.parent))


from borzoi_pytorch.pytorch_borzoi_model import Borzoi
# from borzoi_pytorch.gene_utils import Transcriptome
