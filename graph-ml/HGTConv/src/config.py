from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = PROJECT_ROOT / "data" / "threat_group_APT_.json"
    CHECKPOINT_PATH: Path = PROJECT_ROOT / "checkpoints" / "ea_hgt.pt"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

    SEED: int = 42
    TEST_RATIO: float = 0.2
    VAL_RATIO: float = 0.1

    HIDDEN_DIM: int = 128
    NUM_HEADS: int = 4
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2

    BATCH_SIZE: int = 1
    EPOCHS: int = 50
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-5

    TOP_K: int = 5
    MAX_SEQ_LEN: int = 20

    #DEVICE: str = "cuda"  
    DEVICE: str = "cpu"  
    
    NUM_TECHNIQUES: str = 20