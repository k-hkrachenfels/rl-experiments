from enum import Enum, IntEnum
from pydantic import BaseModel
import yaml
from util import IndexableEnumMeta

class AgentType(str, Enum):
    Q_LEARNING = "q-learning"
    SARSA = "sarsa"

class WorldConfig(BaseModel):
    grid_rows: int
    grid_cols: int

class AgentConfig(BaseModel):
    # agent parameters with default values
    type:  AgentType
    alpha: float = 0.05
    gamma: float = 0.99
    epsilon: float = 0.5
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.01

class TrainingConfig(BaseModel):
    num_episodes: int
    max_steps_per_episode: int
    validate_interval: int

class RewardsConfig(BaseModel):
    default: int
    invalid: int

class FilesConfig(BaseModel):
    experiment_dir: str
    board: str
    output_prefix: str
    q_table_prefix: str

class Color(IntEnum, metaclass=IndexableEnumMeta):
    """The board editor can display boards and edit/change cell states of the initial board"""
    WHITE = 0
    BLACK = 1
    BLUE = 2
    GREEN = 3
    RED = 4
    GRAY = 5

class GridState(IntEnum, metaclass=IndexableEnumMeta):
    """Mapping of a cell state to a color of the board editor"""
    FREE = Color.WHITE # a free cell
    WALL = Color.BLACK # a cell that can not be reached, agent just cant advance to the cell
    INVALID = Color.BLUE # a cell with maximal penalty. Epoch ends when going to this cell
    START = Color.GREEN # the start state, where the agent starts each epoch
    TARGET = Color.RED # the target state
    VISITED = Color.GRAY # a marker (to display a path), that tells us that the cell was visited

class Config(BaseModel):
    world: WorldConfig
    agent: AgentConfig
    training: TrainingConfig
    rewards: RewardsConfig
    files: FilesConfig


def load_config(config_file='config.yaml') -> Config:
    """instantiates Config from yaml file"""
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)
    except FileNotFoundError:
        print(f"Fehler: Konfigurationsdatei '{config_file}' nicht gefunden.")
        return None
    except yaml.YAMLError as e:
        print(f"Fehler beim Parsen der Konfigurationsdatei: {e}")
        return None
  