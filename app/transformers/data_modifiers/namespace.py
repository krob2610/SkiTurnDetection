from enum import Enum

STYLE_COLUMN = "STYLE"
SKIER_LEVEL_COLUMN = "SKIER_LEVEL"
SLOPE_COLUMN = "SLOPE"


# Constants for skiing styles
class SkiStyle(str, Enum):
    SNOWPLOW = "snowplow"
    SKIDDED_SHORT = "skidded_short"
    SKIDDED_LONG = "skidded_long"
    UP_UNWEIGHTING = "up_unweighting"  # NW
    CARVING_LONG = "carving_long"
    CARVING_SHORT = "carving_short"
    QUICK = "quick"  # smig


# Constants for skier levels
class SkierLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Constants for slope difficulty
class Slope(str, Enum):
    GREEN = "green"
    BLUE = "blue"
    RED = "red"
    BLACK = "black"
