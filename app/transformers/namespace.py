STYLE_COLUMN = "STYLE"
SKIER_LEVEL_COLUMN = "SKIER_LEVEL"
SLOPE_COLUMN = "SLOPE"


# Constants for skiing styles
class SkiStyle(str, Enum):
    CARVING = "carving"
    RACING = "racing"
    FREESTYLE = "freestyle"
    ALL_MOUNTAIN = "all_mountain"
    POWDER = "powder"


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
