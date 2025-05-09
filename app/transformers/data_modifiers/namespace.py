from enum import Enum

STYLE_COLUMN = "STYLE"
SKIER_LEVEL_COLUMN = "SKIER_LEVEL"
SLOPE_COLUMN = "SLOPE"


# Constants for skiing styles
class StringEnum(str, Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class SkiStyle(StringEnum):
    SNOWPLOW = "snowplow"
    SKIDDED_SHORT = "skidded_short"
    SKIDDED_LONG = "skidded_long"
    CARVING_LONG = "carving_long"
    CARVING_SHORT = "carving_short"
    QUICK = "quick"  # smig


# Constants for skier levels
class SkierLevel(StringEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Constants for slope difficulty
class Slope(StringEnum):
    GREEN = "green"
    BLUE = "blue"
    RED = "red"
    BLACK = "black"
