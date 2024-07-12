import enum
PATH_TO_TRAIN_DATA = 'data/Train/'
class Shape(enum.Enum):
    UNKNOWN = 0
    CIRCLE  = 1
    TRIANGLE= 2
    RECTANGULE = 3
    OCTAGON = 4
class Color(enum.Enum):
    UNKNOWN = 0
    RED = 1
    BLUE = 2
    OTHER = 3
IMAGE_SHAPE = [Shape.CIRCLE, Shape.TRIANGLE, Shape.RECTANGULE, Shape.OCTAGON, Shape.UNKNOWN]
IMAGE_COLOR = [Color.RED, Color.BLUE, Color.OTHER, Color.UNKNOWN]

LABEL_SHAPE = [[0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42],
               [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
               [12],
               [14],
               [i for i in range(43)]]
LABEL_COLOR = [[0,1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
               [33,34,35,36,37,38,39,40],
               [6,12,32,41,42],
               [i for i in range(43)]]
DICT_SHAPE = dict(zip(IMAGE_SHAPE, LABEL_SHAPE))
DICT_COLOR = dict(zip(IMAGE_COLOR, LABEL_COLOR))