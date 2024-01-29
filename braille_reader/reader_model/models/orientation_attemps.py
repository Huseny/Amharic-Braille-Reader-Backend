import enum

class OrientationAttempts(enum.IntEnum):
    NONE = 0
    ROT180 = 1
    INV = 2
    INV_ROT180 = 3
    ROT90 = 4
    ROT270 = 5
    INV_ROT90 = 6
    INV_ROT270 = 7