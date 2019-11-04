
from enum import Enum, unique

@unique
class ClipType(Enum):
    ratio = 0
    ratio_rollback = 1

    ratio_strict = 2
    ratio_rollback_constant = 3

    a2c = 5

    kl = 20
    kl_strict = 21
    kl_ratiorollback = 22
    kl_klrollback_constant = 23
    kl_klrollback = 24

    kl2clip = 31
    kl2clip_rollback = 32


    adaptivekl = 40

    adaptiverange_advantage = 50


MUJOCO = 'mujoco'
ATARI = 'atari'
