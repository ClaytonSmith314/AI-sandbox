

# this is the location of all main data structures used in the program

awake = True;
# ----------------------- #
# Passing Data Structures #
# ----------------------- #

Objectives_q = []  # note, they could be dictionaries
Objectives = {}

Context_q = []
Context = {}

Results_q = []
Results = {}

Situation_q = []
Situation = {}


# ---------------------- #
# Memory Data Structures #
# ---------------------- #

Info = {}  # the set of all information produced in the system

heap = {}  # the set of action groups that can be employed for given signals

rules = {}  # the set of desired results associated with certain negative signals


# --------------------------- #
#  secondary data structures  #
# --------------------------- #

situation_queue = {}

