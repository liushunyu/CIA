from .q_learner import QLearner
from .cia_q_learner import CIAQLearner
from .qplex_learner import QPlexLearner
from .cia_qplex_learner import CIAQPlexLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["cia_q_learner"] = CIAQLearner
REGISTRY["qplex_learner"] = QPlexLearner
REGISTRY["cia_qplex_learner"] = CIAQPlexLearner
