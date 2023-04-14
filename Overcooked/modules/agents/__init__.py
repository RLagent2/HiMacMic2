REGISTRY = {}

from .rnn_agent import RNNAgent

from .tsg_rnn_agent import TSGRNNAgent

from .tsg_ns_rnn_agent import TSGNSRNNAgent

#level0:
REGISTRY["rnn"] = RNNAgent
#level1:
REGISTRY["tsg_rnn"] = TSGRNNAgent
REGISTRY["tsg_ns_rnn"] = TSGNSRNNAgent