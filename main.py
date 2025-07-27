from taxi_q_learning_epsilon.taxi_agent import TaxiAgent as TaxiQAgentEpsilon
from taxi_q_learning_softmax.taxi_agent import TaxiAgent as TaxiQAgentSoftmax
from taxi_sarsa_epsilon.taxi_agent import TaxiAgent as TaxiSarsaEpsilon
from taxi_sarsa_softmax.taxi_agent import TaxiAgent as TaxiSarsaSoftmax
from taxi_deep_q_learning_epsilon.taxi_agent import TaxiAgent as TaxiDeepQAgentEpsilon
from taxi_deep_q_learning_softmax.taxi_agent import TaxiAgent as TaxiDeepQAgentSoftmax


print("\nRUNNING Q-LEARNING SOFTMAX AGENT")
agent2 = TaxiQAgentSoftmax()
agent2.train()
agent2.play()

print("\nRUNNING Q-LEARNING EPSILON AGENT")
agent1 = TaxiQAgentEpsilon()
agent1.train()
agent1.play()

print("\nRUNNING DEEP Q-LEARNING EPSILON AGENT")
agent5 = TaxiDeepQAgentEpsilon()
agent5.train()
agent5.play()






