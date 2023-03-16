# Automaton-Guided-CL
Github repo for the ICAPS 2023 paper: Automaton-guided Curriculum Generation for Reinforcement Learning Agents

Overview of the paper:

Despite advances in Reinforcement Learning, many sequential decision making tasks remain prohibitively expensive and impractical to learn. Recently, approaches that automatically generate reward functions from logical task specifications have been proposed to mitigate this issue; however, they scale poorly on long-horizon sequential decision making tasks. We propose AGCL, Automaton-guided Curriculum Learning, a novel method for automatically generating curricula for the target task in the form of Directed Acyclic Graphs (DAGs). AGCL encodes the specification in the form of a deterministic finite automaton (DFA), and then uses the DFA along with the Object-Oriented MDP (OOMDP) to generate a curriculum as a DAG, where the vertices correspond to tasks, and edges correspond to the direction of knowledge transfer. Experiments in gridworld and physics-based simulated robotics domains show that the curricula produced by AGCL achieve improved time-to-threshold learning performance on a complex sequential decision-making problem relative to state-of-the-art curriculum learning (e.g, teacher-student, self-play) and automaton-guided reinforcement learning baselines (e.g, Q-Learning for Reward Machines). Further, we demonstrate that AGCL performs well even in the presence of noise in the OOMDP description of the task, and also when distractor objects are present that are not modeled in the logical specification of the tasks' objectives.

The requirements are listed in the file: requirements.txt

To install: pip install -r requirements.txt

The experiments were conducted using a 64-bit Linux Machine, having Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz processor and 126GB RAM memory. 
The maximum duration for running the experiments was set at 24 hours.

Terminology for Source Code:
MC - grid-world domain (Minecraft-like domain)
Noisy - grid-world domain with noisy OOMDP
distractor - grid-world domain with distractor objects

To generate the automaton guided curriculum in the gridworld environment, run:
$ python generate_curriculum.py
(change parameters on #5-6 to vary between sequence-based and graph-based curricula)
(OOMDP parameters can also be changed on line 10-11)

The curriculum given by generate_curriculum is reflected in the environments.py file. This file is present in all the folders, that defines the OOMDP initial state parameters for the curriculum.

The paper demonstrates results from 10 trails. The experiments are conducted on seeds 1-10

To test AGCL-sequence on MC, run:
$ python MC/AGCL-sequence/main.py

To test AGCL-graph on MC, run:
$ python MC/AGCL-graph/main.py

Replace MC with Noisy/Distractor to test AGCL on Noisy and Distractor settings.

Running the above programs would generate a log file that stores the number of timesteps, rewards, episodes and other information from the curriculum run and the learning from scratch run. After conducting 10 trials. 
