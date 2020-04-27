# rubik
Learning how to solve a Rubik's cube using Reinforcement Learning


## Status

Model is learning something.  I tried tweaking the structure of the model but could not get to a loss below 18, which seems quite high.

It's good enough to solve
cubes scrambled with 5 rotations with just a 1-depth greedy search.


Ideas of next steps:
 - investigate the model's behavior more:
    * more metrics than the loss (e.g. average L1 error)
    * slice metrics by the label: are we better at cubes closer or further from
      a solved state?
 - implement MCTS.
 - change the labels of the model so that it looks more like TD learning (the
   label is the prediction in the next state)
 - change the training set so that the labels are the true distance to the
   solved state (generate the labels using a DFS from the solved state).
