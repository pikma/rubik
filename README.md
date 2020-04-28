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


## References

- Agostinelli, F., McAleer, S., Shmakov, A. et al. Solving the Rubik’s cube with
  deep reinforcement learning and search. Nat Mach Intell 1, 356–363 (2019).
  https://doi.org/10.1038/s42256-019-0070-z
    * DeepCubeA.
    * DNN learns value function using TD(0)
    * more complex network: 2 fully connected, then 4 residual blocks.
    * cube represented as a one-hot vector of size 6 for each sticker (54
      stickers total).
    * uses A* for search (with a tweak).

- McAleer, Stephen, et al. "Solving the Rubik's Cube with Approximate Policy
  Iteration." (2018). https://openreview.net/forum?id=Hyfn2jCcKm
    * DeepCube
    * DNN with both a value head (trained using TD(0) and policy head)
    * training set sampled from scrambling the solved cube, each example is
      weighted by 1/{distance to solved}.
    * simple model (sequence of fully connected layers)
    * MCTS to search
    * trick: use the same network weights for the labels, only update them when
      the loss gets below a threshold.
    * cube represented as binary 20x24 array

### Not read yet
- Brunetto, R. & Trunda, O. Deep heuristic-learning in the Rubik’s cube domain:
  an experimental evaluation. Proc. ITAT 1885, 57–64 (2017).
- Johnson, C. G. Solving the Rubik’s cube with learned guidance functions. In
  Proceedings of 2018 IEEE Symposium Series on Computational Intelligence (SSCI)
  2082–2089 (IEEE, 2018).
