This repo contains the implementation for [SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition](https://arxiv.org/abs/2106.03849) and its predecessor [Monet](MONet: Unsupervised Scene Decomposition and Representation).

The codebase uses Jax framework, with Neural Network module using Treex and Elegy for the training framework.

- Run the train script in either `monet/` or `simone` folder.
- An evaluation notebook where Object-representation and Frame-representation are composed can be found in `simone/` folder.

Note: This codebase is for research purpose and does not follow the best engineering practices. It runs on TPU-VM but things will probably breaks, depends on your setup. 

### Acknowledgement
This is made possible thanks to [TPU Research Cloud](https://sites.research.google/trc/about/) for their generous support of TPUs quota.
