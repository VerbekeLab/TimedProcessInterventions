# Timed Process Interventions: Causal Inference vs. Reinforcement Learning </br><sub><sub>Hans Weytjens, Jochen De Weerdt, Wouter Verbeke [[2023]](https://doi.org/10.1007/978-3-031-50974-2_19)</sub></sub>
The shift from the understanding and prediction of processes to their optimization offers great benefits to businesses and other organizations. Precisely timed process interventions are the cornerstones of effective optimization. Prescriptive process monitoring (PresPM) is the sub-field of process mining that concentrates on process optimization. The emerging PresPM literature identifies state-of-the-art methods, causal inference (CI) and reinforcement learning (RL), without presenting a quantitative comparison. Most experiments are carried out using historical data, causing problems with the accuracy of the methods' evaluations and preempting online RL. Our contribution consists of experiments on timed process interventions with synthetic data that renders genuine online RL and the comparison to CI possible, and allows for an accurate evaluation of the results. Our experiments reveal that RL's policies outperform those from CI and are more robust at the same time. Indeed, the RL policies approach perfect policies. Unlike CI, the unaltered online RL approach can be applied to other, more generic PresPM problems such as next best activity recommendations. Nonetheless, CI has its merits in settings where online learning is not an option.

## Citing
Please cite our paper and/or code as follows:

```tex

@inproceedings{weytjens2023,
  title={Timed Process Interventions: Causal Inference vs. Reinforcement Learning},
  author={Weytjens, Hans and Verbeke, Wouter and De Weerdt, Jochen},
  booktitle={International Conference on Business Process Management},
  pages={245--258},
  year={2023},
  organization={Springer}
}


```