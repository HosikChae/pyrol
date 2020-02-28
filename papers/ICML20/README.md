# [Deep Reinforcement Learning with Linear Quadratic Regulator Regions](https://arxiv.org/abs/2002.09820)
## Gabriel I. Fernandez, Colin Togashi, Dennis W. Hong, Lin F. Yang
### Abstract
Practitioners often rely on compute-intensive domain randomization to ensure reinforcement learning policies trained in simulation can robustly transfer to the real world. Due to unmodeled nonlinearities in the real system, however, even such simulated policies can still fail to perform stably enough to acquire experience in real environments. In this paper we propose a novel method that guarantees a stable region of attraction for the output of a policy trained in simulation, even for highly nonlinear systems. Our core technique is to use "bias-shifted" neural networks for constructing the controller and training the network in the simulator. The modified neural networks not only capture the nonlinearities of the system but also provably preserve linearity in a certain region of the state space and thus can be tuned to resemble a linear quadratic regulator that is known to be stable for the real system. We have tested our new method by transferring simulated policies for a swing-up inverted pendulum to real systems and demonstrated its efficacy.

### What is this sub-directory?
The files here were submitted as supplementary code to be reviewed for ICML 2020. This directory contains a self-contained package that can be run to test our methods and results in the paper. Please notify us if there are any bugs.

### How to cite?
```text
@misc{fern2020deep,
    title={Deep Reinforcement Learning with Linear Quadratic Regulator Regions},
    author={Gabriel I. Fernandez and Colin Togashi and Dennis W. Hong and Lin F. Yang},
    year={2020},
    eprint={2002.09820},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```