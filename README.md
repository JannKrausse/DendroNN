This is the official repository which can be used for reproducing the results presented in "DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data". There, we present a novel event-based neural network that computes by identifying unique spatiotemporal sequences of spikes. Training is split into two phases: a rewiring phase to train the DendroNN layers followed by a supervised phase to train the output layer via backpropagation. When infered on custom digital accelerators, DendroNNs show tremendous benefits in terms of energy efficiency which is a testimony to their high degree of static and dynamic sparsity. Please refer to the publication for more information.

Necessary requirements can be installed with
```bash
pip install -r requirements.txt
```

Then, training of DendroNNs is performed via the train.py. In there, simply specify DATASET and the respective hyperparameters before running
```bash
python3 train.py
```
Results, including checkpoints, tensorboard logs, and confusion matrices will be logged in logs/DATASET/EXPERIMENT_NAME.

In order to successfully run the NeuroMorse experiments, please make sure to clone the following repository into the neuromorse_data/ directory:
https:// github.com/Ben-E-Walters/NeuroMorse.

Finally, if you use DendroNNs or our code in your work, please cite our publication:
```latex
@article{krausse2026dendronn,
  title={DendroNN: Dendrocentric Neural Networks for Energy-Efficient Classification of Event-Based Data},
  author={Krausse, Jann and Su, Zhe and Mama, Kyrus and Knobloch, Klaus and Indiveri, Giacomo and Becker, J{\"u}rgen and others},
  journal={arXiv preprint arXiv:2603.09274},
  year={2026}
}
```