# Memorization Circuits

This repository contains code for circuit analysis experiments with VLMs for tasks where memorization from the model results in wrong answers.

# Repository Structure

- `third_party/`: Contains a fork of the TransformerLens library with edits made to make it support VLMs.
- `src/eap`: Contains code for the circuit discovery method adapted from EAP-IG.
  - `attribute_node.py`: Primitives for nodes/edges used in the attribution graph.
  - `attribute.py`: EAP/EAP-IG attribution routines and hooks to compute per-node attributions.
  - `evaluate.py`: Scripts and utilities to run evaluations and compute metrics on circuits.
  - `graph.py`: Builds and manipulates the computation/attention graph for circuit discovery.
  - `utils.py`: Shared helper functions (I/O, tensor utilities, typing helpers).
  - `visualization.py`: Plotting tools to visualize graphs, saliency maps, and circuits.

# Acknowledgements

- We use the TransformerLens fork implemented [here](https://github.com/technion-cs-nlp/vlm-circuits-analysis)
- The circuit discovery method is heavily borrowed from [EAP-IG](https://github.com/hannamw/eap-ig)
- The dataset we use comes from [Vo et al. (2025)](https://huggingface.co/datasets/anvo25/vlms-are-biased)
