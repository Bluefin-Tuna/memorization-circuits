# Circuit Reuse Experiments

Code for circuit reuse experiments.

Core idea:

1. Collect prompt / answer pairs for a simple task.
2. Attribute component importance (attention heads / MLP blocks).
3. Find overlap (shared circuit).
4. Ablate it and measure accuracy drop.

## Quick Start

```bash
pip install -r requirements.txt
python run_experiment.py --model_name gpt2 --task addition --num_examples 50 --method gradient
```

## Attribution Methods

- gradient (fast first-order)
- ig (integrated gradients, slower, more stable)

## References / Inspiration

- TransformerLens (hooks + instrumentation): https://github.com/TransformerLensOrg/TransformerLens
- circuit-stability: https://github.com/alansun17904/circuit-stability
- eap-ig: https://github.com/hannamw/eap-ig
- Datasets referenced: MMLU, Mechanistic Interpretability Benchmark (HuggingFace Hub)

## License

MIT. Inspired by (but not dependent on) the above open-source repos.
