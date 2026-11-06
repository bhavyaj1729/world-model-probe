# Do Transformers Learn Internal World Models?

A small grid-world experiment with a tiny transformer.

## The problem

I trained a small transformer to predict the next move of an agent walking on a 5×5 grid. I never told it where the agent was. A model can predict moves well without really tracking position, so the question is: does it work out the agent's position on its own?

## The result

Yes, it seems to. A simple linear probe can read the agent's cell from the model's hidden states with **97% accuracy at layer 2** (guessing the most common cell gives about 5%). An untrained model of the same size gets only about 20%, and probes trained on shuffled labels stay at the 5% baseline. Walks that end in the same cell also get more similar hidden states as you go deeper.

One catch: this shows the position information is *there*, not that the model actually *uses* it. I did not run a causal test.

## What I did

- Made 30,000 random walks of 20 moves each. The agent only makes legal moves.
- Trained a 3-layer transformer (hidden size 64, 2 heads) to predict the next move. It sees only the moves, never the positions.
- Trained one linear probe per layer to guess the agent's cell from the hidden states.
- Compared with two controls: an untrained model, and shuffled position labels.
- Compared hidden states of walks that end in the same cell vs different cells.

Probes were trained on 2,000 walks and tested on 3,000 held-out walks.

## Results

Next-move prediction: validation loss 1.186, close to the best possible (about 1.19). The model's top predicted move never leaves the grid.

Position probe accuracy by layer:

| Layer | Trained model | Untrained model | Shuffled labels |
|---|---|---|---|
| Embedding | 0.096 | 0.096 | 0.048 |
| Layer 1 | 0.473 | 0.181 | 0.050 |
| Layer 2 | 0.971 | 0.195 | 0.058 |
| Layer 3 | 0.929 | 0.198 | 0.052 |

(Trained and shuffled used a 1000-step probe, untrained used 500 steps.)

![Probe accuracy, trained vs untrained](results/probe_trained_vs_random.png)

Similarity gap (same final cell minus different final cell, last token):

| Layer | Trained | Untrained |
|---|---|---|
| Embedding | 0.07 | 0.07 |
| Layer 1 | 0.14 | 0.07 |
| Layer 2 | 0.37 | 0.07 |
| Layer 3 | 0.44 | 0.07 |

![Similarity gap](results/similarity_gap.png)

Walks ending in the same cell reach a similarity of 0.57, not 1.0, so the model also keeps some of the path.

All exact numbers are in `results/numbers.json`. The full write-up is in `report.pdf`.

## Limits

- No causal test, so readable does not mean used.
- One run, one grid size, no obstacles. The training seed was not fixed, so numbers will change a little between runs.
- Layer 3 is a bit lower than layer 2 (0.929 vs 0.971). A longer probe narrowed the gap, but I did not find out if the rest is real.
- Probes were trained on only 2,000 walks.

## Repo structure

```
environment/gridworld.py        grid-world, walk generation, splits
model/transformer.py            the small transformer (returns hidden states per layer)
experiments/                    Colab notebook: training, probes, similarity
results/                        plots and numbers.json
report.pdf                      short write-up
```

## How to run

1. Open the notebook in `experiments/` in Google Colab and pick a GPU runtime.
2. Mount Google Drive and set `ROOT` to your project folder (the notebook assumes `MyDrive/world_model_probe`).
3. Run the cells in order: data, training, probes, controls, similarity.

Needs `torch`, `numpy` and `matplotlib`.

## Next steps

- Test whether the model uses the position, by editing the position direction and watching its predictions.
- Repeat with several seeds.
- Try obstacles and bigger grids.
