# Phase-3 generalist submission

`main.py` implements `agent(obs_dict)`: returns the 60-card `deck.csv` at deck selection,
then plays via the trained generalist policy (`policy.pt`, from phase3-a checkpoint-0120)
using the `ptcg` inference package and the competition `cg` module. Assembled by
`scripts/make_submission.py` into `dist/submission/`. Upload the CONTENTS of
`dist/submission/` (or `dist/submission.zip`) as the Kaggle agent.
