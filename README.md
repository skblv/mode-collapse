# mode-collapse

Code for "Fine-Tuning Fixes Mode Collapse and Over-Dispersion in LLMs":
[paper](https://arxiv.org/abs/2609.16454),
[project page](https://skobelev.me/mode-collapse),
[models](https://huggingface.co/collections/skblv/diversity-calibration-lora-adapters).

The code compares the diversity of a model's outputs with the diversity of a target distribution. The measure is the collision ratio `R = C(q) / C(p)`: the probability that two model outputs collide, divided by the same probability under the target. `R > 1` is mode collapse, `R < 1` is over-dispersion, and `R = 1` is calibrated diversity.

The experiments are small transformers trained on two synthetic languages with exactly known targets, LoRA fine-tuning of four LLMs on survey responses (GSS, WVS and ANES), and LoRA fine-tuning of the same LLMs on accepted Python solutions from Project CodeNet.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Training, measurement and generation need a CUDA GPU. Figures, tests and CodeNet scoring run on a CPU.

## Figures

```bash
uv run python experiments/figures/make_figures.py
uv run python experiments/figures/figure1/make_figure1.py
```

writes Figures 1 to 4 and the appendix bound check to `outputs/figures/` from the tables in `results/`. Figure 1 needs `pdflatex` with TikZ. Add `--results-dir outputs` to plot a new run.

## Experiments

Run the scripts from the repository root, in the order of their numbers. Inputs are read from `data/` and outputs are written to `outputs/`; set `MODE_COLLAPSE_DATA_DIR` and `MODE_COLLAPSE_OUTPUT_DIR` to use other locations. The Gemma models are gated on Hugging Face: accept their licenses and run `hf auth login` first.

`--model` takes `google/gemma-2-2b-it`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B` or `google/gemma-3-4b-it`. To fine-tune on several GPUs, start `s02_finetune.py` (surveys) or `s03_finetune.py` (CodeNet) with `uv run torchrun --standalone --nproc-per-node <gpus>` instead of `uv run python`; the effective batch size stays the same.

### Synthetic languages

```bash
uv run python experiments/exp1_synthetic/s01_diffuse_grid.py
uv run python experiments/exp1_synthetic/s01_diffuse_grid.py --reduce
uv run python experiments/exp1_synthetic/s02_aligned_grid.py
uv run python experiments/exp1_synthetic/s02_aligned_grid.py --reduce
uv run python experiments/exp1_synthetic/s03_bound_sweep.py
uv run python experiments/exp1_synthetic/s04_initial_alignment.py
```

No input data are needed. `s01` and `s02` train 100 models at each of 100 sample sizes, under a diffuse and a mode-aligned starting distribution, and `--reduce` writes the decomposition of the collision gap into variance, squared bias and cross term. `s03` checks the bound on the collision gap, and `s04` computes the decomposition before fine-tuning. `--level-start` and `--level-count` run a subset of the sample sizes, and `--smoke` runs a small configuration of `s01` to `s03`. The tables are written to `outputs/synthetic/`.

### Surveys

Download the survey files from their providers and place them in `data/surveys/`:

| Survey | File |
| --- | --- |
| [General Social Survey](https://gss.norc.org) | `gss7224_r3.dta` |
| [World Values Survey](https://www.worldvaluessurvey.org), wave 7 | `WVS_Cross-National_Wave_7_csv_v6_0.csv` |
| [American National Election Studies](https://electionstudies.org), cumulative file | `anes_timeseries_cdf_csv_20260205.csv` |

Fine-tuning and measurement log to Weights & Biases: set `WANDB_ENTITY` and `WANDB_PROJECT`, and `WANDB_MODE=offline` to keep the logs local.

```bash
RUN=outputs/surveys/checkpoints/gss_gemma-2-2b-it_n8_seed42

uv run python experiments/exp2_surveys/s01_build_targets.py --dataset gss
uv run python experiments/exp2_surveys/s02_finetune.py --dataset gss \
    --model google/gemma-2-2b-it --n 8
uv run python experiments/exp2_surveys/s03_measure.py --dataset gss \
    --model google/gemma-2-2b-it
uv run python experiments/exp2_surveys/s03_measure.py --dataset gss \
    --model google/gemma-2-2b-it --n 8 \
    --checkpoint early_stopped --adapter-dir $RUN/early_stopped
uv run python experiments/exp2_surveys/s03_measure.py --dataset gss \
    --model google/gemma-2-2b-it --n 8 \
    --checkpoint fully_trained --adapter-dir $RUN/fully_trained
uv run python experiments/exp2_surveys/s04_aggregate.py
```

`s01` builds the answer distribution of every demographic group, `s02` fine-tunes on `n` respondent answers, and `s03` reads the answer distribution of the base model or of an adapter for each group. A full run covers `gss`, `wvs` and `anes`, the four models, and every sample size: the training-pool size, halved repeatedly while it is at least 8. The pool size is `n_train_rows` in the summary that `s01` writes to `outputs/surveys/<survey>/`. `s04` needs all three surveys and all four models, and writes `outputs/surveys/survey_collision_ratio.csv`.

### CodeNet

```bash
ARCHIVE=data/codenet/Project_CodeNet.tar.gz
URL=https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
RUN=outputs/codenet/checkpoints/codenet_gemma-2-2b-it_n12_seed42

mkdir -p data/codenet
curl --location --output $ARCHIVE $URL
tar -xzf $ARCHIVE -C data/codenet --wildcards \
    "Project_CodeNet/metadata/*.csv" "Project_CodeNet/problem_descriptions/*"

uv run python experiments/exp3_codenet/s01_prepare_accepted_python.py \
    --member-list outputs/codenet/accepted_python_members.txt \
    --summary outputs/codenet/accepted_python_summary.json
uv run python experiments/exp3_codenet/s02_build_solution_split.py
uv run python experiments/exp3_codenet/s03_finetune.py \
    --model google/gemma-2-2b-it --n 12
for SHARD in 0 1 2 3 4 5 6 7; do
    uv run python experiments/exp3_codenet/s04_generate.py \
        --model google/gemma-2-2b-it --num-shards 8 --shard-index $SHARD
    uv run python experiments/exp3_codenet/s04_generate.py \
        --model google/gemma-2-2b-it --num-shards 8 --shard-index $SHARD \
        --n 12 --checkpoint fully_trained --adapter-dir $RUN/fully_trained
done
uv run python experiments/exp3_codenet/s05_score_zhang_shasha.py --human
uv run python experiments/exp3_codenet/s05_score_zhang_shasha.py \
    --model gemma-2-2b-it --n 0 --checkpoint base
uv run python experiments/exp3_codenet/s05_score_zhang_shasha.py \
    --model gemma-2-2b-it --n 12 --checkpoint fully_trained
uv run python experiments/exp3_codenet/s06_summarize.py
```

`s01` lists the accepted Python solutions, `s02` splits them into training and validation sets, `s03` fine-tunes on the first `n` training solutions, `s04` samples 32 programs per problem, and `s05` scores pairs of programs with a normalized Zhang–Shasha similarity between their syntax trees. A full run covers the four models, both checkpoints (`early_stopped` and `fully_trained`), and every size in `training_grid` of `outputs/codenet/processed/summary.json`. `s06` needs all of those arms, and writes `outputs/codenet/codenet_collision_ratio.csv`.

An interrupted fine-tuning run resumes from its last checkpoint when the same command is run again. Finished generation shards and scoring caches are reused.

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Layout

- `src/mode_collapse/synthetic/`: the two synthetic languages, the small transformer, and the decomposition and bound computations.
- `src/mode_collapse/surveys/`: survey questions and prompts, target construction, fine-tuning, measurement and aggregation.
- `src/mode_collapse/codenet/`: the solution split, canonical syntax trees, fine-tuning, generation and Zhang–Shasha scoring.
- `experiments/`: the numbered scripts of each experiment and the figure scripts.
- `results/`: the tables behind the figures.
- `tests/`: CPU tests with small fixtures.

## Citation

```bibtex
@misc{skobelev2026finetuningfixesmodecollapse,
      title={Fine-Tuning Fixes Mode Collapse and Over-Dispersion in LLMs},
      author={Kirill Skobelev and Eric Fithian and X. Y. Han},
      year={2026},
      eprint={2609.16454},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.16454},
}
```
