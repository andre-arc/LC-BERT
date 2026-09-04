# TODO

Open work on the `fix/whitening-orientation` branch, in priority order.

**Context.** Three defects were fixed that between them dominated the published results:
the Bi-LSTM head applied `nn.Softmax` before `CrossEntropyLoss` and `nn.ReLU` to the logits,
text preprocessing was silently disabled after publication, and several whitening kernels were
mis-oriented or unrunnable. Current numbers are in the README's *Current results* section
(F1 0.9130 ± 0.0008 for `svd`, against a published 0.8344). The published table itself
reproduces exactly at commit `c42c5a7`.

Every item below is verified against the tree, not recalled.

---

## 1. Measurement — needed before a paper revision

- [ ] **Multi-seed the no-reduction control.** `ag-news-bert-extraction` + `bilstm` has only a
      single seed (F1 0.9002, seed 88), yet it is the entire basis for the claim that whitened
      256-dim features beat unreduced 768-dim ones. Every competing number now carries ±std.
      ~135 min.
      ```bash
      python train_multiple_seeds.py --n_epochs 5 --train_batch_size 32 --valid_batch_size 32 \
        --model_name bilstm --dataset ag-news-bert-extraction --experiment_name ms-raw768 \
        --lr 2e-3 --eps 1e-8 --step_size 1 --gamma 0.9 --early_stop 3 --lower --force
      ```
      Blocked by item 7 (`KeyError: 'dim_technique'` in that script).

- [ ] **Separate whitening from truncation.** The current comparison is whitened-256 (0.9130)
      against *unwhitened*-768 (0.9002), which cannot tell the two apart. A linear probe
      suggests truncation alone slightly *costs* — whitened-768 0.9020 vs whitened-256 0.8951 —
      meaning the gain comes from whitening, not from reducing dimensions. Confirming that needs
      whitened-768 through the Bi-LSTM. Requires item 10.

- [ ] **Re-run RoBERTa (B04–B06) and all MLP rows (C01–C06).** Only the BERT + Bi-LSTM slice has
      been redone. `MLPForWordClassification` never had the broken loss — its `Softmax` is
      commented out and its dropout is correctly placed — so the paper's Scenario 2 vs Scenario 3
      comparison was confounded on one side only. This is the most likely of these gaps to change
      a published conclusion. 8 configs.

- [ ] **Re-run the efficiency curves (paper Figs. 2–3).** `efficient_analysis.py` still fits a
      per-split kernel and uses `optim.Adam` where every other script uses `AdamW`.

- [ ] **Re-measure timing on the paper's hardware.** Published wall times assume an RTX 3050
      4 GB; the current machine is a GTX 1660 Ti 6 GB. Accuracy and peak-memory figures transfer,
      wall-clock does not.

## 2. Code

- [ ] **Remove `_fix_signs`** (`data_utils/ag_news/whitening.py`). Pinning `diag(U) > 0` follows
      Kessy et al. §5, but at d=768 the median `|U[j,j]|` is 0.025, so the sign is decided by a
      noise-dominated quantity. It is a no-op under the default shared kernel and costs ~0.06 F1
      in per-split mode, with no measured benefit anywhere.

- [ ] **Propagate the shared kernel.** `main.py` and `train_multiple_seeds.py` fit the whitening
      kernel on train and reuse it via `whitening_params`; `kfold_analysis.py`,
      `efficient_analysis.py` and `low_resource_analysis.py` still fit per split, so their
      results are not comparable to anything else. Copy the `shared_whitening` block from
      `main.py::setup_dataloaders`.

- [ ] **Fix `KeyError: 'dim_technique'` in the three remaining scripts.** `kfold_analysis.py`,
      `low_resource_analysis.py` and `train_multiple_seeds.py` still index `args['dim_technique']`
      unconditionally on the extraction path, but `append_dataset_args` never sets it for
      `ag-news-{bert,roberta}-extraction`. Those datasets therefore cannot run there. One word:
      `args.get('dim_technique')`, as already applied in `main.py`. This is why every extraction
      line in `windows_scripts/run_task_modified.bat` is commented out.

- [ ] **Replace `DataFrame.append` with `pd.concat`** in `main.py`, `train_multiple_seeds.py`,
      `kfold_analysis.py` and `efficient_analysis.py`. Removed in pandas 2.0; latent only because
      `environment.yml` pins 1.4.4.

- [ ] **Cache embeddings across seeds.** `train_multiple_seeds.py` calls `setup_dataloaders`
      inside the seed loop, re-extracting embeddings for every seed. BERT is frozen and run under
      `no_grad`, so at `--subset_percentage 100` all seeds produce byte-identical vectors — about
      60 of every 135 minutes is wasted. `BertWhiteningDataset.save_array` exists, unused, for
      exactly this.

- [ ] **Add `--target_dim`.** 256 is hardcoded in `BertWhiteningDataset.Dim_reduction`, so
      sweeping `k` currently means editing source. Needed for item 2, and for reporting a cost
      curve instead of citing Su et al.'s d/3 heuristic — which does not transfer here: a linear
      probe over k = 16…768 is monotonically increasing, with no peak at 256.

- [ ] **Consider collapsing `svd` and `pca`.** They are the same transform computed two ways
      (multi-seed F1 0.9130 vs 0.9129, inside 1σ; bit-identical probe scores). Su et al. §3.3 say
      so themselves — their method is "equivalent to Principal Component Analysis theoretically".
      Keeping both dataset names invites reporting them as distinct methods, which the published
      Table III does.

## 3. Documentation

- [ ] **`EPSILON_USAGE.md`** documents the whitening epsilon as `1e-5`; the code and the paper
      both use `1e-8`. Flagged in the README but not rewritten. Note `--eps` is a different
      quantity — the AdamW epsilon.

- [ ] **`windows_scripts/README.md`** says `run_task_modified.bat` calls `main.py`. It calls
      `train_multiple_seeds.py`, as does `run_task_benchmark.bat`. `run_task_modified_kfold.bat`
      calls `kfold_analysis.py` and `run_efficiency_analysis.bat` calls `efficient_analysis.py`.
