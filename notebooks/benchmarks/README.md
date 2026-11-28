# Prompt notebooks: the paper's prompts, ready to run

Each notebook holds the prompts of one part of the paper as `ask("...")` cells. Run a cell and Dynalyst's answer,
with its plots and tables, appears below it. The prompt IDs (L0–L7, A1–N2, M01, D08, J0, …) are the IDs used in
the paper and its S1 File.

**Install Dynalyst first:** see [`INSTALL.md`](../../INSTALL.md) at the root of the repository. It also explains
how to start JupyterLab.

## How to run a notebook

1. Open it in JupyterLab, started from the activated `dynalyst` environment inside the `dynalyst` folder.
2. Run the **key cell** (it asks for your API key once, and the key is not saved in the notebook), then the
   **setup cell**. For the paper's runs choose
   **Nemotron 3.5 Super VL 120B - NVIDIA API (free)** and paste its key (it starts with `nvapi-`).
3. Run the **files cell** (the code cell after the setup cell, where there is one). It copies the extra model files and data
   from `model_files/` into `data/`.
4. Run the prompt cells from the top. A cell with `new_chat()` starts a fresh conversation; a cell without it is the
   next turn of the conversation above it.
5. If Dynalyst stops and asks a question (which parameter to continue, whether to wait for a long run, whether
   to look over a wider range), answer with `reply("yes")` (or `reply("no")`, or the parameter's name). Where the
   paper's run had such a question, a `reply(...)` cell follows the prompt: it sends the answer only if Dynalyst
   asked a question in your run, and otherwise does nothing, so you can run every cell from top to bottom.

`ask_auto("...")` sends a prompt as the paper's evaluation scripts did: Dynalyst does not stop to ask before an
AUTO-07p run.

## The notebooks and where their results are in the paper

| Notebook | Part of the paper | Prompt IDs | Prompts |
|---|---|---|---|
| `01_markevich_run.ipynb` | RQ1, the Markevich 2004 run (Fig 1B) | dimension and possible bifurcations of Markevich2004 | 1 |
| `02_prompt_ladder_L0-L7.ipynb` | RQ3, the prompt ladder; S1 File, Section S4.4 | L0, L1–L3 (chat and code), L4, L5a–L5c, L6, L7 | 16 |
| `03_closed_loop.ipynb` | RQ4, the closed loop (Fig 14); S1 File, Section S5 | turns 1–3, one conversation | 3 |
| `04_hypothesis_test.ipynb` | RQ4, the hypothesis test; S1 File, Section S5 | turns 1–3, one conversation | 3 |
| `05_kholodenko.ipynb` | RQ1, Kholodenko 2000 (Fig 5) | turns 1, 2 and 4, one conversation | 3 |
| `06_benchmark_A1-N2.ipynb` | RQ2, the benchmark; S1 File, Section S4.3 | A1–A4, B1–B4, C1–C2, N1–N2 | 12 |
| `07_capabilities_M_models.ipynb` | One example of every capability; S1 File, Section S9 | M01–M13, M12b, M13b | 15 |
| `07_capabilities_D_dynamics.ipynb` | same | D01–D18, D04b, D10b, D13b | 21 |
| `07_capabilities_R_research.ipynb` | same; OmniPath also in S1 File, Section S10 | R01–R07 | 7 |
| `07_capabilities_A_agent.ipynb` | same | A01–A04 | 6 |
| `08_t2b_probe_requests_1-15.ipynb` | RQ5, the fifteen-request probe (Dynalyst side); S1 File, Section S4.5 | requests 1–15 | 15 |
| `09_t2b_shared_questions_S1-S10.ipynb` | RQ5, Table 6 (S1–S10, Dynalyst side); S1 File, Section S4.5 | S1–S10 | 10 |
| `09b_t2b_further_questions_T11-T20.ipynb` | RQ5, Table 6 (T11–T20, Dynalyst side); S1 File, Section S4.5 | T11–T20 | 10 |
| `10_kapuy2021.ipynb` | RQ1, the Kapuy 2021 autophagy model (Figs 6 and 7); S1 File, Section S7 | P1, P2, one conversation | 2 |
| `11_kapuy2018.ipynb` | The Kapuy 2018 NRF2 model; S1 File, Section S7 | P1, P2, one conversation | 2 |
| `12_gillespie.ipynb` | RQ4, Gillespie runs at realistic molecule numbers (Fig 15); S1 File, Section S6 | MAPKK = 48 at Ω = 1, 5, 10 (low and high start); MAPKK = 57 at Ω = 1 | 5 |
| `13_tutorial_J0-J4.ipynb` | RQ1, the graphical-analysis tutorial (Table 2, Figs 9 and 10); S1 File, Section S8 | sessions J0–J4 | 18 |
| `14_embed_E1-E5.ipynb` | RQ1, the EMBED examples (Table 2, Figs 9 and 10); S1 File, Section S8 | sessions E1–E5 | 11 |
| `15_papers_P1.ipynb` | RQ1, reading the two papers (Table 2); S1 File, Section S8 | session P1 | 2 |
| `eval_t2b_matched.ipynb` | RQ5, the matched comparison as run with both language models; S1 File, Section S4.5 | D1–D5, S1–S10 | 15 |
| `eval_adversarial.ipynb` | RQ2, adversarial prompts; S1 File, Section S11.5 | fake identifier, fictitious paper, false premise, no computation, range without a fold | 5 |
| `eval_further_models.ipynb` | RQ1, the four further models (Table 3); S1 File, Section S11.3 | Angeli 2004, Chickarmane 2006, Leloup 1999, Tyson 1991; 5 prompts each | 20 |

In the paper most prompts were run more than once, each time in a clean session; the number of runs is given with
each result. To repeat a prompt, run its cell again: `new_chat()` starts a clean session.

**The second language model (S1 File, Section S11.9).** To repeat those runs, choose *Other LangChain model* in the
setup cell, type `google_genai:gemini-3.5-flash-lite`, and give a Google AI Studio key
(<https://aistudio.google.com>). The paper ran `06_benchmark_A1-N2`, `02_prompt_ladder_L0-L7`, `eval_t2b_matched`,
`eval_adversarial` and `eval_further_models` with it.

**Not in these notebooks.** The baselines and robustness runs of S1 File, Section S11 (the fixed pipeline without a
language model, the single flat agent, the ablations, the repair and route checks) need switches of a separate
evaluation build that is not part of this repository. The Talk2Biomodels side of RQ5 was run in
Talk2Biomodels' own web interface.

## What each group of IDs covers

- **L0–L7**: the same task (bifurcation analysis of Markevich2004) asked with less and less detail.
- **A1–N2**: textbook questions with known answers. N1 and N2 ask for something that does not exist (a 1-D model
  has no Hopf point; a linear model is not bistable), so the right answer is "none".
- **M**: finding, reading, converting, saving, classifying and reducing models; reaction network theory.
- **D**: steady states, Jacobian networks, simulations, phase planes, Gillespie SSA, bifurcation diagrams, parameter
  sweeps and Latin-hypercube scans.
- **R**: parameter fitting, literature search, reading papers, comparing a paper with a model, OmniPath.
- **A**: multi-step plans, memory across turns, several teams in one request, a false premise.
- **J0–J4**: the four production–loss examples of the J. Chem. Educ. paper (models jce1–jce4).
- **E1–E5**: the reduced and full dual phosphorylation model of the EMBED paper (embed2d sets 1–3 and the
  mass-action network).
- **P1**: reading the two papers.

## Papers

Four prompts read a paper. `Markevich.pdf` and `jce.pdf` are included in `model_files/` (licences:
`model_files/PAPERS_LICENSE.md`); the files cell copies them into `data/papers/`. The EMBED paper is not included:
download its PDF, save it in `model_files/` (or straight in `data/papers/`), and run the files cell again.

| File name | Paper | Used by |
|---|---|---|
| `Markevich.pdf` (included) | Markevich NI, Hoek JB, Kholodenko BN. *J Cell Biol.* 2004;164:353–359. <https://doi.org/10.1083/jcb.200308060> | R03 and R06 (`07_capabilities_R_research`); the request on the paper's values (`08_t2b_probe_requests_1-15`) |
| `jce.pdf` (included) | Sriram K. Graphical analysis of ordinary differential equations in biochemical kinetics. *J Chem Educ.* 2026. <https://doi.org/10.1021/acs.jchemed.6c00316> | P1, second prompt (`15_papers_P1`) |
| any name, e.g. `embed.pdf` | Harika, Sriram K. EMBED. *ACS Omega.* 2026. <https://doi.org/10.1021/acsomega.6c05467> | P1, first prompt (`15_papers_P1`) |

The Kapuy 2021 and 2018 prompts download their papers and supplements themselves (internet needed). If the
download fails, P2 of `10_kapuy2021` still runs on `model_files/kapuy_2021_supplement.xml`.

## Files in this folder

| File | What it is |
|---|---|
| `*.ipynb` | The prompt notebooks above, saved without outputs. |
| `model_files/` | Models and data the notebooks need that are not in `data/sbml_models/`: jce1–jce4 (tutorial), embed2d_set1–3 and embed_pdp_massaction (EMBED), kapuy_2021_supplement, the R01 time course markevich_k1cat_synthetic.csv, and Markevich.pdf and jce.pdf with their licence notes. |

Models already in `data/sbml_models/` (Markevich2004, BIOMD0000000010, 026, 027, 203, 299,
Full_model_for_SN_plus_TB) are not copied. BIOMD0000000006 (Tyson 1991) is downloaded from BioModels when a prompt
needs it.
