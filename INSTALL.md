# Installing Dynalyst and reproducing the paper's prompts

This one file covers everything: the tools your computer needs, the Dynalyst environment, the
AUTO-07p build, the language-model key and the prompt notebooks in `notebooks/benchmarks/`.
Copy each command block into a terminal and run it. When a block starts with a **check**, run the check first: if
it prints a version or a path, skip the install line under it.

A first install takes 15–30 minutes, because PyTorch, JAX and the AUTO-07p build are large. On a remote server, run it
inside `tmux` or `screen` so that a dropped connection does not stop it.

| Your system | Go to |
|---|---|
| Ubuntu, Debian, Linux Mint (with `sudo`) | [Part A1](#a1-ubuntu-debian-linux-mint) |
| Red Hat (RHEL), Rocky, AlmaLinux, CentOS Stream, Fedora (with `sudo`) | [Part A2](#a2-red-hat-rhel-rocky-almalinux-centos-stream-fedora) |
| Any Linux server **without** `sudo` (shared lab or university server) | [Part A3](#a3-any-linux-server-without-sudo) |
| macOS (Apple silicon or Intel) | [Part B](#part-b-macos) |
| Windows 10 or 11 | [Part C](#part-c-windows) |

After Part A, B or C, every system continues with **Part D** (key, check, notebooks).

What you need on every system:

- an internet connection (for the install, and for prompts that use BioModels, PubMed or OmniPath);
- about 8 GB of free disk space;
- an API key for a language model. The paper's runs used **Nemotron 3.5 Super** through NVIDIA's API, which has a
  free key: sign in at <https://build.nvidia.com>, open any model page and choose *Get API Key*. The key starts
  with `nvapi-`. Keep it private: never paste it into a notebook cell that you save or share, or into a file that
  you commit.

---

## Part A. Linux

### A1. Ubuntu, Debian, Linux Mint

**1. System tools** (git, unzip, the Fortran compiler for AUTO-07p)

```bash
sudo apt-get update
sudo apt-get install -y git unzip wget build-essential gfortran
```

**2. conda** (check first)

```bash
conda --version
```

Not found? Install Miniconda in your home folder:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

Then go to [A4](#a4-the-dynalyst-environment-every-linux).

### A2. Red Hat (RHEL), Rocky, AlmaLinux, CentOS Stream, Fedora

**1. System tools**

```bash
sudo dnf install -y git unzip wget gcc gcc-c++ gcc-gfortran make
```

(On older systems without `dnf`, use `sudo yum install -y git unzip wget gcc gcc-c++ gcc-gfortran make`.)

**2. conda** (check first)

```bash
conda --version
```

Not found? Install Miniconda in your home folder:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

Then go to [A4](#a4-the-dynalyst-environment-every-linux).

### A3. Any Linux server without sudo

Everything goes into your home folder.

**1. conda** (check first)

```bash
conda --version
```

Not found?

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

**2. git and unzip** (check first)

```bash
git --version && unzip -v | head -1
```

Not found?

```bash
conda install -y -c conda-forge git unzip
```

**3. gfortran** (check first)

```bash
gfortran --version
```

Not found? Put it in a small conda environment of its own and add it to the PATH of **this** terminal. Run A4 in
the same terminal:

```bash
conda create -y -n fortran -c conda-forge gfortran
export PATH="$(conda info --base)/envs/fortran/bin:$PATH"
```

### A4. The Dynalyst environment (every Linux)

```bash
git clone https://github.com/dynalyst26/dynalyst.git
cd dynalyst
bash setup_env.sh
```

`setup_env.sh` creates the conda environment `dynalyst` (Python 3.11) and installs PyTorch, JAX 0.4.35,
jaxkineticmodel (from `vendor/`), the packages in `requirements.txt`, crnt4sbml and Dynalyst itself. It then
builds AUTO-07p and runs a few import checks. If `conda create` stops and asks you to accept Anaconda's terms of
service, run the `conda tos accept ...` lines it prints and run `bash setup_env.sh` again.

Then add the notebook interface and the Gemini, Anthropic and Ollama connectors:

```bash
conda activate dynalyst
pip install -e ".[notebook,providers]"
```

Go to [Part D](#part-d-every-system-key-check-and-notebooks).

---

## Part B. macOS

**1. Apple's command-line tools** (git, compilers; check first)

```bash
xcode-select -p
```

Error? Install them, and finish the window that opens before you go on:

```bash
xcode-select --install
```

**2. conda** (check first)

```bash
conda --version
```

Not found? Install Miniconda. The command picks the Apple-silicon (`arm64`) or Intel (`x86_64`) installer by
itself:

```bash
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$(uname -m).sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
conda init zsh
```

**3. gfortran** (check first)

```bash
gfortran --version
```

Not found? Put it in a small conda environment of its own and add it to the PATH of **this** terminal. Run step 4
in the same terminal:

```bash
conda create -y -n fortran -c conda-forge gfortran
export PATH="$(conda info --base)/envs/fortran/bin:$PATH"
```

(With Homebrew, `brew install gcc` also works.)

**4. The Dynalyst environment**

```bash
git clone https://github.com/dynalyst26/dynalyst.git
cd dynalyst
bash setup_env.sh
conda activate dynalyst
pip install -e ".[notebook,providers]"
```

Go to [Part D](#part-d-every-system-key-check-and-notebooks).

---

## Part C. Windows

Windows runs Dynalyst natively, in an *Anaconda Prompt*. (If you already use WSL with Ubuntu, you can follow
[Part A1](#a1-ubuntu-debian-linux-mint) inside WSL instead.)

**1. git** (in PowerShell; check first)

```powershell
git --version
```

Not recognized?

```powershell
winget install -e --id Git.Git
```

**2. conda.** Look for **Anaconda Prompt** or **Miniconda Prompt** in the Start menu. If neither is there, install
Miniconda (in PowerShell), then close PowerShell:

```powershell
winget install -e --id Anaconda.Miniconda3
```

**3. Everything below runs in Anaconda Prompt.** Open it from the Start menu, go to the folder where you keep
projects, and get the code:

```bat
cd %USERPROFILE%
git clone https://github.com/dynalyst26/dynalyst.git
cd dynalyst
```

**4. The environment and gfortran**

```bat
conda create -y -n dynalyst python=3.11
conda activate dynalyst
conda install -y -c conda-forge m2w64-gcc-fortran
conda env config vars set CUDA_ROOT=%CONDA_PREFIX%
conda deactivate
conda activate dynalyst
```

**5. The packages and the AUTO-07p build** (the same steps `setup_env.sh` runs on Linux and macOS)

```bat
python -m zipfile -e vendor\jaxkineticmodel.zip vendor
pip install "torch>=2.6"
pip install "jax==0.4.35"
pip install -e vendor\jaxkineticmodel
pip install -r requirements.txt
pip install --no-deps "crnt4sbml==0.0.15"
pip install -e ".[notebook,providers]"
python scripts\build_auto.py
```

If `build_auto.py` reports that gfortran is missing, install it with
`winget install BrechtSanders.WinLibs.POSIX.UCRT`, open a new Anaconda Prompt, run `conda activate dynalyst` and
`cd %USERPROFILE%\dynalyst`, and run `python scripts\build_auto.py` again.

Go to Part D. In Part D, type `\` instead of `/` in paths where Windows asks for it.

---

## Part D. Every system: key, check and notebooks

All commands run in the `dynalyst` folder with the environment active (`conda activate dynalyst`).

**1. Check the install**

```bash
python scripts/build_auto.py --check
```

It must report two folds, at p = -0.3849 and p = 0.3849. To run the test suite as well (no key is needed; the
command below leaves out the slow AUTO tests and takes a few minutes; `python -m pytest -q` runs everything, 1–2 hours):

```bash
python -m pytest -q -m "not auto"
```

**2. (Recommended) Build the knowledge-base index.** The paper's runs had it. Without it every tool still works,
but the agent cannot search the jaxkineticmodel code, its documentation and the saved model descriptions. On a
computer without an NVIDIA GPU this takes 10–30 minutes:

```bash
python -m dynalyst.rag.build_index --chunks data/chunks_code.jsonl data/chunks_docs.jsonl data/chunks_models.jsonl --device cpu
```

(With an NVIDIA GPU, use `--device cuda`.)

**3. (Optional) Give the key once per terminal.** The setup cell of each notebook asks for the key in a password
box, which is not saved in the notebook. If the key is already set as below, leave that box empty.

Linux and macOS (this terminal only):

```bash
export NVIDIA_API_KEY="nvapi-..."
```

Windows Anaconda Prompt (this window only):

```bat
set NVIDIA_API_KEY=nvapi-...
```

**4. Start JupyterLab**

On your own computer:

```bash
jupyter lab
```

On a remote server, start it without a browser:

```bash
jupyter lab --no-browser --port 8888
```

and on your laptop open a tunnel, then open the `http://localhost:8888/lab?token=...` link that the server printed:

```bash
ssh -L 8888:localhost:8888 your_user@your_server
```

**5. Run the prompts.** In JupyterLab open `notebooks/benchmarks/`. `README.md` in that folder lists the notebooks,
which prompts each holds, and where each result is in the paper. In each notebook:

1. Run the **key cell** (the first code cell). It asks for the key once; press Enter to skip it if you set
   `NVIDIA_API_KEY` in step 3. The key is kept only in this session, never in the notebook.
2. Run the **setup cell**. Choose **Nemotron 3.5 Super VL 120B - NVIDIA API (free)** for the paper's runs, and
   leave its key box empty (the key from the key cell is used). Then run the **files cell**, where there is one: it
   copies the extra model files from `notebooks/benchmarks/model_files/` into `data/`.
3. Run the prompt cells one by one, from the top. A cell with `new_chat()` starts a fresh conversation; a cell
   without it is the next turn of the conversation above it.
4. If Dynalyst stops and asks a question (for example which parameter to continue), answer with
   `reply("yes")`. Where the paper's run had such a question, a `reply(...)` cell follows the prompt. It sends
   the answer only if Dynalyst asked in your run, so you can run all the cells from top to bottom.

Each answer, with its plots and tables, appears under the cell. The full record of each run (every tool call
and result) is saved in `outputs/runs/`.

---

## Every later session

```bash
cd dynalyst                  # Windows: cd %USERPROFILE%\dynalyst
conda activate dynalyst
jupyter lab                  # on a server: jupyter lab --no-browser --port 8888
```

To get the latest version of the code and the notebooks:

```bash
git pull
pip install -e ".[notebook,providers]"
```

---

## What to expect when you rerun the prompts

Language models do not give the same text twice, even with the same settings, so a rerun gives the same
analyses and numbers in different words. The numbers computed by the tools (fold and Hopf points, steady states,
periods) agree with the paper's to the precision it reports. Choices the language model makes can differ between
runs, as the paper reports for its repeated runs (for example the continuation range of an open question). The
paper's own answers, with every tool call, are in its run records (Data availability section of the paper).

The exact package versions of the paper's Linux runs are in `env/requirements-lock.txt`. If a newer release of a
package breaks something, install those versions in the `dynalyst` environment:

```bash
pip install -r env/requirements-lock.txt
```

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `conda: command not found` right after installing Miniconda | Close the terminal and open a new one (or run `source ~/.bashrc`; on macOS `source ~/.zshrc`). |
| `conda create` asks you to accept the terms of service | Run the `conda tos accept ...` commands it prints, then repeat the step. |
| `setup_env.sh` stops at once because the environment `dynalyst` already exists (a second run) | Remove the half-built one with `conda env remove -n dynalyst -y`, then run `bash setup_env.sh` again; or keep it and use another name: `ENV_NAME=dynalyst2 bash setup_env.sh`. |
| `gfortran` not found, or `FAILED` while building AUTO-07p | Linux and macOS: `conda activate dynalyst`, `conda install -y -c conda-forge gfortran`, `python scripts/build_auto.py`. Windows: `winget install BrechtSanders.WinLibs.POSIX.UCRT`, open a new Anaconda Prompt, `conda activate dynalyst`, `python scripts\build_auto.py`. |
| `import jax` fails with `TypeError: expected str, bytes or os.PathLike object, not NoneType` | Linux and macOS: `conda env config vars set CUDA_ROOT="$CONDA_PREFIX"`. Windows: `conda env config vars set CUDA_ROOT=%CONDA_PREFIX%`. Then `conda deactivate` and `conda activate dynalyst`. |
| `No module named 'ipywidgets'` in a notebook | `pip install -e ".[notebook]"`, then restart the kernel (*Kernel > Restart*). |
| `No module named 'langchain_google_genai'` when choosing Gemini | `pip install -e ".[providers]"`, then restart the kernel. |
| The notebook cannot import `dynalyst` | Start `jupyter lab` from the activated `dynalyst` environment, inside the `dynalyst` folder. |
| `404 Not Found` from the language model | The model name is not available to your key. Check the key at build.nvidia.com, or choose another model in the setup cell. |
| A prompt that uses BioModels, PubMed or OmniPath fails | Check the internet connection; these tools fetch data online. |
| `NameError: name 'nn' is not defined`, or "Disabling PyTorch because PyTorch >= 2.6 is required" | PyTorch is too old for `transformers`. In the activated environment: `pip install "torch>=2.6"`. |
| pip prints "crnt4sbml 0.0.15 requires numpy==1.16.4 … which is incompatible" (and similar lines) | Harmless, ignore it. crnt4sbml is installed with `--no-deps` on purpose: its old version pins cannot be installed on Python 3.11, and it works with the newer packages (the CRNT tests pass). |
| A cell runs for a very long time | *Kernel > Interrupt*. After a restart, run the setup and files cells again. |
