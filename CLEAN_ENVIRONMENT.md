# Clean and Recreate Conda Environment

## Quick Commands

### Option 1: Remove and Recreate (Recommended)

```bash
# 1. Deactivate current environment
conda deactivate

# 2. Remove existing environment
conda env remove -n lc-bert

# 3. Create fresh environment
conda env create -f environment.yml

# 4. Activate environment
conda activate lc-bert

# 5. Install PyTorch (choose based on your system)
# For CUDA 11.3:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# For CPU only:
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Option 2: Clean Existing Environment (Keep and Update)

```bash
# 1. Activate environment
conda activate lc-bert

# 2. Remove all packages except conda
conda list | awk '{print $1}' | tail -n +4 | xargs conda remove -n lc-bert --force -y

# 3. Update environment from file
conda env update -f environment.yml --prune

# 4. Install PyTorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

---

## Step-by-Step Detailed Instructions

### Step 1: List Current Environment Info

```bash
# Check current environment
conda info --envs

# Check environment location
conda env list

# See all installed packages in lc-bert
conda list -n lc-bert

# Export current environment for backup (optional)
conda env export -n lc-bert > environment-backup.yml
```

### Step 2: Deactivate Environment

```bash
# If lc-bert is active
conda deactivate

# Make sure you're in base environment
conda activate base
```

### Step 3: Remove Environment

```bash
# Remove lc-bert environment completely
conda env remove -n lc-bert

# Verify it's removed
conda env list
```

### Step 4: Clean Conda Cache

```bash
# Clean all cached packages and tarballs
conda clean --all -y

# Clean pip cache
pip cache purge

# If you want to free more space, also clean unused packages
conda clean --packages -y
conda clean --tarballs -y
```

### Step 5: Create Fresh Environment

```bash
# Navigate to project directory
cd "C:\Users\Andri\Documents\New Cloud\Projects\Tesis\LC-BERT"

# Create new environment from clean environment.yml
conda env create -f environment.yml

# This will create environment named "lc-bert"
```

### Step 6: Activate and Install PyTorch

```bash
# Activate new environment
conda activate lc-bert

# Install PyTorch based on your GPU
# For CUDA 11.3 (most common):
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Or for CPU only:
# pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

# Or for other CUDA versions:
# CUDA 11.6: pip install torch==1.11.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7: pip install torch==1.11.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Step 7: Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
conda list

# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Test Transformers
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Test all imports
python -c "import torch, transformers, numpy, pandas, sklearn, scipy, matplotlib, seaborn, tqdm, nltk; print('All imports successful!')"

# Run test script
python test_new_models.py
```

---

## Troubleshooting

### If Environment Removal Fails

```bash
# Force remove
conda remove -n lc-bert --all -y

# If still fails, manually delete directory
# Windows:
rmdir /s "C:\Users\Andri\.conda\envs\lc-bert"
# or
rd /s /q "C:\Users\Andri\.conda\envs\lc-bert"
```

### If Conda is Slow

```bash
# Update conda first
conda update conda -y

# Set libmamba solver (much faster)
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
```

### If Package Conflicts Occur

```bash
# Try creating with no dependencies first
conda create -n lc-bert python=3.10 -y

# Activate
conda activate lc-bert

# Install packages one by one
conda install numpy=1.22.4 -y
conda install pandas=1.4.4 -y
conda install scikit-learn=1.0.2 -y
conda install scipy=1.8.0 -y
conda install matplotlib=3.5.1 -y
conda install seaborn=0.11.2 -y
conda install tqdm=4.64.0 -y
conda install nltk=3.7 -y

# Then install pip packages
pip install transformers==4.25.1 tokenizers==0.13.2 sentencepiece==0.1.97

# Finally install PyTorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Check Disk Space

```bash
# Check conda cache size
conda clean --all --dry-run

# Check environment size
du -sh ~/.conda/envs/lc-bert  # Linux/Mac
# or check folder size in Windows Explorer
```

---

## Before vs After Comparison

| Metric | Before (Old env) | After (Clean env) |
|--------|------------------|-------------------|
| **Packages** | 367+ | ~18 |
| **Size** | ~10-15 GB | ~3-4 GB |
| **Install Time** | 30-60 min | 5-10 min |
| **Conflicts** | High risk | Low risk |
| **Boot Time** | Slow | Fast |

---

## What Gets Removed

Unnecessary packages that will be removed:
- ❌ TensorFlow (2.8.0) and related packages
- ❌ Computer vision: opencv, mediapipe, face-recognition, mtcnn
- ❌ Web scraping: scrapy, selenium, beautifulsoup4
- ❌ Topic modeling: bertopic, top2vec, gensim
- ❌ Dashboard tools: dash, plotly-dash, flask
- ❌ Database: pymongo, sqlalchemy, mysql
- ❌ Data science suites: pycaret, yellowbrick
- ❌ Jupyter extensions: jupyterlab (optional to keep)
- ❌ Documentation: sphinx and related
- ❌ 300+ other unused packages

What gets kept:
- ✅ PyTorch 1.11.0+cu113
- ✅ Transformers 4.25.1
- ✅ NumPy, Pandas, Scikit-learn, SciPy
- ✅ Matplotlib, Seaborn (visualization)
- ✅ TQDM, NLTK
- ✅ Jupyter (optional)

---

## Quick Reference

```bash
# List environments
conda env list

# Activate environment
conda activate lc-bert

# Deactivate environment
conda deactivate

# List packages in environment
conda list

# Remove package
conda remove package_name -y

# Update environment
conda env update -f environment.yml --prune

# Export environment
conda env export > environment.yml

# Clean cache
conda clean --all -y
```

---

## Post-Installation Checklist

- [ ] Environment created successfully
- [ ] PyTorch installed with CUDA support
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] All core packages import without errors
- [ ] `test_new_models.py` runs successfully
- [ ] Can run basic training command
- [ ] Environment size is ~3-4 GB (check with `conda env list`)

---

## Next Steps

After cleaning environment:

1. **Test installation**: `python test_new_models.py`
2. **Run baseline**: `python main.py --dataset ag-news-normal --model_name bert-base-uncased --n_epochs 5 --train_batch_size 32`
3. **Check GPU**: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
4. **Start experiments**: See `CLAUDE.md` for experiment configurations
