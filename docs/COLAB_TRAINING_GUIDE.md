# Google Colab Training Guide

## Quick Start (5 minutes to training!)

### Step 1: Upload Notebook to Colab (2 minutes)

1. **Go to Google Colab**: https://colab.research.google.com/

2. **Upload the notebook:**
   - Click "File" → "Upload notebook"
   - Click "Choose File"
   - Select: `notebooks/train_models_colab.ipynb`
   - Wait for upload to complete

### Step 2: Enable GPU (1 minute)

1. **Change runtime:**
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: Select **"GPU"**
   - GPU type: **"T4"** (free tier)
   - Click "Save"

2. **Verify GPU:**
   - Run the first cell (click play button)
   - Should see: "GPU: Tesla T4" ✅
   - If not, repeat step 1

### Step 3: Upload Training Data (2 minutes)

**Option A: Upload your data (Recommended)**
1. Click the folder icon 📁 on the left sidebar
2. Click upload button (file with up arrow)
3. Select: `data/raw/amazon_polarity_20k.csv` (8.4 MB)
4. Wait for upload (~30 seconds)

**Option B: Use sample data (Automatic)**
- Skip upload, notebook will download data automatically
- Takes ~2 minutes on first run

### Step 4: Run Training (3-4 hours)

1. **Run all cells:**
   - Click "Runtime" → "Run all"
   - Or press Ctrl+F9

2. **Training timeline:**
   - Install packages: 1-2 minutes
   - Download models: 3-5 minutes
   - DistilBERT training: ~90 minutes
   - RoBERTa training: ~120 minutes
   - **Total: ~3.5 hours**

3. **What you'll see:**
   ```
   Training distilbert-base-uncased
   ======================================
   📂 Loading data...
      Total samples: 20000
      Train: 12000 | Val: 4000 | Test: 4000

   🤗 Loading distilbert-base-uncased...
   🔤 Tokenizing...
   🚀 Starting training...

   Epoch 1/3
   Step 100/750 | Loss: 0.435 | Accuracy: 0.823
   ...

   ✅ Training complete!
   ```

### Step 5: Download Models (5 minutes)

After training completes:

1. **Models are automatically zipped**
   - `distilbert_sentiment.zip` (~260 MB)
   - `roberta_sentiment.zip` (~500 MB)
   - `model_comparison.csv`

2. **Download methods:**

   **Method 1: Auto-download (will prompt)**
   - Browser will ask to save files
   - Click "Save" for each file

   **Method 2: Manual download**
   - Click folder icon 📁 on left
   - Right-click each zip file
   - Select "Download"

3. **Extract on your local machine:**
   ```bash
   cd "/mnt/d/Project/AI projects"

   # Extract models
   unzip distilbert_sentiment.zip -d models/
   unzip roberta_sentiment.zip -d models/

   # Check they exist
   ls -lh models/distilbert_sentiment/
   ls -lh models/roberta_sentiment/
   ```

---

## Important Notes

### ⏰ Session Limits
- **Free Colab:** 12-hour maximum runtime
- **Solution:** Training takes ~3.5 hours (well within limit)
- If session disconnects, you'll need to restart

### 💾 Data Persistence
- **Colab storage is temporary!**
- Files are deleted when runtime disconnects
- **Always download models before closing!**

### 🔄 If Session Disconnects Mid-Training

**Don't panic!** You have options:

1. **Check if files still exist:**
   - Click folder icon 📁
   - If models exist → Download immediately
   - If gone → Need to re-run

2. **Resume from checkpoint (if DistilBERT completed):**
   - Download DistilBERT
   - Re-run only RoBERTa cell
   - Saves ~90 minutes

3. **Prevent disconnects:**
   - Keep browser tab open
   - Run this in a cell to prevent idle timeout:
     ```python
     # Keep alive script (run in separate cell)
     import time
     while True:
         time.sleep(60)
         print(".", end="", flush=True)
     ```

### 📊 Expected Results

Based on similar experiments, you should see:

| Model | Accuracy | F1-Score | Training Time | Size |
|-------|----------|----------|---------------|------|
| **BERT** (yours) | 91.875% | 0.922 | - | 639 MB |
| **DistilBERT** | ~89-91% | ~0.89-0.91 | 90 min | 260 MB |
| **RoBERTa** | ~92-94% | ~0.92-0.94 | 120 min | 500 MB |

**Typical pattern:**
- DistilBERT: Slightly lower accuracy, much faster
- RoBERTa: Slightly higher accuracy, similar speed to BERT

---

## Troubleshooting

### Problem: "GPU not available"

**Solution:**
1. Runtime → Change runtime type → GPU
2. Make sure it says "T4" not "None"
3. Re-run first cell to verify

### Problem: "Out of memory"

**Solution:**
Reduce batch size in training cells:
```python
train_model(
    ...
    batch_size=8,  # Changed from 16
    ...
)
```

### Problem: "Session disconnected"

**Solution:**
1. Reconnect: Runtime → Reconnect
2. Check if files still exist (folder icon)
3. If models exist → Download immediately
4. If gone → Re-run from start

### Problem: Upload data is slow

**Solution:**
Use Option B (automatic download) instead:
- Just run the notebook
- It will download data from Hugging Face
- Takes ~2 minutes

---

## After Training

### Compare Results Locally

1. **Add BERT results to comparison:**
   ```python
   # On your local machine
   import pandas as pd

   # Read Colab results
   colab_results = pd.read_csv('model_comparison.csv')

   # Add BERT
   bert_row = pd.DataFrame({
       'Model': ['BERT'],
       'Accuracy': ['0.9188'],
       'F1-Score': ['0.9220'],
       'ROC-AUC': ['0.9708'],
       'Training Time (min)': ['-'],
       'Model Size (MB)': ['639.0']
   })

   all_results = pd.concat([bert_row, colab_results], ignore_index=True)
   print(all_results)
   ```

2. **Select winner** based on:
   - Best accuracy (most important)
   - Acceptable size (for deployment)
   - Reasonable speed

3. **Next: Build deployment** with winning model!

---

## Cost

**Google Colab:** FREE ✅
- T4 GPU included
- 12-hour sessions
- No credit card needed
- Perfect for this project

**Upgrade options (not needed):**
- Colab Pro: $9.99/month (faster GPUs, longer sessions)
- Only needed if training many models

---

## Tips for Success

1. ✅ **Keep browser tab open** during training
2. ✅ **Download models immediately** after training
3. ✅ **Monitor progress** - check notebook every hour
4. ✅ **Save comparison.csv** - you'll need it later
5. ✅ **Don't close tab** until you've downloaded everything

---

## Ready to Start?

1. Open: https://colab.research.google.com/
2. Upload: `notebooks/train_models_colab.ipynb`
3. Enable GPU: Runtime → Change runtime → GPU
4. Run all: Runtime → Run all
5. Wait ~3.5 hours
6. Download models
7. Celebrate! 🎉

**Questions?** Check the troubleshooting section above.
