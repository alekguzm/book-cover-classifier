# Book Cover Genre Classifier

A deep learning image classifier that predicts the genre of a book from its cover art, trained on 50,000+ Amazon book cover images across 30 categories.

> "Can a model judge a book by its cover?"

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Custom CNN (from scratch) | 13.84% | Baseline |
| ResNet50 (final layer only) | 22.04% | Transfer learning |
| ResNet50 (full fine-tune) | 28.35% | Best result |

Random baseline: **3.3%** (1/30 classes) — model performs ~8.7x better than chance.

---

## Dataset

- **Source**: [Book Cover 30 Dataset](https://www.kaggle.com/datasets/mexwell/book-cover-dataset) (Kaggle)
- **Size**: 50,000+ Amazon book cover images
- **Classes**: 30 genres (Biographies, Children's Books, Science Fiction, Travel, etc.)
- **Image size**: 224×224 px

---

## Approach

### 1. Custom CNN (Baseline)
Built a 3-layer convolutional neural network from scratch to establish a baseline. Achieved **13.84% accuracy**.

### 2. Transfer Learning with ResNet50
Loaded a pretrained ResNet50 (ImageNet weights) and replaced the final fully connected layer with a 30-class output head. Trained only the final layer. Achieved **22.04% accuracy**.

### 3. Full Fine-Tuning
Unfroze all layers of ResNet50 and continued training with a lower learning rate (`1e-4`). Achieved **28.35% accuracy** — the best result.

---

## Model Architecture

```
ResNet50 (pretrained on ImageNet)
└── Final FC layer replaced: 2048 → 30 classes
```

**Training config:**
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 10 (fine-tune) + 5 (full fine-tune)
- Hardware: Kaggle GPU (T4)

---

## Project Structure

```
book-cover-classifier/
├── book_cover_classifier.ipynb   # Main notebook
├── README.md
└── book-covers/                  # Dataset (not included, see setup)
    ├── book30-listing-train.csv
    ├── book30-listing-test.csv
    └── title30cat/224x224/       # Images
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/aleksander-guzman/book-cover-classifier
cd book-cover-classifier
```

**2. Install dependencies**
```bash
pip install torch torchvision pandas pillow scikit-learn matplotlib kagglehub
```

**3. Download the dataset**
```python
import kagglehub
path = kagglehub.dataset_download("mexwell/book-cover-dataset")
```

**4. Run the notebook**

Open `book_cover_classifier.ipynb` in Jupyter Lab or Kaggle Notebooks.

---

## Key Learnings

- Writing a **custom PyTorch Dataset class** to load images from a CSV-labeled dataset
- Applying **transfer learning** with a pretrained ResNet50 model
- Understanding the difference between freezing/unfreezing layers during fine-tuning
- Book genre classification is a genuinely hard problem — genre boundaries are subjective, and some covers give very few visual cues

---

## Author

**Aleksander Guzman**  
[GitHub](https://github.com/aleksander-guzman) · [LinkedIn](https://linkedin.com/in/aleksander-guzman)
