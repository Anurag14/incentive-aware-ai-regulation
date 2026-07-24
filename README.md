# Incentive-Aware AI Regulation

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/muandet-lab/incentive-aware-ai-regulation.git](https://github.com/muandet-lab/incentive-aware-ai-regulation.git)
cd incentive-aware-ai-regulation

# Install dependencies
```bash
1. numpy
2. scipy
3. matplotlib
4. scienceplots (for publication-quality figures)
5. torch & torchvision (for Waterbirds experiments)
```
## 📊 Experiments

The repository reproduces the three main empirical results from the paper:

### 1. The Geometry of Gaming (Synthetic)
Demonstrates why regulations must be convex. We simulate a "Bad Agent" who mixes three forbidden distributions to fool a Naive Regulator.

* **Script:** `experiments/gaming_the_regulation/simulation.ipynb`
* **Result:** Comparison of wealth accumulation between Naive (Exploding wealth = False Negative) and Credal (Zero wealth = True Negative) regulators.

## 📂 Project Structure
```bash
.
├── experiments/
│   ├── gaming_the_regulation   # Fig 1a: Naive vs Credal gaming
│   ├── waterbirds              # Fig 1b/c: Waterbirds results
│   ├── fairness/               # Fig 1d: Implicit fairness regulation experiment
│   └── testing/                # Fig 2: Incetive aware tests  
├── meta_data/                  # Saved plots and assets
└── README.md
```
