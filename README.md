# Object Detection, the Neuromatch Project

This repository contains our Python implementation of **single-agent** and **multi-agent reinforcement learning systems** for collaborative object detection.
Developed as part of the **Neuromatch Summer School 2025**, this project explores how multiple agents can **collaboratively move and resize bounding boxes** to locate objects in images.
The system uses **IoU-based rewards** and **VGG-16 extracted features** within a custom **PettingZoo environment**.

---

## Motivation

* Explore **multi-agent reinforcement learning (MARL)** for visual tasks.
* Build a **custom PettingZoo environment** tailored for bounding-box control.
* Compare **single-agent vs. multi-agent** performance.
* Implement QMIX for **coordinated decision-making** among agents.
* Visualize training progress with **IoU metrics**, reward/loss curves, and GIF animations.

---

## Repository Contents

### 1. Custom Environment

* Designed with **PettingZoo**, allowing agents to collaboratively adjust bounding boxes.
* Includes a 9-action space: move, resize, aspect-ratio changes, and a trigger action.
* Uses **VGG-16** for feature extraction.

### 2. Dataset Preparation

* Filtered **Pascal VOC 2012** dataset to include only **cats and dogs**.
* Preprocessing and normalization compatible with VGG-16.

### 3. RL Models

* **Single-Agent Model** – baseline for comparison.
* **Three Multi-Agent Models**, including one using **QMIX for shared decision-making**.
* Implemented from scratch in **PyTorch**, with replay buffers, target networks, and epsilon-greedy exploration.

### 4. Training & Evaluation

* Tracks **IoU-based rewards** and cumulative performance.
* Produces GIFs and plots for **visualizing detection accuracy** and agent coordination.

---

## Requirements

**Install dependencies:**

```bash
pip install torch torchvision pettingzoo gymnasium numpy matplotlib imageio opencv-python
```

---

## Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/collaborative-object-detection-rl.git
cd collaborative-object-detection-rl
```

2. **Prepare dataset:**

The script will automatically download **Pascal VOC 2012**, or you can place it in the `./data` folder.

3. **Run training:**

```bash
python train_single_agent.py
python train_multi_agent.py
```

4. **View results:**

* Generated **GIF animations** of agents during testing.
* Reward/loss plots saved in the output directory.

---

## Results

* Agents progressively improve bounding box accuracy over training.
* Multi-agent QMIX setup shows **better coordination** and higher IoU than the single-agent baseline.
* Visualizations include:

  * IoU metric plots
  * Reward and loss curves
  * GIF animations of detection sequences

---

## Reference

* **Pascal VOC Dataset:** [http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)
* **QMIX Paper:** *Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning* (Rashid et al., 2018)
* **PettingZoo Documentation:** [https://www.pettingzoo.ml/](https://www.pettingzoo.ml/)
* **Neuromatch Summer School:** [https://neuromatch.io](https://neuromatch.io)

---

## Acknowledgments

* **Neuromatch Academy** – for providing the summer school platform and mentorship
* **PyTorch** – deep learning framework
* **PettingZoo** – multi-agent RL toolkit
* **Pascal VOC creators** – dataset providers
* Open-source contributors and the MARL research community

