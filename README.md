🌍 Land Use Image Classification using CNN & ResNet

This project implements an end-to-end image classification pipeline to identify land use and land cover categories from satellite imagery using deep learning models.
Both a custom Convolutional Neural Network (CNN) and a fine-tuned ResNet18 model are trained and evaluated on the EuroSAT dataset.

⸻

🚀 Overview
	•	Multi-class image classification on 27,000+ satellite images
	•	Compared a from-scratch CNN with a pretrained ResNet18
	•	Focused on generalization, validation discipline, and reproducibility
	•	Implemented early stopping and model checkpointing to avoid overfitting

⸻

📊 Dataset
	•	Dataset: EuroSAT (RGB)
	•	Classes: 10 land-use categories (e.g., Forest, Residential, River, Highway, etc.)
	•	Image Size: 64 × 64 RGB
	•	Split:
	•	Train: ~70%
	•	Validation: ~10%
	•	Test: ~20%

⸻

🧠 Models Implemented

1️⃣ Custom CNN
	•	Multiple convolutional layers with ReLU activation
	•	Max pooling for spatial downsampling
	•	Fully connected layers with dropout
	•	Trained end-to-end using cross-entropy loss

2️⃣ ResNet18 (Transfer Learning)
	•	Initialized with ImageNet-pretrained weights
	•	Final classification layer adapted for 10 classes
	•	Fine-tuned using a low learning rate
	•	Compared with a non-pretrained baseline

⸻

⚙️ Training Strategy
	•	Optimizer: Adam
	•	Loss Function: Cross Entropy Loss
	•	Early Stopping: Based on validation loss
	•	Model Selection: Best checkpoint chosen via validation performance
	•	Reproducibility: Fixed random seed

⸻

📈 Evaluation
	•	Tracked training and validation loss/accuracy
	•	Evaluated final model on held-out test set
	•	Generated:
	•	Accuracy plots
	•	Confusion matrix
	•	Per-class recall
	•	Analyzed cases where the model was confident but incorrect

⸻

🗂 Project Structure

    ├── notebooks/
    │   ├── cnn_training.ipynb
    │   ├── resnet_finetuning.ipynb
    ├── logs_cnn/
    │   └── best_model.pth
    ├── logs_resnet/
    │   └── best_model.pth
    ├── data/
    │   └── euroSAT_train_val_test.pkl
    ├── plots/
    │   ├── loss_curves.png
    │   ├── confusion_matrix.png
    └── README.md

⸻

💡 Key Learnings
	•	Transfer learning with pretrained ResNet significantly improves convergence and generalization.
	•	Early stopping is critical for preventing overfitting on relatively small image datasets.
	•	Validation-driven model selection produces more reliable test performance than fixed-epoch training.

⸻

🛠 Tech Stack
	•	Language: Python
	•	Framework: PyTorch
	•	Models: CNN, ResNet18
	•	Tools: NumPy, Matplotlib, scikit-learn
	•	Platform: Kaggle / Jupyter Notebook

⸻

📌 Notes
	•	This project emphasizes model evaluation rigor and training discipline, not just accuracy.
	•	Designed to reflect real-world ML experimentation workflows.

⸻

👤 Author

Prosenjit Kundu
Backend / Platform Engineer
🔗 LinkedIn￼
🔗 GitHub￼
