# IBM AI Engineering Capstone Project: Deep Learning Applications

![Project Banner](https://github.com/hotrungtruc/AI-Capstone-Project-with-Deep-Learning/blob/main/assets/data.png?raw=true)
This repository contains the source code, labs, and final project for the **AI Capstone Project with Deep Learning** course, part of the **IBM AI Engineering Professional Certificate** on Coursera.

The project focuses on mastering Deep Learning frameworks (PyTorch & Keras), implementing Computer Vision architectures (CNNs & Vision Transformers), and building a hybrid model for real-world image classification tasks.

## 📂 Project Structure

The repository is organized into four main modules, progressing from data handling to advanced model architectures.

### Phase 1: Data Preparation & Loading
*Focus: Efficient data pipelines and augmentation techniques.*

| File | Description |
|------|-------------|
| `Compare_Memory-Based_Versus_Generator-Based_Data_Loading.ipynb` | **Efficiency Analysis**: Compares memory usage and speed between loading entire datasets into RAM vs. using generator-based loading for large datasets. |
| `Lab_M1L2_Data Loading_and_Augmentation_Using_Keras.ipynb` | **Keras Pipeline**: Implementing `ImageDataGenerator` and `tf.data` for efficient image loading and real-time augmentation in TensorFlow/Keras. |
| `Lab_M1L3_Data_Loading_and_Augmentation_Using_PyTorch.ipynb` | **PyTorch Pipeline**: Building custom `Dataset` classes and using `DataLoader` with transforms for effective data handling in PyTorch. |

### Phase 2: Convolutional Neural Networks (CNNs)
*Focus: Transfer learning with ResNet and framework comparison.*

| File | Description |
|------|-------------|
| `Lab_M2L1_Train_and_Evaluate_a_Keras-Based_Classifier.ipynb` | **Keras ResNet**: Fine-tuning a pre-trained ResNet model using Keras to classify images. |
| `Lab_M2L2_Implement_and_Test_a_PyTorch-Based_Classifier.ipynb` | **PyTorch ResNet**: Implementing and fine-tuning a pre-trained ResNet model using PyTorch. |
| `Lab_M2L3_Comparative_Analysis_of_Keras_and_PyTorch_Models.ipynb` | **Framework Comparison**: A comparative study of the Keras and PyTorch models regarding training time, syntax complexity, and accuracy. |

### Phase 3: Vision Transformers (ViT)
*Focus: Implementing state-of-the-art Attention-based architectures.*

| File | Description |
|------|-------------|
| `Lab_M3L1_Vision_Transformers_in_Keras.ipynb` | **ViT in Keras**: Implementation and fine-tuning of the Vision Transformer architecture using TensorFlow/Keras. |
| `Lab_M3L2_Vision_Transformers_in_PyTorch.ipynb` | **ViT in PyTorch**: Implementation and fine-tuning of the Vision Transformer architecture using PyTorch / Hugging Face. |

### Phase 4: Final Capstone Project
*Focus: Advanced Hybrid Architecture.*

| File | Description |
|------|-------------|
| `lab_M4L1_Land_Classification_CNN-ViT_Integration_Evaluation.ipynb` | **Hybrid CNN-ViT Model**: The final project combining the feature extraction power of CNNs with the attention mechanisms of ViT to solve a Land Use Classification problem. |

## 🛠 Technologies & Tools

* **Languages:** Python 3.x
* **Frameworks:** PyTorch, TensorFlow (Keras)
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Torchvision.
* **Architectures:** ResNet50, ResNet18, Vision Transformer (ViT base-patch16).

## 📝 Key Results

This project demonstrates:
1.  **Cross-Framework Proficiency:** Ability to build identical pipelines in both PyTorch and Keras.
2.  **Modern Architecture:** Transitioning from standard CNNs to Transformer-based vision models.
3.  **Optimization:** Handling large datasets efficiently using generators and data loaders.
4.  **Hybrid Modeling:** achieving higher accuracy by integrating CNN and ViT logic.

---
*Created as part of the IBM AI Engineering Professional Certificate.*
