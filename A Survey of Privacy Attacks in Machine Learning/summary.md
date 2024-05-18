# Introduction
 * Rising concern about the security, privacy, and fairness in machine learning
 * Privacy leaks can occur due to model structure, poor generalization, memorization of sensitive data, or adversarial robustness training.

### Types of Attacks on Machine Learning Systems:
- Integrity Attacks: Cause misclassification (e.g., evasion, poisoning backdoor attacks).
- Availability Attacks: Aim to increase misclassification errors (e.g., poisoning attacks).
- Privacy and Confidentiality Attacks: Try to infer information about user data or the models themselves.

#  MACHINE LEARNING
## Types of Learning
### Supervised Learning
### Unsupervised Learning
### Reinforcement Learning
### Semi-supervised Learning
E.g.: GAN - Generative Adversarial Networks (GANs)
### Generative and Discriminative Learning
Discriminative Classifiers:
- Model the conditional probability `p(y∣x)`
- Learn decision boundaries directly based on input data `x`
- Examples: Logistic regression, neural networks.

Generative Classifiers:
- Capture the joint distribution `p(x,y)`
- Often do not require labels, modeling `p(x)` instead.
- Examples: Naive Bayes, language models predicting next word(s), GANs (Generative Adversarial Networks), and Variational Autoencoders (VAEs).
- Aim to generate data samples that reflect the properties of the training data.

This explains the distinction between algorithms that directly model decision boundaries (discriminative) and those that model the underlying data distribution (generative).

## Learning Architectures
### Centralized Learning
- Definition: Data and models are located together.
- Setup: Data is gathered in one central location, which could be a single machine or multiple machines in the same data center.
- Example: Machine Learning as a Service (MLaaS) where data owners upload data to a cloud service for model training.

### Distributed Learning
- Motivation: Needed for handling large data volumes, computing capacity, memory capacity, and privacy concerns.

#### Variants Relevant to Privacy:

Collaborative/Federated Learning:
- Process: Data remains on remote devices; local models are updated with local data, and only intermediate updates are sent to a central server to create a global model.
- Privacy Benefit: Data does not leave the remote devices.

Fully Decentralized/P2P Learning:
- Process: No central server; devices exchange updates directly with each other.
- Privacy Aspect: Reduces need to trust a central server but may still face privacy attacks.

Split Learning:
- Process: The model is split between edge devices (initial layers) and a central server (final layers).
- Benefit: Reduces communication costs by sending intermediate outputs instead of raw data.
- Usage: Common in scenarios with resource-limited edge devices, such as IoT setups.

## Training and Inference
1. **Training and Inference Overview**:
   - Supervised ML models are typically trained using Empirical Risk Minimization (ERM).
   - Objective: Minimize the risk or objective function over the training dataset.
   - Loss function $\( l(·) \)$, such as cross-entropy loss, is used.

2. **Empirical Risk Minimization (ERM)**:
   - Minimizes an estimated objective function over the available data samples.
   - Regularization may be added to reduce overfitting and stabilize training.

3. **Training in Centralized Settings**:
   - Uses iterative optimization algorithms like gradient descent.
   - For large datasets, Stochastic Gradient Descent (SGD) and its variants (mini-batches) are preferred.
   - **SGD Formula**:
     - $\( \theta_{t+1} = \theta_t - \eta g \)$
     - $\( g = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta l(f(x_i; \theta), y_i) \)$
   - Improvements include momentum, adaptive learning rates (e.g., RMSprop), and combinations like Adam.

4. **Training in Distributed Settings**:
   - **Federated Averaging**:
     - Each device calculates gradient descent locally and shares updated weights with a parameter server.
     - Server averages weights to update the global model.
     - **Formula**:
       - $\( \theta_{t+1} = \frac{1}{K} \sum_{k=1}^{K} \theta^{(k)}_t \)$

   - **Downpour (Synchronized) SGD**:
     - Devices share loss gradients with a parameter server.
     - Server aggregates gradients and performs a gradient descent step.
     - **Formula**:
       - $\( \theta_{t+1} = \theta_t - \eta \sum_{k=1}^{K} \frac{m^{(k)}}{M} g^{(k)}_t \)$
       - $\( g^{(k)}_t \)$ is the gradient from participant $\( k \)$.

5. **Inference**:
   - Trained models are used to make predictions on new, unseen data.
   - Model parameters are assumed fixed during inference.
   - Most privacy attacks target the inference phase, except for some on collaborative learning, which occur during training.

# Threat Model

To comprehend and counter attacks in machine learning with a focus on privacy, establishing a comprehensive threat model is crucial. This model outlines the environment, involved actors, and assets requiring protection.

## Assets:
- **Training Dataset (D)**: Contains sensitive data.
- **Model Parameters (θ)**: Core components of the model.
- **Model Hyperparameters**: Configuration settings influencing model behavior.
- **Model Architecture**: Structure of the model.

## Actors:
1. **Data Owners**: Possess sensitive data utilized for training.
2. **Model Owners**: May or may not own the data, but manage the model.
3. **Model Consumers**: Utilize services offered by model owners.
4. **Adversaries**: Have access to model interfaces and may attempt unauthorized access.

## Information Flow and Possible Actions:
- Depicted in Figure 1 of the threat model.
- Logical representation; assets may be collocated or distributed.

## Distributed Learning:
- Introduces spatial models of adversaries.
- In federated learning, adversaries may be global or local (Figure 2).

## Adversarial Knowledge:
- Ranges from limited (access to ML API) to full model parameters and training settings.
- Includes partial knowledge of model architecture, hyperparameters, or training setup.
- Assumptions about dataset knowledge vary; often assumes no access to training data samples but some knowledge of data distribution.

## Taxonomy of Attacks:
- **Black-box Attacks**: Adversary lacks knowledge of model parameters, architecture, or training data. Example: MLaaS.
- **White-box Attacks**: Full access to model parameters or loss gradients during training, common in distributed training.
- **Partial White-box Attacks**: Stronger assumptions than black-box attacks but not full access to model parameters.

## Time of Attack:
- Majority of research focuses on attacks during inference.
- Collaborative learning attacks assume access to model parameters or gradients during training.
- Training phase attacks enable different adversarial behaviors: passive (observation only) or active (interference).

## Scope of Survey:
- Focuses on privacy attacks related to unintentional information leakage about data or machine learning models.
- Does not cover security-based attacks like model poisoning or evasion attacks, or attacks against hosting infrastructure.

# 4. Attack Types

In privacy-related attacks, adversaries aim to gain unauthorized knowledge about training data $\( D \)$, the model $\( f \)$, or data properties, including unintentional biases. These attacks are categorized into four types:

## 4.1. Membership Inference Attacks
- **Objective**: Determine if input sample $\( x \)$ was part of training set $\( D \)$.
- **Popular Category**: First introduced by Shokri et al.
- **Approaches**: Black-box (model's output prediction vector) or white-box (access to model parameters and gradients).
- **Targets**: Supervised and generative models like GANs and VAEs.
- **Perspective**: Data owners may audit black-box models for unauthorized data usage.

## 4.2. Reconstruction Attacks
- **Objective**: Recreate training samples and/or their labels.
- **Variants**: Partial or full reconstruction.
- **Terminology**: Also known as attribute inference or model inversion.
- **Focus**: Recovery of sensitive features or full data samples.
- **Major Distinction**: Actual data reconstruction vs. creation of class representatives.

## 4.3. Property Inference Attacks
- **Objective**: Extract dataset properties not explicitly encoded or correlated with the learning task.
- **Examples**: Gender ratio in patient datasets, presence of glasses in gender classification.
- **Implications**: Privacy concerns, insights into training data, potential security implications.

## 4.4. Model Extraction Attacks
- **Objective**: Extract or reconstruct a model by creating a substitute $\( \hat{f} \)$ that behaves similarly to the target model $\( f \)$.
- **Goals**: Match accuracy of $\( f \)$ on test set or replicate its decision boundary.
- **Approaches**: Task accuracy extraction (match task) vs. fidelity extraction (match decision boundary).
- **Efficiency**: Minimize queries; knowledge of target model architecture may not be strictly necessary.
- **Other Approaches**: Recovery of model information such as hyperparameters or architectural properties.

# 5. Causes of Privacy Leaks

Understanding the conditions under which machine learning models leak information is crucial. Here are the main causes:

## 5.1 Membership Inference Attacks
- **Poor Generalization**: Overfitting can enhance attack accuracy.
- **Model Complexity**: More complex models are prone to higher attack accuracy.
- **Dataset Characteristics**: Certain models and datasets are more vulnerable.
- **Adversarial Training**: Robust training methods can increase susceptibility to attacks.
- **GANs**: Overfitting in generators can facilitate successful attacks.

## 5.2 Reconstruction Attacks
- **Generalization Error**: Higher error increases the likelihood of inferring data attributes.
- **Predictive Power**: Models with high predictive power are more susceptible.

## 5.3 Property Inference Attacks
- **Generalization Not a Cause**: Property inference is possible even with well-generalized models.
- **Limited Understanding**: Less information available on factors influencing these attacks.

## 5.4 Model Extraction Attacks
- **Inverse Relationship with Overfitting**: Higher overfitting decreases model extraction success.
- **Model Accuracy**: Higher accuracy models are harder to steal.
- **Dataset Complexity**: Higher class count can lead to worse attack performance.

# 6. IMPLEMENTATION OF THE ATTACKS
## 6.1 Attacks Against Centralized Learning

In this setting, where models and data are collocated during training, the common approach involves shadow training and various attack types:

### 6.1.1 Shadow Training
- **Design Pattern**: Utilizes shadow models and meta-models.
- **Architecture**: Shadow models emulate target model behavior on shadow datasets.
- **Meta-Model**: Trained to infer membership or properties based on shadow model outputs.
- **Threshold-based Attacks**: Simplify by using threshold functions to indicate membership.

### 6.1.2 Membership Inference Attacks
- **Shadow Models**: Commonly used for black-box attacks.
- **Attack Dataset**: Constructed using shadow models' outputs.
- **Threshold-based Reduction**: Some attacks simplify to threshold functions.
- **Data-Driven Attacks**: Attack success can be independent of target model type.

### 6.1.3 Reconstruction Attacks
- **Assumption**: Adversary has access to model, prior distribution, and output.
- **Optimization Methods**: MAP estimates, gradient descent used for feature recovery.
- **GAN-Based Approaches**: Auxiliary information improves reconstruction quality.

### 6.1.4 Property Inference Attacks
- **Meta-Model Training**: Infers differences in output vectors for data with/without properties.
- **Feature Representations**: Support vectors, neural network layer outputs used in white-box attacks.
- **Application**: Language model embeddings, graph embeddings, and dataset properties.

### 6.1.5 Model Extraction Attacks
- **System of Equations**: Inputs and outputs treated as equations to retrieve model parameters.
- **Active Learning Strategy**: Queries target model for data points to train substitute model.
- **Adaptive Training**: Queries extended based on adaptive strategies close to decision boundary.
- **Theoretical Approaches**: Direct extraction shown feasible for nonlinear models.



