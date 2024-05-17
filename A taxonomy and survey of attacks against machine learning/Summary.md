## Abstract and Introduction
Paper presents **attacks against ML** classified by shared: 
- Key characteristics
- Defense approaches

Attacks:
- Poisoning attack: against **training** data
- Evasion attack: against **test** data

to: 
-  decrease performance
-  miscalssification
-  backdoors
-  neural torjans

Adversarial ML: 
- ML techniques against adversarial opponent
- Lies in intersection of CS and ML.

# Proposed taxonomy
## Preparation
1. Attacker Knowledge: `K 1`: Ground truth, `K 2`: learning algorithm
– Blackbox attacks: ¬ `K 1` ∧ ¬ `K 2`
– Graybox attacks: `K 1` ∨ `K 2`
– Whitebox attacks: `K 1` ∧ `K 2`

2. Algorithm:
3. Game Theory: equipped with a strategic element

## Manifestation
1. Attack Specificity:
- Targeted
- Indiscriminate

2. Attack Type
- Poisoning
- Evasion

3. Attack Mode
- colluding
- non-colluding

## Attack evaluation
1. Evaluation Approach:
- analytically
- in simulation
- experimentally

2. Performance impact
- False positives
- False negatives
- Both false positives and false negatives
- Clustering accuracy reduction

# Adversarial attacks on machine learning
## Intrusion detection
| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Evaluation Approach |
|---|---|---|---|---|---|---|
| Barreno et al. [29] | Whitebox | Naive learning algorithm | Poisoning | Targeted | Both False Positives and False Negatives | Experimental |
| Biggio et al. [38] | Graybox to Whitebox | Support Vector Machine (SVM) | Evasion | Targeted | False Negatives | Experimental |
| Xiao et al. [39] | Graybox to Whitebox | Least Absolute Shrinkage and Selection Operator (LASSO) & Elastic Net | Poisoning | Both Targeted and Indiscriminate |  Both False Positives and False Negatives, Clustering Accuracy Reduction | Experimental |
| Chen et al. [40] | Graybox & Whitebox | Spectral Clustering, Community Discovery, node2vec | Poisoning | Both Targeted and Indiscriminate | False Negatives, Clustering Accuracy Reduction | Experimental |
| Rubinstein et al. [41] | Blackbox to Whitebox | Principal Component Analysis (PCA) | Poisoning | Both Targeted and Indiscriminate | Both False Positives and False Negatives | Experimental |
| Wang et al. [42] | Graybox to Whitebox | Support Vector Machines (SVM), Bayesian, Decision Trees & Random Forests | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives, Clustering Accuracy Reduction | Experimental |
| Demetrio et al. [43] | Blackbox | Convolutional Neural Networks (CNNs) | Evasion | Targeted | False Negatives | Experimental |

**Notes:**

* **Attacker Knowledge:**  Describes the attacker's knowledge about the IDS (Blackbox, Graybox, Whitebox).
* **Algorithm Targeted:**  The specific ML algorithm targeted by the attack.
* **Attack Type:**  Whether the attack targets the training data (Poisoning) or test data (Evasion).
* **Attack Specificity:**  Whether the attack targets specific data points (Targeted) or a broader range of samples (Indiscriminate).
* **Performance Impact:**  The impact of the attack on the IDS's performance, measured by metrics like False Positives, False Negatives, and Clustering Accuracy Reduction.
* **Evaluation Approach:**  The methods used to evaluate the effectiveness of the attack (Analytical, Simulation, Experimental).


## Spam filtering
### 1. Linear classifiers
**The Target:**

* **Spam filter:**  The target is a spam filter that uses a Naive Bayes classifier for classifying emails as spam or legitimate. 
* **Naive Bayes:** This is a probabilistic classifier commonly used in spam filtering due to its simplicity and effectiveness. It calculates the probability of an email belonging to a class (spam or legitimate) based on the presence of certain words or features.

**The Attacker:**

* **Knowledge:** The attacker has a *graybox* understanding of the spam filter. This means they know the general classification algorithm (Naive Bayes) but don't have complete access to the training data or parameters.
* **Goal:** The attacker aims to *increase the rate of false negatives*. This means they want to trick the spam filter into classifying spam emails as legitimate, allowing them to bypass the filter. 
* **Strategy:** The attacker uses a *membership query attack*. This means they send emails to the spam filter and observe how the filter classifies them. They then use this information to learn about the spam filter's decision boundaries and to craft spam emails that are likely to be misclassified as legitimate.
* **Indiscriminate:** The attack is *indiscriminate* in the sense that the attacker aims to affect the filter's behavior across the entire set of potential spam emails, not just a specific email. They want to optimize their attack to maximize the overall rate of false negatives.

**ACRE Model:**

The ACRE model is a specific example of this type of attack.  The attacker uses the following steps:

1. **Membership queries:** The attacker sends various emails to the spam filter (both known spam and legitimate emails) and observes how they are classified.
2. **Learn decision boundaries:**  The attacker uses the classification results to understand the spam filter's decision-making process. They try to identify features (words, patterns) that strongly influence the filter's classification.
3. **Craft adversarial emails:** The attacker uses this knowledge to craft spam emails that are likely to be misclassified as legitimate by the filter.  These emails might include features that the filter typically associates with legitimate emails while minimizing features that are considered typical of spam.

### 2. Game theory


| Research | Game Type | Attacker Action | Defender Action | Attack Knowledge | Attack Type | Attack Specificity | Key Insight |
|---|---|---|---|---|---|---|-----|
| Li & Vorobeychik [46] | Stackelberg Game | Manipulate email features (feature reduction) | Choose a linear classifier | Graybox | Evasion | Indiscriminate | Attackers use budget constraints and optimization to manipulate features. |
| Bruckner & Scheffer [47] | Stackelberg Game | Generate test data at application time | Commit to a predictive model | Graybox | Evasion | Indiscriminate | Attackers can influence email generation at application time, increasing false negatives. |
| Bruckner & Scheffer [48] | Static Game | Poison data at training and test time | Choose classifier (SVM) | Graybox | Poisoning | Indiscriminate | Attackers target training data and test data, potentially increasing both false positives and false negatives. |
| Liu & Chawla [49] | Infinite Stackelberg Game | Spam (modify emails) or Status Quo | Retrain or Status Quo | Whitebox | Poisoning | Indiscriminate |  Attackers aim to increase false negatives through whitebox poisoning attacks targeting training data. |
| Zhang & Zhu [50] | Game-theoretic framework | Modify training data (poisoning) | Choose classifier (SVM) | Whitebox | Poisoning | Indiscriminate | Attackers manipulate training data to maximize error rate (both false positives and false negatives). |
| Grosshans et al. [51] | Bayesian Game (non-zero sum, incomplete information) | Exercise control over data distribution (evasion) | Choose classifier | Whitebox | Evasion | Indiscriminate | Attackers exploit incomplete information about the defender's cost function to misclassify their data as benign. | 

**Notes:**

* **Stackelberg Game:** A hierarchical game where the "leader" moves first, and the "follower" responds.
* **Static Game:** Both players make decisions simultaneously, without knowing the other's actions.
* **Bayesian Game:** A game with incomplete information, where players have uncertainty about the other player's payoffs or actions.
* **Graybox Attack:** Attacker has partial knowledge of the system.
* **Whitebox Attack:** Attacker has full knowledge of the system.
* **Evasion Attack:** Attacker manipulates test data to bypass detection.
* **Poisoning Attack:** Attacker modifies training data to influence the system's behavior.
* **Indiscriminate Attack:** Attacker targets a broad set of samples

### 3. Naive Bayes
Naive Bayes is a probabilistic classifier that calculates the likelihood of an email being spam or legitimate based on the presence of specific words. 

**Key Attacks:**

| Attack | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact |
|---|---|---|---|---|
| Dalvi et al. [30] | Whitebox | Evasion | Indiscriminate | False Negatives |
| Barreno et al. [35] | Graybox & Whitebox | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives |
| Nelson et al. [52] | Both Blackbox & Whitebox | Poisoning | Both Targeted and Indiscriminate | Both False Positives and False Negatives | 
| Naveiro et al. [53] | Graybox | Evasion | Indiscriminate | False Negatives |

### 4. Support Vector Machines (SVM)

| Attack | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Goal |
|---|---|---|---|---|---|
| Xiao et al. [54] | Whitebox | Poisoning | Targeted | Both False Positives and False Negatives | maximize the classification error |

## Visual Recognition
**3.3.1. Principal Component Analysis (PCA)**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Carlini & Wagner [55] |  Various (Blackbox to Whitebox) | Both Poisoning and Evasion | Targeted | Both False Positives and False Negatives |  Investigated three threat models based on attacker knowledge (perfect, limited, zero) and corresponding attack types (whitebox, graybox, blackbox). The attacks aim to cause misclassification by manipulating images (evasion) or training data (poisoning). Experiments were conducted on MNIST and CIFAR-10 datasets. |

**3.3.2. Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbour**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Hayes & Danezis [58] | Blackbox | Evasion | Targeted | Both False Positives and False Negatives | Launched blackbox attacks against various models (Random Forest, SVM, K-Nearest Neighbor) hosted on Amazon. They generated adversarial examples by manipulating images from the MNIST dataset and tested their effect on the classifier. |

**3.3.3. Artificial Neural Network (ANN)**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Shaham et al. [15] | Whitebox | Poisoning | Indiscriminate | Both False Positives and False Negatives | Proposed a training framework to increase the stability and robustness of artificial neural networks against adversarial examples.  The adversary attacks the classifier by creating perturbations to the input data during training.  |

**3.3.4. Convolutional Neural Networks (CNNs)**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Radford et al. [67] | Graybox to Whitebox | Both Poisoning and Evasion | Indiscriminate | Both False Positives and False Negatives, Clustering Accuracy Reduction | Introduced deep convolutional generative adversarial networks (DCGANs) and used trained discriminators for image classification tasks. They showed DCGANs can achieve competitive performance with unsupervised algorithms, proving their effectiveness even in a clustering setting. |
| Schottle et al. [68] | Graybox | Evasion | Indiscriminate | False Positives | Combined methods from multimedia forensics to bring robustness to CNNs against adversarial examples.  The attacker uses Projected Gradient Descent to create adversarial examples. |
| Chivukula et al. [69] | Blackbox | Poisoning | Targeted | False Negatives |  Proposed a genetic algorithm for deep learning on CNNs, formulated as a two-player zero-sum Stackelberg game.  The attacker (with no prior knowledge) targets the learner with data generated using genetic operators, aiming to alter labels. The learner adapts by retraining the CNN weights. |
| Madry et al. [70] | Various (Blackbox to Whitebox) | Both Poisoning and Evasion | Indiscriminate | Both False Positives and False Negatives | Studied the robustness of deep learning algorithms against adversaries. Their theoretical and experimental results showed that networks robust against Projected Gradient Descent (PGD) adversaries are also robust against a broader range of attacks. |
| Athalye et al. [72] | Whitebox | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives | Investigated obfuscated gradients in neural networks.  They showed that many defenses intentionally or unintentionally cause gradient descent to fail, making the models vulnerable. |
| Dong et al. [73] | Blackbox | Evasion | Both Targeted and Indiscriminate | False Negatives | Proposed momentum iterative gradient-based methods to generate adversarial examples that successfully fool robust classification models. |
| Jeong et al. [74] | Blackbox | Poisoning | Indiscriminate | Both False Positives and False Negatives | Focused on the impact of adversarial samples on multimedia video surveillance using deep learning.  They injected adversarial samples into the training of autoencoder and CNN models. |

**3.3.5. Generative Adversarial Networks (GANs)**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Springenberg [71] | Graybox | Poisoning | Targeted | False Negatives | Studied the problem of an adversarial generative model regularizing a discriminatively trained classifier. Proposed an algorithm based on generative adversarial networks (GANs) to provide robustness against an optimal adversary. |

**3.3.6. Deep Learning**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Biggio et al. [59] | Whitebox | Poisoning | Both Targeted and Indiscriminate | Clustering Accuracy Reduction |  Studied obfuscation and poisoning attacks against clustering algorithms, focusing on handwritten digit recognition. |
| Szegedy et al. [17] | Graybox | Poisoning | Indiscriminate | Both False Positives and False Negatives |  Showed that DNNs are vulnerable to adversarial examples. They found small perturbations that can cause misclassification. |
| Goodfellow et al. [60] | Whitebox | Poisoning | Targeted | Both False Positives and False Negatives | Demonstrated the vulnerability of neural networks and DNNs to adversarial perturbation. |
| Goodfellow et al. [61] | Whitebox | Evasion | Indiscriminate | False Negatives |  Proposed "adversarial nets" for deep learning, which leverages a min-max two-player game. The attacker aims to generate samples that are misclassified as legitimate. |
| Nguyen et al. [16] | Blackbox to Graybox | Evasion | Indiscriminate | False Negatives | Proved that DNNs can be fooled by adversarial examples using evolutionary algorithms and gradient ascent. |
| Papernot et al. [63] | Blackbox | Evasion | Indiscriminate | False Negatives |  Highlighted the importance of adversarial sample transferability in DNNs. |
| Papernot et al. [64] | Blackbox | Evasion | Indiscriminate | False Negatives |  Provided a method to craft adversarial samples using a substitute DNN, exploiting the transferability of adversarial examples. |
| Kurakin et al. [65] | Various (Blackbox to Whitebox) | Both Poisoning and Evasion | Indiscriminate | Both False Positives and False Negatives, Clustering Accuracy Reduction |  Injected adversarial examples into the training set using adversarial training methodology, demonstrating transferability between models. |
| Hayes & Danezis [62] | Whitebox | Poisoning | Both Targeted and Indiscriminate | False Negatives | Investigated universal adversarial networks, which can fool a target classifier when generated output is added to a clean sample. |
| Evtimov et al. [66] | Whitebox | Evasion | Targeted | Both False Positives and False Negatives |  Introduced "Robust Physical Perturbations," which can fool road sign recognition systems using DNNs. |

## Other Applications
**3.4.1. Reinforcement Learning**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Uther et al. [76] | Graybox | Evasion | Targeted | False Negatives | Created a framework (two-player hexagonal grid soccer) to evaluate algorithms in multi-agent reinforcement learning. The attacker observes the opponent's moves and aims to increase the false negative rate of the attacked classifier. |

**3.4.2. Collaborative Filtering**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Guillory & Bilmes [77] | Graybox | Poisoning | Targeted | Clustering Accuracy Reduction | Studied a submodular set cover problem where a set of movies is suggested to a user. The attacker injects adversarial noise during query learning, aiming to cause clustering accuracy reduction and poor recommendations. |

**3.4.3. Recurrent Neural Networks (RNNs)**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Papernot et al. [18] | Graybox | Evasion | Indiscriminate | Both False Positives and False Negatives | Proposed an attack model for misclassifying outputs of RNNs by crafting adversarial sequences during test time. They demonstrated that altering a small percentage of words in movie reviews can cause 100% misclassification. |

**3.4.4. Autoregressive Forecasting Models**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Alfeld et al. [78] | Graybox | Poisoning | Indiscriminate | Both False Positives and False Negatives | Presented a mathematical framework of an attack method against autoregressive forecasting models.  The attacker aims to augment initial values so that the forecasted values are as close as possible to a desired value. |

**3.4.5. Game Theory**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Zeager et al. [79] | Whitebox | Evasion | Indiscriminate | False Negatives | Modeled attacks in credit card fraud detection systems as a game, where the adversary aims to modify transaction attributes to make fraudulent charges undetectable. |

**3.4.6. Deep Learning**

| Reference | Attacker Knowledge | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|
| Shokri et al. [80] | Blackbox | Evasion | Indiscriminate | False Positives | Demonstrated an inference membership attack, where the adversary tries to determine if specific data records were used to train a model. |
| Biggio et al. [81] | Whitebox | Poisoning | Targeted | False Positives |  Proposed two face recognition attacks that demonstrate how to poison biometric systems to impersonate a targeted client. |
| Sharif et al. [82] | Graybox to Whitebox | Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives |  Demonstrated techniques against facial biometric systems using inconspicuous accessories like glasses frames to deceive the classifier. |

## Multipurpose

Adversarial attacks across multiple application domains.
**3.5.1. Naive Bayes - Principal Component Analysis**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Huang et al. [10] |  Various (Blackbox to Whitebox) |  Both Naive Bayes and PCA | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives, Clustering Accuracy Reduction |  One of the first papers to explore adversarial attacks against machine learning.  They demonstrated attacks against spam detection and network anomaly detection. |

**3.5.2. Support Vector Machine (SVM)**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Biggio et al. [32] | Graybox | Support Vector Machine (SVM) | Poisoning | Targeted | Both False Positives and False Negatives |  Explored a poisoning attack against the SVM algorithm. The attacker aims to find a specific point that, when added to the training dataset, maximally decreases the classification accuracy. |
| Zhou et al. [83] | Graybox to Whitebox | Support Vector Machine (SVM) | Evasion | Both Targeted and Indiscriminate | False Negatives |  Studied two evasion attack models in the context of spam email detection and credit card fraud detection. |
| Demontis et al. [84] | Graybox to Whitebox |  Linear Classifiers | Evasion | Targeted | False Negatives | Investigated the vulnerability of linear classifiers to evasion attacks in various domains, including handwritten digit classification, spam filtering, and PDF malware detection.  |
| Zhang et al. [85] | Graybox to Whitebox |  Linear Classifiers | Evasion | Indiscriminate | False Negatives |  Proposed two algorithms for feature selection against adversaries.  The goal is to maximize the generalization capability of the linear classifier and its robustness to evasion attacks. |

**3.5.3. Support Vector Machine (SVM) - Logistic Regression**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Biggio et al. [86] | Whitebox | Both SVM and Logistic Regression | Evasion | Both Targeted and Indiscriminate | False Negatives | Assumed that adversaries can launch evasion attacks against classifiers in various applications, aiming to avoid detection of their activities by manipulating the distribution of training and testing data. |
| Mei et al. [87] | Whitebox | Both SVM and Logistic Regression | Poisoning | Indiscriminate | False Negatives |  Presented an efficient solution for poisoning attacks against machine learners, focusing on spam email detection and wine quality modeling systems. |

**3.5.4. Support Vector Machine (SVM) - Multiple Classifier System (MCS)**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Biggio et al. [88] | Graybox to Whitebox |  Both SVM and MCS | Evasion | Targeted | False Negatives |  Created a "one-and-a-half-class" classifier to achieve a good trade-off between classification accuracy and security.  They tested the approach against real-world data in spam detection and PDF malware detection. |

**3.5.5. Game Theory**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Liu et al. [89] | Graybox | Various (including SVM) | Evasion | Indiscriminate | False Negatives | Modeled the interaction between a data miner and an adversary using a one-step game-theoretic model. The adversary aims to minimize the difference between distributions of positive and negative classes while also minimizing their own movement. |
| Bulo et al. [90] | Graybox | Various (including SVM) | Evasion | Indiscriminate | False Negatives | Extended the work of Bruckner et al. [48] on static prediction games by introducing randomization.  They demonstrated that classifiers can be learned by the adversary when they deviate from their hypothesized actions. |
| Alpcan et al. [92] | Whitebox | Support Vector Machine (SVM) | Poisoning | Indiscriminate | False Negatives |  Used game theory to provide adversarial machine learning strategies based on linear SVMs.  The attacker injects data to maximize the error rate of the classifier. |

**3.5.6. Deep Learning**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Munoz et al. [91] | Various (Graybox to Whitebox) |  Deep Learning | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives |  Proposed a poisoning attack model that maximizes misclassification performance in multiclass classification settings. |
| Gonzalez et al. [93] | Various (Graybox to Whitebox) |  Deep Learning | Both Poisoning and Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives | Proposed a poisoning algorithm based on back-gradient optimization, aiming to reduce the complexity of attacks. |
| Wang et al. [20] | Various (Blackbox to Whitebox) |  Deep Learning | Poisoning | Targeted | False Positives |  Proposed an adversary-resistant model to construct robust DNNs, employing a random feature nullification technique. |
| Cisse et al. [97] | Graybox |  Deep Learning | Targeted | False Negatives |  Used Parseval networks as a regularization method to reduce the sensitivity of DNNs to adversarial noise. |

**3.5.7. Generative Adversarial Networks (GANs)**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Li et al. [94] | Whitebox |  Generative Adversarial Networks (GANs) | Poisoning | Indiscriminate | Clustering Accuracy Reduction |  Focused on Generative Adversarial Networks (GANs) in various applications, such as image translation, speech synthesis, and robot trajectory prediction. The adversary aims to generate samples that are similar to the data distribution. |
| Louppe & Cranmer [95] | Graybox to Whitebox |  Generative Adversarial Networks (GANs) | Poisoning | Indiscriminate | Clustering Accuracy Reduction |  Focused on computer simulations and used data generation to assist various scientific fields.  The adversary launches poisoning attacks to reduce the clustering accuracy of the system. |

**3.5.8. Support Vector Machine (SVM) - Deep Learning**

| Reference | Attacker Knowledge | Algorithm Targeted | Attack Type | Attack Specificity | Performance Impact | Description |
|---|---|---|---|---|---|---|
| Bhagoji et al. [96] | Whitebox | Both SVM and Deep Learning | Evasion | Both Targeted and Indiscriminate | Both False Positives and False Negatives |  Focused on image recognition and human activity recognition.  They used linear transformation as a data defense against evasion attacks.  |

# Discussion and conclusions

**4.1 Distribution of Papers Per Taxonomy Phase**

* This section analyzes the frequency of research papers focusing on each feature of the proposed taxonomy. This provides insights into the research landscape and potentially identifies areas that require further investigation. 
* **Key findings:** 
    * Whitebox attacks are most commonly studied, as researchers often begin with the most informed attacker scenario.
    * Blackbox attacks are also well-researched, showing an awareness of scenarios where the attacker has limited knowledge.
    * A surprisingly high number of papers utilize game theory, highlighting the increasing recognition of its value in modeling strategic interactions between defenders and attackers.
    * Indiscriminate attacks are frequently studied, as they are more likely to be successful due to their broader scope. 
    * Experimental evaluations dominate, indicating a preference for practical testing of attack methods.
    * False negatives are most frequently measured, reflecting the focus on preventing malicious actions from being undetected.

**4.2 Open Problems**

* This section identifies several areas where further research is needed to enhance the security of machine learning systems against adversarial attacks.
* **Key open problems:**
    * **Creating immunity to perturbations:**  Developing defenses that can effectively counter various adversarial perturbation methods is crucial.
    * **Multiple adversaries:**  Research needs to consider the impact of attacks launched by multiple, potentially collaborating adversaries.
    * **Randomized adversarial approaches:** Determining the optimal level of randomization in training data to enhance robustness against attacks remains an open challenge.
    * **Digital forensics for adversarial machine learning:**  Developing forensic techniques to analyze and detect adversarial behavior is critical for security and law enforcement.
