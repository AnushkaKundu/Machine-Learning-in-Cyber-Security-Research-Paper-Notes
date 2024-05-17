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

