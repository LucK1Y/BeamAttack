<!-- TextTrojaners at  -->
# CheckThat! 2024: Robustness of Credibility Assessment with Adversarial Examples through BeamAttack 

## Table of Contents
* [Introduction](#introduction)
* [Repository Organization](#repository-organization)
* [Usage](#Usage)
* [Contact](#contact)
* [Citation](#citation)

## Introduction
This repository contains the code for the paper [TextTrojaners at CheckThat! 2024: Robustness of Credibility Assessment with Adversarial Examples through BeamAttack]. The paper describes the submission of TextTrojaners for the [CheckThat! 2024 lab task 6: Robustness of Credibility Assessment with Adversarial Examples](https://checkthat.gitlab.io/clef2024/task6/) at the [Conference and Labs of the Evaluation Forum 2024](https://clef2024.imag.fr/) in Grenoble.


#### About the shared Task
Many social media platforms employ machine learning for content filtering, attempting to detect content that is misleading, harmful, or simply illegal. For example, imagine the following message has been classified as harmful misinformation:

`Water causes death! 100%! Stop drinking now! #NoWaterForMe`

However, would it also be stopped if we changed ‘causes’ to ‘is responsible for’? Or ‘cuases’, ‘caυses’ or ‘Causes’? Will the classifier maintain its accurate response? How many changes do we need to make to trick it into changing the decision? This is what we aim to find out in the shared task.

For more information please refer to the shared task description here [CheckThat! 2024 lab task 6: Robustness of Credibility Assessment with Adversarial Examples](https://checkthat.gitlab.io/clef2024/task6/).

#### Why BeamAttack
Our approach BeamAttack is a novel algorithm for generating adversarial examples in natural language processing
through the application of beam search. To further improve the search process, we integrate a semantic filter that
prioritizes examples with the highest semantic similarity to the original sample, enabling early termination of the
search. Additionally, we leverage a model interpretability technique, LIME, to determine the priority of word
replacements, along with existing methods such as that determine word importance through the model’s logits.
Our approach also allows for skipping and removing words, enabling the discovery of minimal modifications that
flip the label. Furthermore, we utilize a masked language model to predict contextually plausible alternatives to the
words to be replaced, enhancing the coherence of the generated adversarial examples. BeamAttack demonstrates
state-of-the-art performance, outperforming existing methods with scores of up to 0.90 on the BiLSTM, 0.84 on
BERT, and 0.82 on the RoBERTa classifier.

## Repository Organization

This repository is structured into three primary directories:

* `beam_attack/`: This directory contains the implementation of our attack algorithm.
* `BODEGA/`: This [sub repository](https://github.com/piotrmp/BODEGA) houses the evaluation functionality.
* `clef2024-checkthat-lab/`: This [sub repository](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab) stores the datasets and target models utilized in this project.

Additionally, we would like to draw attention to the following key file:

* `beamattack.ipynb`: This Jupyter Notebook provides an exemplary demonstration of how our algorithm can be employed in an OpenAttack setup, with a focus on evaluating the BODEGA score. This example is based on the template presented in the shared task, which can be accessed at [this link](https://colab.research.google.com/drive/1zxjwiztRLILFUjw5jR5xyL398bNSx8TI?usp=sharing). We used this file to create our submission of the shared task.

The directory structure is as follows:
```bash
.
├── beam_attack/
├── BODEGA/
├── clef2024-checkthat-lab/
|
└── beamattack.ipynb
```

## Usage
#### Installation of BeamAttack, evaluation methods and datasets
Clone this repository with all its subrepositories:
```bash
git clone --recurse-submodules https://github.com//ml4nlp2-adversarial-attack
```

In case the submodules where not clones automatically, it is possible to do with:
````bash
# cd ml4nlp2-adversarial-attack/
git submodule update --init --recursive
````

<!-- Then install the necessary packages:
````bash
# cd ml4nlp2-adversarial-attack/
pip install -r requirements.txt
```` -->

## Contact 
The author emails are {arnisa.fazla, david.guzmanpiedrahita, lucassteffen.krauter}@uzh.ch

Please contact david.guzmanpiedrahita@uzh.ch first if you have any questions.

## Citation 
note: add after camera ready submission
