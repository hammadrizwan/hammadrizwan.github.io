---
layout: page
title: Resolving Lexical Bias in Model Editing
description: A weights preserving model editing approach using Projector Editor Network (PENME). 
img: assets/img/PENME.jpg
importance: 1
category: work
related_publications: true
---

### Summary:
Model editing aims to modify the outputs of large language models after they are trained. Previous approaches have often involved direct alterations to model weights, which can result in model degradation. Recent techniques avoid making modifications to the model’s weights by using an adapter that applies edits to the model when triggered by semantic similarity in the representation space. We demonstrate that current adapter methods are critically vulnerable to strong
lexical biases, leading to issues such as applying edits to irrelevant prompts with overlapping words. This paper presents a principled approach to learning a disentangled representation space that facilitates precise localization of edits by maintaining distance between irrelevant prompts while preserving proximity among paraphrases. In our empirical study, we show that our method (Projector Editor Networks for Model Editing - PENME) achieves state-of-the-art model editing results while being more computationally efficient during inference than previous methods and adaptable across different architectures. {% cite Rizwan25Editing %}

### Overview
Our research shows that performance of weight-preserving meth-ods is heavily reliant on scoping mechanism which suffers from a critical vulnerability of Lexical bias Figure 1, prompts with similar lexical tokens but different semantics that are closer together in the representation space compared to a prompt and its respective paraphrases. Lexical bias prevents current adapter-based methods from effectively being able to balance generalization to unseen paraphrases and “misfiring” on semantically dissimilar (irrelevant) prompts.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/lexical_domince_visual_figure1.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Figure 1. An illustration of lexical bias in embeddings: a) a low similarity threshold (illustrated with the circle) results in failing to edit
paraphrases. b) A high similarity threshold results in misfires with irrelevant prompts. c) illustrates our solution which restructures the
representation space.
</div>

To examine the lexical bias of representations, we randomly sampled 500 entries from the Counterfact dataset. For each entry, we created triplets consisting of an edit prompt, a randomly sampled paraphrase prompt and an irrelevant prompt with $$\textbf{high lexical overlap}$$. These triplets are fed into various models, and representation vectors ($$\vec{x_{i}},\vec{p_{i}},\vec{p^{\neg}_{i}}$$) from the feed-forward block of each layer $$l$$ are extracted.  We select either averaged token representations or dedicated sentence representations, based on whether a given model offers a specific token for sentence-level representation. We calculate two sets of pairwise Euclidean distances (1) Between edit representations and paraphrase representations $$\|\|\vec{x_{i}}-\vec{p_{i}}\|\|_2 $$ (2) Between edit representations and irrelevant prompts representations $$\|\|\vec{x_{i}}-\vec{p^{\neg}_{i}}\|\|_2 $$. We then compare these distances to determine if irrelevant prompts are closer to the edits than the paraphrases $$\|\|\vec{x_{i}} -\vec{p_{i}}\|\|_2 > \|\|\vec{x_{i}} -\vec{p^{\neg}_{i}}\|\|_2$$. Figure 2 displays the percentage of samples where irrelevant prompts $$\textit{were closer}$$ to the edits.


<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/lexical_dominance_models_full.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


<div class="caption">
Figure 2.Percentage of samples where edits are closer to irrelevant prompts as compared to paraphrases in the representations space of
different models across all layers. T5-small, GPT2-XL and Llama-2-7b have 6, 32, 48 layers, respectively.
</div>

To resolve this issue we propose PENME, a model editing framework that learns a projection network that maps the model’s representation space to a new representation space where lexical bias is minimized. We integrate our projection network in an adapter-based retrieval scheme for model editing, demon- strating, for the first time in adapter-based approaches, high efficacy in both paraphrase execution (generalization) and prevention of misfires on irrelevant prompts (locality).

PENME, illustrated in Figure 3, consists of two components: (1) $$\textbf{Projection Network (g)}$$ projects model activations denoted $h_l(input)$ at layer $l$ into a distinct representation space $g(h_l(input))$. (2) \textbf{Key-Value Codebook} stores the projected model activations $g(h_l(input))$ at layer $l$  as keys and corresponding values containing a learned similarity threshold ($\delta$) and the new associated output information $y_i$. This paper only considers storing strings as $$y_i$$, but vectors or LoRA block indices can also be stored as values, which facilitate playback approaches.


<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/PENME.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">
Figure 3. PENME uses a projection network that interfaces with the pointwise feed-forward layer output in a transformer block. The
projection network, coupled with key-value codebook storage, acts as a scoping mechanism by comparing projection outputs with
codebook entries. This determines whether the current input relates to a specific edit or should pass through the model unmodified.
</div>




### Projector Training
The project consists of two layer neural network with non-linearity in between much like pointwise feed forward layer in the transformer mode. Our training loss is inspired by contrastive learning and is defined by the following loss function:

$$
\begin{aligned}
\mathcal{L}(\vec{x_i}, \vec{z}) &= (1-t)\,\tfrac{1}{2}\lVert \vec{x_i} - \vec{z} \rVert_2^2 \\
&\quad + t \,\tfrac{1}{2} \big[\max(0, m - \lVert \vec{x_i} - \vec{z} \rVert_2)\big]^2, \\
t &=
\begin{cases}
1, & \text{if } \vec{z} \gets \vec{p_{ij}}, \\
0, & \text{if } \vec{z} \gets \vec{p^{\neg}_{ij}} \lor \vec{x_l}.
\end{cases}
\end{aligned}
$$

where $t$ is the target $\{0,1\}$ which is 0 when the training pair is $$\{x_i,p_{ij}\}$$ (edit, paraphrase) and 1 when the training pair is $$\{x_i,p^{\neg}_{ij}\}$$ (edit, irrelevant) or the inter-edit (or edit-to-edit) pair $$\{x_i,x_l\}$$ where we sample an unrelated edit, $$m$$ is the margin which pushes $$\vec{p^{\neg}_{ij}}$$ at least $m$ distance away from $$\vec{x_{i}}$$. The projection network is trained such that for all samples in a dataset, edits $x_i$ and edit paraphrases $$p_{ij}$$ are close together while edits $$x_i$$ and irrelevant $p^{\neg}_{ij}$ paraphrases or unrelated edits  $$x_l$$  are pushed apart in the projection space. Training is performed by sampling pairs at random. Note that $$\vec{z}$$ is a variable that is assigned either a paraphrase, an irrelevant prompt, or an unrelated edit just as a way to make the loss function more concise.

The results presented in Figure 4 demonstrate that the projector network effectively learns to distance lexically similar but unrelated irrelevant prompts in comparison to paraphrases.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Lexical_Dominance_projector.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="caption">
Figure 4. Projector networks mitigate lexical bias: a critical problem in adapter-based model editing techniques. Percentage of samples where irrelevant but lexically similar prompts are closer than semantically similar paraphrases in the representation space before and after our learned projection (PENME).
</div>

### Results
We assess the performance of PENME across a spectrum of transformer-based LLMs, including T5, GPT2-XL and Llama-2-7b in the zsRE and Counterfact datasets. For comparitive performance to relevant literature please refer to the paper pdf at the bottom.

<table style="width:100%; border-collapse: collapse; text-align:center;">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Model</th>
      <th colspan="4">Counterfact</th>
      <th colspan="4">zsRE</th>
    </tr>
    <tr>
      <th>ES</th><th>Loc</th><th>Para</th><th>Score</th>
      <th>ES</th><th>Loc</th><th>Para</th><th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>PENME</td><td>T5-small</td><td>1.000</td><td>0.787</td><td>0.808</td><td>0.865</td><td>1.000</td><td>0.941</td><td>0.913</td><td>0.951</td></tr>
    <tr><td>PENME</td><td>Llama-2-7b</td><td>1.000</td><td>0.869</td><td>0.906</td><td>0.925</td><td>1.000</td><td>0.987</td><td>0.966</td><td>0.984</td></tr>
    <tr><td>PENME</td><td>GPT2-XL</td><td>1.000</td><td>0.847</td><td>0.875</td><td>0.907</td><td>1.000</td><td>0.957</td><td>0.940</td><td>0.966</td></tr>
    <tr><td>PENME<sub>stream</sub></td><td>T5-small</td><td>1.000</td><td>0.782</td><td>0.756</td><td>0.846</td><td>1.000</td><td>0.615</td><td>0.550</td><td>0.721</td></tr>
    <tr><td>PENME<sub>stream</sub></td><td>Llama-2-7b</td><td>1.000</td><td>0.871</td><td>0.818</td><td>0.896</td><td>1.000</td><td>0.716</td><td>0.792</td><td>0.836</td></tr>
    <tr><td>PENME<sub>stream</sub></td><td>GPT2-XL</td><td>1.000</td><td>0.850</td><td>0.768</td><td>0.872</td><td>1.000</td><td>0.733</td><td>0.768</td><td>0.833</td></tr>
  </tbody>
</table>

### Finding Hyperparameter and Scaling Edits

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/generalization_vs_locality_threshold_main.png" title="Figure 1" class="img-fluid rounded z-depth-1" %}
    </div>
    
</div>
<div class="caption">
    Figure 5. Shows the trade-off between generalization and locality performance across different hyperparameter settings. The distance threshold τ varies from 0.01 to 0.2 (0.01 increments and τ is normalized by 100), while the edit-pairing similarity threshold ϕ ranges from 0.5 to 0.9 (0.1 increments). Higher ϕ values enforce stricter edit similarity requirements. The results showcase the effect of hyperparameter tuning on the projector network’s learning capacity and overall performance.
</div>


<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/penme_ablation_samples.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>
