---
layout: page
title: Resolving Lexical Bias in Model Editing
description: A weights preserving model editing approach using Projector Editor Network (PENME). 
img: assets/img/PENME.jpg
importance: 1
category: Academic
related_publications: true
---

### Summary:
Model editing aims to modify the outputs of large language models after they are trained. Previous approaches have often involved direct alterations to model weights, which can result in model degradation. Recent techniques avoid making modifications to the model’s weights by using an adapter that applies edits to the model when triggered by semantic similarity in the representation space. We demonstrate that current adapter methods are critically vulnerable to strong
lexical biases, leading to issues such as applying edits to irrelevant prompts with overlapping words. This paper presents a principled approach to learning a disentangled representation space that facilitates precise localization of edits by maintaining distance between irrelevant prompts while preserving proximity among paraphrases. In our empirical study, we show that our method (Projector Editor Networks for Model Editing - PENME) achieves state-of-the-art model editing results while being more computationally efficient during inference than previous methods and adaptable across different architectures. {% cite Rizwan25Editing %}

### Overview
Our research shows that performance of weight-preserving methods is heavily reliant on scoping mechanism which suffers from a critical vulnerability of Lexical bias Figure 1, prompts with similar lexical tokens but different semantics that are closer together in the representation space compared to a prompt and its respective paraphrases. Lexical bias prevents current adapter-based methods from effectively being able to balance generalization to unseen paraphrases and “misfiring” on semantically dissimilar (irrelevant) prompts.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/lexical_domince_visual_figure1.png" title="Lexical dominance visual" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
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
        {% include figure.liquid loading="lazy" path="assets/img/lexical_dominance_models_full.png" title="Lexical dominance model representations" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
</div>


<div class="caption">
Figure 2.Percentage of samples where edits are closer to irrelevant prompts as compared to paraphrases in the representations space of
different models across all layers. T5-small, GPT2-XL and Llama-2-7b have 6, 32, 48 layers, respectively.
</div>

To resolve this issue we propose PENME, a model editing framework that learns a projection network that maps the model’s representation space to a new representation space where lexical bias is minimized. We integrate our projection network in an adapter-based retrieval scheme for model editing, demonstrating, for the first time in adapter-based approaches, high efficacy in both paraphrase execution (generalization) and prevention of misfires on irrelevant prompts (locality).

PENME, illustrated in Figure 3, consists of two components: (1) $$\textbf{Projection Network (g)}$$ projects model activations denoted $h_l(input)$ at layer $l$ into a distinct representation space $g(h_l(input))$. (2) $\textbf{Key-Value Codebook}$ stores the projected model activations $g(h_l(input))$ at layer $l$  as keys and corresponding values containing a learned similarity threshold ($\delta$) and the new associated output information $y_i$. This paper only considers storing strings as $$y_i$$, but vectors or LoRA block indices can also be stored as values, which facilitate playback approaches. $(\textit{In-context based generation is explored in the paper})$.


<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/PENME.png" title="PENME Overview" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
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

The inherent lexical and semantic similarities among edits increase the probability of certain edit paraphrases exhibiting greater proximity to other unrelated edits. This phenomenon can lead to erroneous paraphrase-edit associations during execution, potentially triggering inappropriate edit operations. This is why we also push unrelated edits farther away in Eq. 1 as well as unrelated prompts.

The results presented in Figure 4 demonstrate that the projector network effectively learns to distance lexically similar but unrelated irrelevant prompts in comparison to paraphrases.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/Lexical_Dominance_projector.png" title="projector result" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
</div>

<div class="caption">
Figure 4. Projector networks mitigate lexical bias: a critical problem in adapter-based model editing techniques. Percentage of samples where irrelevant but lexically similar prompts are closer than semantically similar paraphrases in the representation space before and after our learned projection (PENME).
</div>

### Results
We assess the performance of PENME across a spectrum of transformer-based LLMs, including T5, GPT2-XL and Llama-2-7b in the zsRE and Counterfact datasets. $(\textit{For comparitive performance to relevant literature please refer to the paper pdf at the bottom.})$

PENEME is evaluated under two experimental settings. The first is a batch evaluation, where performance is measured on the held-out test split of the training data—specifically on test paraphrases and irrelevant prompts. The second setting assesses projector generalization in a stream, or lifelong editing, scenario. In this we evaluate zero-shot generalization regime, the codebook is updated once per edit while keeping the projector frozen. Since a trained projection network is required, PENEME-stream is initialized with 2,000 previously unseen samples from the Counterfact dataset. For zsRE, we evaluate cross dataset zero-shot generalization. The results are shown in Table 1, which indicate that the projector achieves robust performance while maintaining a balance between generalization and locality.

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
<div class="caption">
    Table 1. A comparative analysis of PENME and recent model editing methods on 2000 edits from the Counterfactual dataset and 1000 edits on zsRE. The metrics are Edit Success (ES), Locality (Loc) and Paraphrase Generalization (Para).
</div>
### Finding Hyperparameter
To demonstrate the trade-off between generalization and locality, we conducted an ablation study by varying the τ parameter, which modulates the similarity threshold defining an edit’s scope. Figure 5 presents the results for GPT2-XL and T5-small. 
<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/generalization_vs_locality_threshold_main.png" title="Hyperparamter tuning" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
    
</div>
<div class="caption">
    Figure 5. Shows the trade-off between generalization and locality performance across different hyperparameter settings. The distance threshold τ varies from 0.01 to 0.2 (0.01 increments and τ is normalized by 100), while the edit-pairing similarity threshold ϕ ranges from 0.5 to 0.9 (0.1 increments). Higher ϕ values enforce stricter edit similarity requirements. The results showcase the effect of hyperparameter tuning on the projector network’s learning capacity and overall performance.
</div>

### Scaling Edits
We evaluate the projection network’s stability under varying numbers of edits using incrementally larger training sets ranging from 1000 to 5000 edits, with 1000-edit increments per training session. The results of the experiment are shown in Figure 6. 
<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="lazy" path="assets/img/penme_ablation_samples.png" title="Scaling PENME" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
</div>
<div class="caption">
    Figure 6. PENME's performance in terms of Locality (dotted) and Generalization (continuous line) across varying numbers of edits
</div>


$\textit{Downstream performance analysis and long form generation can be found in section 8,9 of the main paper text and appendix section H.}$
### Conclusion
In this paper, we raised awareness of a critical vulnerability in weight-preserving adapter-based model editing techniques: lexical bias in the representation space. We developed a projection-based method PENME trained via contrastive learning to disentangle lexical and semantic similarity which originally would cause misfiring on irrelevant prompts with a high lexical overlap. Empirical evaluations showed PENME’s superior performance across varying levels of task complexity. On the zsRE dataset, it achieved impressive generalization and locality scores exceeding 0.90, demonstrating that our method is satisfactorily able to balance generalization and locality using distance metrics in this new projected space. Notably, when assessed on the more challenging Counterfact benchmark, the system maintained robust performance, attaining scores above 0.80 for both generalization and locality metrics. This performance on Counterfact is particularly significant given the benchmark’s increased difficulty, underscoring PENME’s efficacy. In future work, we aim to investigate whether a projector pre-
trained on a large-scale dataset can serve as a plug-and-play component for cross-lingual generalization. Additionally,
we plan to explore whether the projector can be trained and updated incrementally with new edits, thereby reducing training overhead and improving scalability.
