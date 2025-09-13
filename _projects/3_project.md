---
layout: page
title: Hate-Speech and Offensive Language Detection in Roman Urdu
description: The first extensive work on data curation and evaluation for hate speech detection in Roman Urdu.
img: assets/img/hatespeech.jpg
importance: 3
category: Academic
related_publications: true
---

### Summary:
The task of automatic hate-speech and offensive language detection in social media content is of utmost importance due to its implications in unprejudiced society concerning race, gender, or religion. Existing research in this area, however, is mainly focused on the English language, limiting the applicability to particular demographics. Despite its prevalence, Roman Urdu (RU) lacks language resources, annotated datasets, and language models for this task. In this study, we: (1) Present a lexicon of hateful words in RU, (2) Develop an annotated dataset called RUHSOLD consisting of 10, 012 tweets in RU with both coarse-grained
and fine-grained labels of hate-speech and offensive language, (3) Explore the feasibility of transfer learning of five existing embedding models to RU, (4) Propose a novel deep learning architecture called CNN-gram for hatespeech and offensive language detection and compare its performance with seven current baseline approaches on RUHSOLD dataset, and (5) Train domain-specific embeddings on more than 4.7 million tweets and make them publicly available. We conclude that transfer learning is more beneficial as compared to training embedding from scratch and that the proposed model exhibits greater robustness as compared to the baselines. {% cite RizwanSK20 %}

### Overview
The task of automatic hate-speech and offensive language detection in social media content is of utmost importance due to its implications in unprejudiced society concerning race, gender, or religion. Existing research in this area, however, is mainly focused on the English language, limiting the applicability to particular demographics. Despite its prevalence, Roman Urdu (RU) lacks language resources, annotated datasets, and language models for this task. In this study, we:

- First, we provide a lexicon base of 621 hateful words for the RU language.
- Second, we develop a gold-standard dataset, called Roman Urdu Hate-Speech and Offensive Language Detection (RUHSOLD), from tweets in RU with binary coarse-grained as well as multi-class fine-grained labels.
- Third, we explore the transfer learning capabilities of five existing multilingual embedding models to RU language through extensive experiments.
- Fourth, we propose a novel deep learning model called Convolutional Neural Network n-gram (CNN-gram) and compare its performance with seven baseline models on the RUHSOLD dataset. In our presentation, we demonstrate that CNN-gram displays a greater robustness across both coarse-grained as well as fine-grained classification tasks.
- Fifth, to exhibit contrast with transfer learning of embedding models, we train domainspecific embeddings called “RomUrEm” on

### Dataset

First we construst our own lexicon of hateful words (by searching for such keywords online and interviewing people). this lexicon consists of abusive and derogatory terms along with slurs or terms pertaining to religious hate and sexist language. Using this lexicon along with a separate collection of RU common words, we search and collect $20, 000$ tweets and perform a manual preliminary analysis to find new slang, abuses, and identify frequently occurring common terms. The choice to add common RU words is made in order to extract random inoffensive tweets and the tweets that are offensive but do not contain any offensive words.

Using this updated lexicon we search and collect $50, 000$ new tweets. From this updated tweet base, around $10, 000$ tweets are randomly sampled for annotations. To avoid issues related to user distribution bias we restrict a maximum of 120 tweets per user.

The dataset is annotated for two sub-tasks. First sub-task is based on binary labels of \"Hate-Offensive\" content and \"Normal/Neutral content\" (i.e., inoffensive language). These labels are self-explanatory. We refer to this sub-task as “coarse-grained classification”. Second sub-task defines Hate-Offensive content with four labels at a granular level. These labels are the most relevant for the demographic of users who converse in RU and are defined in related literature. We refer to this sub-task as “fine-grained classification”. The objective behind creating two sub-tasks is to enable the researchers to evaluate the hatespeech detection approaches on both easier (coarsegrained) and challenging (fine-grained) scenarios. All labels and their definitions are summarized as follows: 
- Abusive/Offensive: Profanity, strongly impolite, rude or vulgar language expressed with fighting or hurtful words in order to insult a targeted individual or group.
- Sexism: Language used to express hatred towards a targeted individual or group based on gender or sexual orientation.
- Religious Hate: Language used to express hatred towards a targeted individual or group based on their religious beliefs or lack of any religious beliefs and the use of religion to incite violence or propagate hatred against a targeted individuals or group.
- Profane: The use of vulgar, foul or obscene language without an intended target.
- Normal: This contains text that does not fall into the above categories.

Samples with translations are provided in Table 1 and the dataset statistics are provided in Table 2. 
<div style="text-align:center; margin-bottom:10px; font-weight:bold; color:red;">
⚠️ Warning: The following table contains offensive / explicit language (hover to show)
</div>

<div style="filter: blur(6px); transition: filter 0.3s;" 
     onmouseover="this.style.filter='blur(0px)';" 
     onmouseout="this.style.filter='blur(6px)';">

<table style="width:100%; border-collapse: collapse; text-align:left;">
  <thead>
    <tr>
      <th>Tweet</th>
      <th>Translation</th>
      <th>Target Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>randi ke bache tu apne hashar ki fikar kar</td>
      <td>you son of a prostitute, you should worry for what will happen to you.</td>
      <td>Abusive/Offensive</td>
    </tr>
    <tr>
      <td>Hindu bhenchod hi ki gaand ma hi keerra hota hay Tum hindu ho hi harami tumhara kabhi 1 baap nhi hota</td>
      <td>There are always insects in asses of Hindu sisterfu**kers. These hindus have multiple fathers instead of 1.</td>
      <td>Religious Hate</td>
    </tr>
    <tr>
      <td>No wonder you can’t make it to First Lady. At least you managed to grab the title of FIRST RANDDI</td>
      <td>No wonder you can’t make it to First Lady. At least you managed to grab the title of FIRST PROSTITUTE.</td>
      <td>Sexism</td>
    </tr>
    <tr>
      <td>bahria central park karachi forms sold out in two days. Abhi tax maango bhenchodo ka rona shru hojayega</td>
      <td>bahria central park karachi forms sold out in two days. Now ask them for tax these motherf**kers start crying.</td>
      <td>Profane</td>
    </tr>
    <tr>
      <td>pakistan me ptv news or ptv parliment ne hi mulk k liye acha kam kia</td>
      <td>in pakistan, only ptv news and ptv parliment has done good work for the country.</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>
<div class="caption">
Table 1. Samples of tweets for each label from RUHSOLD dataset.
</div>

<table style="width:100%; border-collapse: collapse; text-align:center;">
  <thead>
    <tr>
      <th>Label</th>
      <th>Tweet Count</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Abusive/Offensive</td><td>2,402</td></tr>
    <tr><td>Sexism</td><td>839</td></tr>
    <tr><td>Religious Hate</td><td>782</td></tr>
    <tr><td>Profane</td><td>640</td></tr>
    <tr><td>Normal</td><td>5,349</td></tr>
    <tr><td><b>Total</b></td><td><b>10,012</b></td></tr>
  </tbody>
</table>

<div class="caption">
Table 2. Tweet counts with respect to labels in the RUHSOLD dataset.
</div>

### Experimental Setup 
The experiments were conducted on the RUHSOLD dataset in two settings: a coarse-grained binary classification between normal and hate/offensive content, and a fine-grained five-class setup (Normal, Abusive/Offensive, Profane, Sexism, and Religious Hate). The data was split into 7,209 training tweets, 801 validation tweets, and 2,003 test tweets, with a class imbalance favoring the normal category.

We tested six types of embeddings, including LASER, ELMo, multilingual BERT, XLM-RoBERTa, FastText, and RomUrEm, the latter being domain-specific Roman Urdu embeddings trained on approximately 4.7 million tweets. For baselines, seven models were implemented: LSTM with gradient boosted decision trees, Bi-LSTM with attention, FastText with CNN, domain embeddings with CNN, ensemble classifiers combining SVM, random forest and AdaBoost, BERT with LAMB optimizer, and BERT with LASER features combined with LightGBM.

To improve upon baseline approaches we propose CNN-gram model (Figure 1), this model stacks convolutional blocks to learn unigram, bigram, trigram, and four-gram patterns, followed by pooling layers and dense layers for classification. CNN-gram was tested with BERT, XLM-RoBERTa, FastText, and RomUrEm embeddings.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/hatespeech.jpg" title="Lexical dominance visual" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
</div>
<div class="caption">
Figure 1: CNN-gram model for hate-speech and offensive language detection in Roman Urdu.
</div>


### Coarse-grained Classification
<!-- Table 3 -->
<table style="width:100%; border-collapse: collapse; text-align:center;">
  <thead>
    <tr>
      <th rowspan="2">Embedding</th>
      <th colspan="4">Without Fine-tuning</th>
      <th colspan="4">With Fine-tuning</th>
    </tr>
    <tr>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LASER</td><td>0.74</td><td>0.74</td><td>0.74</td><td>0.74</td><td>0.76</td><td>0.76</td><td>0.76</td><td>0.76</td></tr>
    <tr><td>ELMo</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.79</td><td>0.79</td></tr>
    <tr><td>BERT</td><td>0.68</td><td>0.70</td><td>0.68</td><td>0.67</td><td>0.89</td><td>0.90</td><td>0.89</td><td>0.89</td></tr>
    <tr><td>XLM-RoBERTa</td><td>0.53</td><td>0.27</td><td>0.50</td><td>0.35</td><td>0.85</td><td>0.85</td><td>0.85</td><td>0.85</td></tr>
    <tr><td>FastText</td><td>0.74</td><td>0.75</td><td>0.73</td><td>0.73</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
    <tr><td>RomUrEm</td><td>0.85</td><td>0.84</td><td>0.84</td><td>0.84</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
  </tbody>
</table>
<div class="caption">
Table 3. Out-of-the-box performance of different embeddings for <i>coarse-grained</i> classification.
</div>

<!-- Table 4 -->
<table style="width:100%; border-collapse: collapse; text-align:center; margin-top:18px;">
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LSTM+GBDT</td><td>0.54</td><td>0.58</td><td>0.51</td><td>0.38</td></tr>
    <tr><td>BERT+LASER+GBDT</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td></tr>
    <tr><td>FastText+CNN</td><td>0.87</td><td>0.87</td><td>0.87</td><td>0.87</td></tr>
    <tr><td>SVM+RF+AB</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td></tr>
    <tr><td>BERT+LAMB</td><td>0.90</td><td>0.90</td><td>0.89</td><td>0.89</td></tr>
    <tr><td>Domain Embeddings+CNN</td><td>0.88</td><td>0.89</td><td>0.88</td><td>0.88</td></tr>
    <tr><td>BiLSTM with Attention</td><td>0.86</td><td>0.86</td><td>0.85</td><td>0.85</td></tr>
    <tr><td>BERT+CNN-gram</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td></tr>
    <tr><td>XLM-RoBERTa+CNN-gram</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
    <tr><td>FastText+CNN-gram</td><td>0.81</td><td>0.81</td><td>0.80</td><td>0.80</td></tr>
    <tr><td>RomUrEm+CNN-gram</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td></tr>
  </tbody>
</table>
<div class="caption">
Table 4. Comparisons of the proposed approach with baseline models on <i>coarse-grained</i> classification.
</div>



### Fine-grained Classification
<!-- Table 5 -->
<table style="width:100%; border-collapse: collapse; text-align:center; margin-top:18px;">
  <thead>
    <tr>
      <th rowspan="2">Embedding</th>
      <th colspan="4">Without Fine-tuning</th>
      <th colspan="4">With Fine-tuning</th>
    </tr>
    <tr>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LASER</td><td>0.66</td><td>0.62</td><td>0.42</td><td>0.46</td><td>0.67</td><td>0.59</td><td>0.52</td><td>0.54</td></tr>
    <tr><td>ELMo</td><td>0.70</td><td>0.64</td><td>0.52</td><td>0.56</td><td>0.60</td><td>0.66</td><td>0.50</td><td>0.55</td></tr>
    <tr><td>BERT</td><td>0.61</td><td>0.60</td><td>0.36</td><td>0.37</td><td>0.77</td><td>0.72</td><td>0.65</td><td>0.67</td></tr>
    <tr><td>XLM-RoBERTa</td><td>0.53</td><td>0.11</td><td>0.20</td><td>0.14</td><td>0.79</td><td>0.70</td><td>0.75</td><td>0.72</td></tr>
    <tr><td>FastText</td><td>0.62</td><td>0.55</td><td>0.33</td><td>0.35</td><td>0.77</td><td>0.69</td><td>0.63</td><td>0.66</td></tr>
    <tr><td>RomUrEm</td><td>0.70</td><td>0.69</td><td>0.51</td><td>0.56</td><td>0.79</td><td>0.76</td><td>0.63</td><td>0.67</td></tr>
  </tbody>
</table>
<div class="caption">
Table 5. Out-of-the-box performance of different embeddings for <i>fine-grained</i> classification.
</div>

<!-- Table 6 -->
<table style="width:100%; border-collapse: collapse; text-align:center; margin-top:18px;">
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LSTM+GBDT</td><td>0.44</td><td>0.28</td><td>0.29</td><td>0.27</td></tr>
    <tr><td>BERT+LASER+GBDT</td><td>0.73</td><td>0.67</td><td>0.63</td><td>0.64</td></tr>
    <tr><td>FastText+CNN</td><td>0.71</td><td>0.62</td><td>0.57</td><td>0.58</td></tr>
    <tr><td>SVM+RF+AB</td><td>0.76</td><td>0.68</td><td>0.63</td><td>0.65</td></tr>
    <tr><td>BERT+LAMB</td><td>0.76</td><td>0.67</td><td>0.63</td><td>0.64</td></tr>
    <tr><td>Domain Embeddings+CNN</td><td>0.74</td><td>0.64</td><td>0.60</td><td>0.61</td></tr>
    <tr><td>BiLSTM with Attention</td><td>0.71</td><td>0.62</td><td>0.59</td><td>0.60</td></tr>
    <tr><td>BERT+CNN-gram</td><td>0.78</td><td>0.69</td><td>0.66</td><td>0.67</td></tr>
    <tr><td>XLM-RoBERTa+CNN-gram</td><td>0.78</td><td>0.70</td><td>0.67</td><td>0.68</td></tr>
    <tr><td>FastText+CNN-gram</td><td>0.73</td><td>0.64</td><td>0.61</td><td>0.62</td></tr>
    <tr><td>RomUrEm+CNN-gram</td><td>0.79</td><td>0.71</td><td>0.67</td><td>0.69</td></tr>
  </tbody>
</table>

<div class="caption">
Table 6. Comparisons of the proposed approach with baseline models on <i>fine-grained</i> classification.
</div>


### Conclusion
In this work, we presented a dataset in Roman Urdu for the task of hate-speech detection in social media content, annotated with five fine-grained labels. We also make publicly available domain-specific embeddings trained on a parallel corpora of more than 4.7 million tweets. Furthermore, an extensive experimentation with respect to multiple embeddings, their power of transfer learning, and comparison with existing baseline models is carried out. As a future research, semantically challenging cases at fine-grained level with respect to complexities of Abusive/Offensive (targeted) and Profane (untargeted) language demand further investigation.