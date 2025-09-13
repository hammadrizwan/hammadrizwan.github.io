---
layout: page
title: Estimating DBH of Forest Trees via Monocular Vision
description: another without an image
img: assets/img/forest.jpg
importance: 4
category: fun

---

### Summary:
Deforestation is one of the major cause of climate change in Pakistan, which in turn is having an adverse effect not only on the country's agricultural sector that plays a vital role in its economy; but also has ramifications related to global warming, flash floods, and ever-increasing landslides. To overcome this problem the administration has taken steps to maintain proper forest inventory, but due to lack of resources, equipment and trained personnel, these measures are insufficient. To that end, we have designed and developed an easy-to-use android based system, that automates the process of forest inventory that is otherwise done manually. The system uses a pair of images taken from any android phone and utilizes the Detectron 2 library along with SIFT features matching to segment the tree trunks. Segmented trunks are used with photogrammetry and lens imaging principles to compute Diameter at Breast Height (DBH). DBH can further be used to calculate above-ground biomass (ABG), which can be used for calculating carbon content and maintaining the inventory. 

### Overview
Foresters in Pakistan keep records of the number of trees present in the forests, as well as the species and the carbon content of the trees. This helps them evaluate how much plantation is required to keep the forest/land ratio balanced. The task of forest inventory management includes calculation of diameter at breast height (DBH) which is a quite tedious and lengthy procedure, as foresters must do this task manually due to unavailability of proper equipment, which also introduces human error. 

Automation of this task is a necessity, but it comes with its own share of problems. Firstly, the forests are sometimes so dense and cluttered that trees cannot be separated by naked eye, let alone any image processing tool. Secondly, there are numerous species of trees in Pakistan, with up to 30 species of trees that are found in the wild.

Keeping in mind the challenges and obstacles at hand, we devise an android based system that uses pictures of the tree trunk to calculate the DBH. To achieve this, we employ a multi-image strategy with a photogramattery equation to calculate the distance from camera to the tree which we use along with lens imaging principal to calculate DBH.

### Dataset
We have collected the dataset in two phases. In the first phase, we collected tree images from our university and nearby forest. This dataset contains trees of species that are generally found in the province of Punjab, Pakistan. We capture the images as single unpaired images whose focus is on the entire tree\textit{(both canopy and trunk)}. We refer to this dataset as semi-urban dataset. 

The second data we have collected is from Haripur forest reserve near Islamabad. Most of the trees here are of Chir Pine species which can grow quite tall, usually between 98–164 feet. Due to the height of trees and dense forest setting we are only able to capture trunk images of these trees which is sufficient for our application as our focus in this study is on DBH.
This dataset contains both single unpaired images as-well-as paired images that are taken in succession, with one image being taken after moving a certain distance closer to the tree, as described in the methodology section. For the paired images, we also record the ground truth for distance from the far image to the tree, distance moved between taking images and the DBH. We refer to this dataset as a forest dataset. We capture these images with two cameras, namely Xiaomi POCO X3 and the OnePlus 8 Pro. The purpose of using two devices is to check the effect of the camera quality on our results, with the OnePlus having a better camera in this case.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sample_dataset_images_combined.jpg" title="Lexical dominance visual" class="img-fluid rounded z-depth-1 w-70 mx-auto d-block" %}
    </div>
</div>
<div class="caption">
Figure 1.  Sample images for forest dataset(left) and semi-urban dataset(right).
</div>

### Experimental Setup 
The experiments were conducted on the RUHSOLD dataset in two settings: a coarse-grained binary classification between normal and hate/offensive content, and a fine-grained five-class setup (Normal, Abusive/Offensive, Profane, Sexism, and Religious Hate). The data was split into 7,209 training tweets, 801 validation tweets, and 2,003 test tweets, with a class imbalance favoring the normal category.

We tested six types of embeddings, including LASER, ELMo, multilingual BERT, XLM-RoBERTa, FastText, and RomUrEm, the latter being domain-specific Roman Urdu embeddings trained on approximately 4.7 million tweets. For baselines, seven models were implemented: LSTM with gradient boosted decision trees, Bi-LSTM with attention, FastText with CNN, domain embeddings with CNN, ensemble classifiers combining SVM, random forest and AdaBoost, BERT with LAMB optimizer, and BERT with LASER features combined with LightGBM.

To improve upon baseline approaches we propose CNN-gram model (Figure 1), this model stacks convolutional blocks to learn unigram, bigram, trigram, and four-gram patterns, followed by pooling layers and dense layers for classification. CNN-gram was tested with BERT, XLM-RoBERTa, FastText, and RomUrEm embeddings.




### Conclusion
In this work, we presented a dataset in Roman Urdu for the task of hate-speech detection in social media content, annotated with five fine-grained labels. We also make publicly available domain-specific embeddings trained on a parallel corpora of more than 4.7 million tweets. Furthermore, an extensive experimentation with respect to multiple embeddings, their power of transfer learning, and comparison with existing baseline models is carried out. As a future research, semantically challenging cases at fine-grained level with respect to complexities of Abusive/Offensive (targeted) and Profane (untargeted) language demand further investigation.