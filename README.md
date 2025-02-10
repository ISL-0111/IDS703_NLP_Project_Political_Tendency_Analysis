## Political Leaning in News with BERT

### Abstract
This paper explores the application of BERT-based architectures for document classification of
political leanings, focusing on DistilBERT for its versatility and lightweight design while
maintaining performance comparable to larger models. The study compares three methods: Naive
Bayes, refined fully connected (FC) layers, and Low-Rank Adaptation (LoRA). Naive Bayes
serves as a baseline, highlighting the superior performance of transformer-based approaches.
Enhancements to FC layers, incorporating cosine similarity and neural networks, provided
incremental improvements and benchmarks. LoRA, integrated into the DistilBERT architecture,
delivered the best results by optimizing attention mechanisms with trainable query layers and
improved FC layers.

Experimental results demonstrate the effectiveness of these approaches. On the 2017 dataset,
DistilBERT with cosine similarity achieved an accuracy of 0.45, while the method using a neural
network fully connected layer model improved it to 0.63. However, LoRA significantly
outperformed both, achieving 0.91 accuracy with precision, recall, and F1-scores of 0.9083,
0.9074, and 0.9066, respectively. On the 2019 dataset, while distilBERT with neural networks FC
layer model’s accuracy dropped to 0.4162, distilBERT+LoRA maintained robust performance
with 0.7942 accuracy.

These results highlight the adaptability of BERT-based models to varying data conditions and
emphasize LoRA’s role in optimizing classification tasks, even with data drift across years. Future
work will explore headline-specific performance challenges, alternative a

#### Contributor
Janet Chen, Ilseop Lee, Jahnavi Maddhuri, Alejandro Par
