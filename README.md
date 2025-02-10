### Political Leaning in News with BERT

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

Trained model(Nov.19)


#### Contributor
Janet Chen, Ilseop Lee, Jahnavi Maddhuri, Alejandro Par


(BERT trained with BODY) https://duke.box.com/s/6clw9gx2vqpu26s4p7yh7z64gg3vi8yg

(BERT trained with HEADLINE) https://duke.box.com/s/8bax76my1wfbdu2715xtrp662zu6fl72

(BERT trained with SUMMARY) https://duke.box.com/s/zq5bu72d29k83tbuen5b24deqojzhiaj
 -> The accuracy is 62%. Using 10% (15K lines) of the full dataset (150K)
    Summarization was conduced with the T-5 'small' model. Would using the T-5 'Base' model improve the results......?


https://ground.news/

https://www.kaggle.com/code/mikiota/data-augmentation-csv-txt-using-back-translation

https://www.kaggle.com/code/nkitgupta/text-representations

https://huggingface.co/docs/transformers/en/tasks/summarization
