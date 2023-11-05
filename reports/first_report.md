# First Report: Initial Approach to Text Detoxification

## Introduction

In my quest to develop a text detoxification solution, I aimed to reduce toxicity in textual data while preserving the content's original meaning and intent. This complex task required the exploration of diverse architectures, strategies, and datasets to develop an effective approach to mitigating offensive language.

## Initial Solution Concept

I began with a basic model that targeted and removed recognized offensive words from the text. This approach was based on the assumption that directly removing profanities would linearly decrease the text's toxicity.

### Utilization of a Bad Word List

- The model scanned and sanitized text against a predefined list of offensive words.
- The list was integrated into the model to provide a benchmark for text filtering.

#### Challenges Faced

- **Distortion of Meaning:** The blunt removal of words sometimes altered the original message.
- **Disregard for Context:** The method did not consider the context, leading to potential over-censorship.
- **Lack of Finesse:** The approach was simplistic and did not capture the subtleties of linguistic expression.

## Advancement to T5 Model Architecture

I transitioned to the T5 model to address these issues, leveraging its ability to transform text which aligns with detoxification objectives.

### Evaluation Metrics

To gauge the performance of the models, I employed two main metrics: style transfer accuracy and mean cosine similarity.

#### Style Transfer Accuracy

This metric quantifies the model's ability to maintain a non-toxic tone. It reflects how effectively the model can transform toxic text to neutral, while preserving the original intent.

#### Mean Cosine Similarity

Cosine similarity measures the semantic proximity between the original and detoxified text, with a higher score indicating that the core message is retained after detoxification.

### Challenges in Fine-Tuning GPT-2

The attempt to fine-tune the GPT-2 model encountered hurdles due to:

- **Training Duration:** The GPT-2 model required an impractical amount of time to train.
- **Computational Demand:** Extensive computational resources were necessary, which were not available.
- **Quality of Output:** The model's outputs often included hallucinations, straying from the factual content.

### Deep Dive into T5 Model Architecture

T5, or Text-to-Text Transfer Transformer, was selected for its universality in handling diverse NLP tasks by framing them as text-to-text problems. Its encoder-decoder structure, pre-training on a corrupted text reconstruction task, and attention mechanisms, provide a sophisticated understanding of context and language structure.

![T5 Architecture](figures/T5.png)

## Fine-Tuning Strategy

The T5 model was fine-tuned on a curated dataset, with performance monitored through loss metrics to ensure effective learning.

## Results and Observations

Comparative results showed that the T5 model outperformed the basic approach in terms of style transfer accuracy and maintaining meaning.

| Model          | Style Transfer Accuracy | Mean Cosine Similarity |
| -------------- | ----------------------- | ---------------------- |
| Deletion Model | 0.48                    | 0.98                   |
| T5 Based Model | 0.17                    | 0.99                   |

## Conclusion

The initial method of deleting offensive words was an instructive but inadequate solution, highlighting the need for a more nuanced approach. The T5 model represents significant progress towards a sophisticated text detoxification process. Second report will delve into detailed performance metrics and evaluations of the fine-tuned T5 model.
