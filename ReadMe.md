# 1. Accuracy on IMDB test set
Acc = (True positive + True negative) / No. of test sentences
| Model                   | Acc |
|-------------------------|-----|
| With pretrained BERT    | 82% |
| Without pretrained BERT | 78% |

# 2. Training set
* IMDB training set (35.4MB, 1,222 supervised sentences) is used for both training pretrained BERT and fine-tuning for sentiment classification.
* Yes. It will give better result if BERT is pretrained on bigger, unlabeled DB.
