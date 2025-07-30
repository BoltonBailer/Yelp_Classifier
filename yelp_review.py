import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


GiveMeSpace = "-" * 50
print(GiveMeSpace)


if __name__ == '__main__':
    output = r'../output'
    #loading in the data set
    path = r'../data/yelp-reviews.csv'
    seed = 123857

    df = pd.read_csv(path, encoding='latin1')
    #checking data
    df.info()
    print(GiveMeSpace)

    #checking the sample text and labels
    print(df.head(5))
    print(GiveMeSpace)

    #graph the possitvive and negative
    print(df['opinion'].value_counts())
    sns.countplot(x='opinion', data=df)
    plt.title("possitive vs negative reviews")
    plt.show()

    #running through the hugging face pipline
    classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    #testing samples sentences
    test_sample = df['review'][:20].tolist()
    actual_sample = df['opinion'][:20].tolist()
    print(GiveMeSpace)

    #accuracy check
    print(accuracy_score(actual_sample, predictions))
    cm = confusion_matrix(actual_sample, predictions)

    #change the 
    def flip_to_zeros(pred):
        return 1 if pred['label'] == 'POSITIVE' else 0

    predictions = [flip_to_zeros(classifier(text)[0]) for text in test_sample]


    print("Accuracy:", accuracy_score(actual_sample, predictions))
    cm = confusion_matrix(actual_sample, predictions)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NEGATIVE', 'POSITIVE'],
                yticklabels=['NEGATIVE', 'POSITIVE'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



