#Natural Language Processing (NLP) and Text Analysis

#pip install nltk textblob
#pip install networkx

import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

def summarize_text(text, num_sentences):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Calculate word frequencies
    word_freq = FreqDist(words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Get the top N sentences with the highest scores
    summarized_sentences = [sentence for sentence, _ in sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]]

    # Join the summarized sentences to create the summary
    summary = " ".join(summarized_sentences)
    return summary


def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"
    
    
#keywords
def extract_keywords(text, num_keywords):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Calculate word frequencies
    word_freq = FreqDist(words)

    # Get the top N keywords with the highest frequencies
    keywords = [word for word, freq in word_freq.most_common(num_keywords)]

    return keywords


if __name__ == "__main__":
    text = input("Enter the text you want to analyze:\n")
    num_keywords = int(input("Enter the number of keywords you want to extract: "))
    num_sentences = int(input("Enter the number of sentences you want in the summary: "))
    
    summary = summarize_text(text, num_sentences)
    print("Summary:")
    print(summary)

    sentiment = analyze_sentiment(text)
    print("\nSentiment:", sentiment)

    #major keywords:
    keywords = extract_keywords(text, num_keywords)
    print("\nKeywords:", ", ".join(keywords))