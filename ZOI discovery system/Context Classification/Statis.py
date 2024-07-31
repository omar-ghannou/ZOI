import re
from collections import Counter
import os
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt  # For visualization
from wordcloud import WordCloud  # For visualization



def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
def read_text_clustering(file_path):
    corpus_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            clean_line = line.strip()
            clean_line = clean_line.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
            clean_line = re.sub(r'\s+', ' ', clean_line)
            clean_line = re.sub(r'[0-9]', '', clean_line)
            if clean_line:
                corpus_lines.append(clean_line)
    return corpus_lines
    

# Function to clean and tokenize text
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)  # Extract words ignoring punctuation
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens #tokens

def remove_words(target_list, words_to_remove):
    cleaned_list = [word for word in target_list if word not in words_to_remove]
    return cleaned_list

def calculate_word_stats(words):
    # Get total word count
    total_words = len(words)

    # Get unique word count
    unique_words = len(set(words))

    # Calculate word frequency
    word_frequency = Counter(words)

    return total_words, unique_words, word_frequency

# Function to generate a word frequency report
def generate_report(word_frequency, report_file):
    with open(report_file, 'w', encoding='utf-8') as file:
        for word, frequency in word_frequency.most_common():
            file.write(f"{word}: {frequency}\n")

# Function to visualize word frequency
def visualize_word_frequency(word_frequency, top_n=100):

    most_common_words = word_frequency.most_common(top_n)
    words, frequencies = zip(*most_common_words)

    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.title(f'Top {top_n} Most Common Words')
    plt.show()

def visualize_word_cloud(word_frequency, label = ''):

    wordcloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(word_frequency)
    print(word_frequency.keys())

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    title = "Word Cloud of Corpus " + str(label)
    plt.title(title)
    plt.show()

def Unique_words(corpus):
    
    all_words = []

    #text = read_text(corpus)
    words = tokenize(corpus)
    unique_words = set(words)
    all_words.extend(unique_words)

    # Calculate word statistics
    ##total_words, unique_words, word_frequency = calculate_word_stats(all_words)
    return all_words

def stat_function(corpus, report_file):
    
    all_words = []

    text = read_text(corpus)
    #for i in range(len(embed_corpus)):
    text = text.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    words = tokenize(text)
    all_words.extend(words)

    remove_words(all_words,['','',])

    # Calculate word statistics
    total_words, unique_words, word_frequency = calculate_word_stats(all_words)

    # Generate a report with word frequencies
    generate_report(word_frequency, report_file)

    # Display total and unique word counts
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")

    #visualize_word_frequency(word_frequency, top_n=100)
    visualize_word_cloud(word_frequency)

