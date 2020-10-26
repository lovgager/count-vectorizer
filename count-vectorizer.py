class CountVectorizer:
    def __init__(self):
        self.vocabulary = []

    def fit_transform(self, corpus):
        words_in_lines = [[word.lower() for word in line.split()] for line in corpus]
        all_words = [word for line in words_in_lines for word in line]
        self.vocabulary = list(dict.fromkeys(all_words))
        return [[line.count(vocabulary_word) for vocabulary_word in self.vocabulary] for line in words_in_lines]
    
    def get_feature_names(self):
        return self.vocabulary
        
        
if __name__ == "__main__":
    lines = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(lines)
    print(vectorizer.get_feature_names())
    print(count_matrix)
