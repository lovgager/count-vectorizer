class CountVectorizer:
    def __init__(self):
        self.vocabulary = []

    def fit_transform(self, corpus):
        words_in_lines = [line.lower().split() for line in corpus]
        all_words = [word for line in words_in_lines for word in line]
        self.vocabulary = list(dict.fromkeys(all_words))
        return [
            [line.count(v_word) for v_word in self.vocabulary]
            for line in words_in_lines
        ]

    def get_feature_names(self):
        return self.vocabulary


def tf_transform(count_matrix):
    return [
        [round(count / sum(line), 3) for count in line]
        for line in count_matrix
    ]


def idf_transform(count_matrix):
    import math

    all_documents = len(count_matrix)
    occurs = [[count > 0 for count in line] for line in count_matrix]
    docs_with_word = [sum(i) for i in zip(*occurs)]
    return [
        round(math.log((all_documents + 1) / (d + 1)) + 1, 3)
        for d in docs_with_word
    ]


class TfidfTransformer:
    @staticmethod
    def fit_transform(count_matrix):
        tf = tf_transform(count_matrix)
        idf = idf_transform(count_matrix)
        return [
            [round(line_tf[i] * idf[i], 3) for i in range(len(idf))]
            for line_tf in tf
        ]


class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.count_matrix = []
        self.tfidf_matrix = []

    def fit_transform(self, corpus):
        self.count_matrix = super().fit_transform(corpus)
        self.tfidf_matrix = TfidfTransformer().fit_transform(self.count_matrix)
        return self.tfidf_matrix


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    count_matrix = CountVectorizer().fit_transform(corpus)
    print("Count matrix:\n", count_matrix)
    tf_matrix = tf_transform(count_matrix)
    print("tf matrix:\n", tf_matrix)
    idf_matrix = idf_transform(count_matrix)
    print("idf matrix:\n", idf_matrix)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("Feature names:\n", vectorizer.get_feature_names())
    print("tfidf matrix:\n", tfidf_matrix)
