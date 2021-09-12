

text = "This paper explores the political economy of network. Our model shows a r sqaured of 5%."

text_clean = str(text).lower()

text_clean = clean_text(text_clean)
print(text_clean)


text_tokens = word_tokenize(text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
tokens_without_sw = [word for word in text_tokens if not word in ['play', 'however']]
filtered_sentence = (" ").join(tokens_without_sw)


# developing

df = pd.DataFrame(data=[
        [1, 'John', 'Smith', 'a'],
        [1, 'John', 'Smith', 'b'],
        [2, 'Kate', 'Smith', 'c'],
    ],
    columns=['ID', 'First', 'Last', 'Avail']
)

output = (df
          .groupby(['ID', 'First', 'Last'], as_index=False)
          .agg({'Avail': lambda x: ';'.join(x)}))
