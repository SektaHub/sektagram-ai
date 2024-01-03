from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def embed_sentence(sentence):
    embedding = model.encode(sentence)
    return embedding.tolist()


# def embed_sentences(sentences):
#     embeddings = model.encode(sentences)
#     return embeddings
