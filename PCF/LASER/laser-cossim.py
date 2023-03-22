import numpy as np
from laserembeddings import Laser

# Get cosine similarity between 2 vectors
def cosine_similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2.T).squeeze() / n1 / n2

laser = Laser()

src = "집에 가고 싶다"
tgt = "I want to go home"

# Get embedding vector of sentences
src_embed = laser.embed_sentences(src, lang='ko')
tgt_embed = laser.embed_sentences(tgt, lang='en')

# Get cosine similarity score between src, tgt embeddings
cs_score = cosine_similarity(src_embed, tgt_embed)

print(cs_score)