import re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_word = re.compile(r"[A-Za-z0-9_\-]+")

def keywords(s:str):
    return _word.findall(s.lower())

def lexical_overlap(ans:str, bullets:str)->float:
    a=set(keywords(ans)); b=set(keywords(bullets))
    if not a or not b: return 0.0
    inter=len(a & b); return inter/max(1,len(b))  # coverage of facts

# Optional: add a tiny TF-IDF cosine for stability
_tf = TfidfVectorizer(min_df=1)
def tfidf_cosine(a:str, b:str)->float:
    X = _tf.fit_transform([a,b]).toarray()
    va, vb = X[0], X[1]
    num = (va*vb).sum(); den = np.linalg.norm(va)*np.linalg.norm(vb) + 1e-8
    return float(num/den)

def verify(answer:str, bullets:str)->float:
    # weighted combo; tune on dev set
    return 0.7*lexical_overlap(answer, bullets) + 0.3*tfidf_cosine(answer, bullets)
