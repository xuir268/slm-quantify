from typing import List, Dict
# In production, use SentenceTransformers + FAISS. For now, dummy stubs:

_DOCS = [
  {"text":"CVE-2024-0001 ... JSON payload RCE ... patch 3.2.5"},
  {"text":"CVE-2024-0002 ... SQL injection ... upgrade 5.1"},
  {"text":"CVE-2024-0003 ... DoS via image ... fix 1.0.8"}
]

def retrieve_topk(question:str, k:int=3)->List[Dict]:
    # naive: return first k docs; swap with FAISS later
    return _DOCS[:k]

def compress_to_bullets(chunks:List[Dict], max_bullets:int=6)->str:
    facts=[]
    for c in chunks:
        facts.append("- " + c["text"][:180])
        if len(facts)>=max_bullets: break
    return "\n".join(facts)

def build_prompt(question:str, bullets:str)->str:
    return ( "You are a concise cybersecurity analyst. Answer ONLY using these facts.\n"
             f"{bullets}\n\nQuestion: {question}\nAnswer:" )
