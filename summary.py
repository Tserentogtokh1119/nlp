import os 
import re
import argparse
from urllib.parse import urlparse, parse_qs

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from youtube_transcript_api import YouTubeTranscriptApi  
from sentence_transformers import SentenceTransformer 
from keybert import KeyBERT  
import yake 


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def video_id_from_url(url: str) -> str:
    u = urlparse(url)
    if "youtu.be" in u.netloc:
        return u.path.strip("/").split("/")[0]
    if "youtube.com" in u.netloc:
        q = parse_qs(u.query)
        return q.get("v", [""])[0]
    return url  # if user already passed ID


def fetch_transcript(video_id: str, languages):
    ytt = YouTubeTranscriptApi()
    ft = ytt.fetch(video_id, languages=languages)
    return ft.to_raw_data()  # [{'text','start','duration'}, ...]


def chunk_by_time(snips, window_sec=60):
    chunks = []
    cur_text = []
    cur_start = snips[0]["start"]
    cur_end = cur_start + window_sec

    for s in snips:
        t = s["start"]
        if t <= cur_end:
            cur_text.append(s["text"])
        else:
            chunks.append({"start": cur_start, "text": " ".join(cur_text)})
            cur_start = t
            cur_end = cur_start + window_sec
            cur_text = [s["text"]]

    if cur_text:
        chunks.append({"start": cur_start, "text": " ".join(cur_text)})
    return chunks


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # engiin split hiih uildel
    sents = re.split(r"(?<=[\.\!\?\u2026])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) >= 20]


def extract_keywords_keybert(doc: str, kw_model: KeyBERT, top_n=12):
    kws = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 3),
        stop_words=None,
        top_n=top_n
    )
    return [k for k, score in kws]
def extract_keywords_yake(doc: str, lan="en", top_n=12):
    ke = yake.KeywordExtractor(lan=lan, n=3, top=top_n)
    kws = ke.extract_keywords(doc)
    return [k for k, score in kws]


def build_outline(chunks, embedder, kw_model, threshold_percentile=25):
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, normalize_embeddings=True)

    sims = []
    for i in range(len(emb) - 1):
        sims.append(float(np.dot(emb[i], emb[i + 1])))

    if len(sims) == 0:
        return [{"start": chunks[0]["start"], "title": "Intro", "text": chunks[0]["text"]}]

    thr = np.percentile(sims, threshold_percentile)

    boundaries = [0]
    for i, s in enumerate(sims):
        if s < thr:
            boundaries.append(i + 1)
    boundaries.append(len(chunks))

    sections = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        sec_text = " ".join([chunks[i]["text"] for i in range(a, b)])
        title = extract_keywords_keybert(sec_text, kw_model, top_n=3)[0] if sec_text.strip() else "Section"
        sections.append({"start": chunks[a]["start"], "title": title, "text": sec_text})

    for i in range(1, len(sections)):
        if sections[i]["title"] == sections[i - 1]["title"]:
            sections[i]["title"] = sections[i]["title"] + " (cont.)"

    return sections


def extractive_summary(full_text: str, embedder, k=6):
    sents = split_sentences(full_text)
    if len(sents) <= k:
        return sents

    E = embedder.encode(sents, normalize_embeddings=True)
    centroid = np.mean(E, axis=0, keepdims=True)
    scores = cosine_similarity(E, centroid).ravel()

    top_idx = np.argsort(scores)[::-1][:k]
    top_idx = sorted(top_idx)  # эх дарааллаар нь
    return [sents[i] for i in top_idx]


def fmt_time(sec: float) -> str:
    sec = int(sec)
    mm = sec // 60
    ss = sec % 60
    return f"{mm:02d}:{ss:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="YouTube URL эсвэл video id")
    ap.add_argument("--lang", default="mn,en", help="priorities: mn,en гэх мэт")
    ap.add_argument("--window", type=int, default=60, help="chunk window seconds")
    ap.add_argument("--summary_k", type=int, default=6)

    ap.add_argument("--cache_dir", default="hf_cache", help="HuggingFace cache folder")
    ap.add_argument("--model_dir", default="saved_models/paraphrase-multi", help="Saved model folder (local)")
    ap.add_argument("--offline", action="store_true", help="Offline mode (internet ашиглахгүй)")

    args = ap.parse_args()

    vid = video_id_from_url(args.url)
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]

    snips = fetch_transcript(vid, languages=langs)
    full_text = " ".join([s["text"] for s in snips])

    cache_dir = os.path.abspath(args.cache_dir)
    model_dir = os.path.abspath(args.model_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if os.path.exists(os.path.join(model_dir, "modules.json")):
        embedder = SentenceTransformer(model_dir)
    else:
        embedder = SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        embedder.save(model_dir)

    kw_model = KeyBERT(model=embedder)

    # 1) Keywords (document-level)
    keywords = extract_keywords_keybert(full_text, kw_model, top_n=12)

    # 2) Outline
    chunks = chunk_by_time(snips, window_sec=args.window)
    sections = build_outline(chunks, embedder, kw_model)

    # 3) Summary (extractive)
    summary = extractive_summary(full_text, embedder, k=args.summary_k)

    print("\n=== KEYWORDS ===")
    for k in keywords:
        print("-", k)

    print("\n=== OUTLINE ===")
    for s in sections:
        print(f"- [{fmt_time(s['start'])}] {s['title']}")

    print("\n=== SUMMARY ===")
    for s in summary:
        print("•", s)


if __name__ == "__main__":
    main()
