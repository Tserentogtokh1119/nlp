import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM

st.set_page_config(page_title="NLP Microscope", layout="wide")

MODEL_NAME = "bert-base-multilingual-cased"  # олон хэлтэй BERT
st.title("🔬 NLP Microscope — Token/ID/Vector/Attention/MASK")

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()

text = st.text_area("Текст оруул:", "энэ кино маш гоё биш", height=80)

colA, colB = st.columns([1,1])
with colA:
    layer_choice = st.slider("Attention layer (last=хамгийн сүүлчийнх)", 0, model.config.num_hidden_layers - 1, model.config.num_hidden_layers - 1)
with colB:
    topk = st.slider("MASK санал болгох top-k", 3, 20, 8)

if not text.strip():
    st.stop()

# --- Tokenize ---
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

df = pd.DataFrame({"token": tokens, "id": input_ids.tolist()})
st.subheader("1) Tokenization & IDs")
st.dataframe(df, use_container_width=True)

# --- Forward with attentions/hidden states ---
with torch.no_grad():
    out = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)

# hidden states: tuple length = layers+1, each (B, L, d)
last_hidden = out.hidden_states[-1][0]  # (L, d)
L, d = last_hidden.shape
st.subheader("2) Vector хэмжээ (embedding / hidden state)")
st.write(f"Sequence length L = **{L}** tokens, hidden size d = **{d}**  → матриц хэмжээ: **({L} × {d})**")

# show small preview for one token
pick = st.slider("Аль токены vector-ыг харах вэ? (index)", 0, L-1, min(1, L-1))
vec = last_hidden[pick].cpu().numpy()
st.write(f"Token: `{tokens[pick]}`  |  L2 norm ≈ {float(torch.norm(last_hidden[pick]).item()):.4f}")
st.code("vector[:12] = " + str([float(x) for x in vec[:12]]))

# --- Attention heatmap ---
# attentions: tuple length = layers, each (B, heads, L, L)
att_layer = out.attentions[layer_choice][0]   # (heads, L, L)
att = att_layer.mean(dim=0).cpu().numpy()     # average heads -> (L, L)

st.subheader("3) Attention heatmap (average over heads)")
fig, ax = plt.subplots()
ax.imshow(att, aspect="auto")
ax.set_xticks(range(L))
ax.set_yticks(range(L))
ax.set_xticklabels(tokens, rotation=90, fontsize=8)
ax.set_yticklabels(tokens, fontsize=8)
ax.set_title(f"{MODEL_NAME} | layer={layer_choice} | mean(heads)")
st.pyplot(fig, use_container_width=True)

# --- MASK prediction ---
st.subheader("4) MASK тавиад ямар үг таамаглах вэ?")
mask_idx = st.slider("MASK болгох токены index (CLS/SEP дээр битгий тавиарай)", 1, L-2, min(3, L-2))
masked_ids = input_ids.clone()
masked_ids[mask_idx] = tokenizer.mask_token_id

masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
st.write("Masked input:")
st.code(masked_text)

with torch.no_grad():
    mout = model(input_ids=masked_ids.unsqueeze(0))
logits = mout.logits[0, mask_idx]  # (vocab,)
probs = torch.softmax(logits, dim=-1)
vals, idxs = torch.topk(probs, k=topk)

suggest = []
for p, i in zip(vals.tolist(), idxs.tolist()):
    suggest.append((tokenizer.convert_ids_to_tokens(i), i, p))

st.table(pd.DataFrame(suggest, columns=["token", "id", "prob"]))
