"""
NHPT Heritage AI — Streamlit App
Machine Learning and Related Applications (NB627BSDS) — Coursework 01

Combines Part B (EfficientNetB0 architectural style classifier + Grad-CAM) and Part C
(LangChain RAG assistant over live-fetched Wikipedia heritage documents, Gemma 3 12B via Ollama)
into one interactive app: upload a photo of a building, get its predicted architectural style
with an explainability heatmap, and chat with the assistant about it.

Run with:
    streamlit run app.py

Requirements (same as the two source notebooks):
    pip install streamlit tensorflow opencv-python-headless pillow numpy matplotlib
    pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama
    pip install sentence-transformers beautifulsoup4
    ollama pull gemma3:12b   (and have `ollama serve` running)

Expects `efficientnetb0_heritage_style_final.keras` (produced by heritage_style_classification.ipynb)
in the same folder as this script.
"""

import json
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NHPT Heritage AI", page_icon="🏛️", layout="wide")

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Gothic architecture", "Georgian architecture", "Baroque architecture",
    "Romanesque architecture", "Queen Anne architecture", "Palladian architecture",
]
MODEL_PATH = "efficientnetb0_heritage_style_final.keras"
PERSIST_DIR = "chroma_heritage_db_v2"

# Real UK heritage buildings, one per architectural style class — same set used in
# heritage_langchain_assistant.ipynb (Part C), kept identical here for consistency.
SITE_URLS = {
    "ashcombe_abbey":    "https://en.wikipedia.org/wiki/Westminster_Abbey",   # Gothic architecture
    "falmoor_house":     "https://en.wikipedia.org/wiki/Kenwood_House",       # Georgian architecture
    "thornfield_palace": "https://en.wikipedia.org/wiki/Blenheim_Palace",     # Baroque architecture
    "st_edwins_priory":  "https://en.wikipedia.org/wiki/Durham_Cathedral",    # Romanesque architecture
    "marlow_court":      "https://en.wikipedia.org/wiki/Uppark",              # Queen Anne architecture
    "chiswick_hall":     "https://en.wikipedia.org/wiki/Chiswick_House",      # Palladian architecture
}


# ---------------------------------------------------------------------------
# Cached resource loaders — each runs once per app session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained CV model...")
def load_cv_model():
    model = tf.keras.models.load_model(MODEL_PATH)

    # Find the base EfficientNetB0 submodel and its last conv layer (for Grad-CAM)
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(base_model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["global_average_pooling2d", "dropout", "dense"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    return model, conv_model, classifier_model


@st.cache_resource(show_spinner="Connecting to Gemma 3 12B via Ollama...")
def load_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="gemma3:12b", temperature=0.2)


@st.cache_resource(show_spinner="Loading heritage knowledge base (Chroma)...")
def load_vectorstore():
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.isdir(PERSIST_DIR):
        # Reuse the index already built by heritage_langchain_assistant.ipynb — avoids
        # re-fetching all 6 Wikipedia pages (and re-embedding ~500 chunks) on every app start.
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="nhpt_heritage_docs",
        )

    # Fallback: build it fresh if no persisted index is found (first run, or a clean clone)
    import re

    import bs4
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    def clean_wikipedia_text(text: str) -> str:
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[edit\]", "", text)
        for stop_heading in ["\nSee also", "\nReferences", "\nExternal links", "\nNotes", "\nFurther reading"]:
            idx = text.find(stop_heading)
            if idx != -1:
                text = text[:idx]
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    raw_docs = []
    for slug, url in SITE_URLS.items():
        loader = WebBaseLoader(web_paths=[url], bs_kwargs=dict(parse_only=bs4.SoupStrainer(id="mw-content-text")))
        page_docs = loader.load()
        for d in page_docs:
            d.page_content = clean_wikipedia_text(d.page_content)
            d.metadata["source_name"] = slug.replace("_", " ").title()
            d.metadata["source_url"] = url
            d.metadata["source"] = slug
        raw_docs.extend(page_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nhpt_heritage_docs",
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# CV inference + Grad-CAM
# ---------------------------------------------------------------------------
def predict_structured(pil_image, model, class_names=CLASS_NAMES):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return {
        "predicted_style": class_names[top_idx],
        "confidence": float(preds[top_idx]),
        "all_scores": {class_names[i]: float(preds[i]) for i in range(len(class_names))},
    }


def structured_to_context(result):
    return (
        f"A visitor uploaded a photo. The computer vision model classified it as "
        f"'{result['predicted_style']}' with {result['confidence']*100:.1f}% confidence. "
        f"Full distribution: {json.dumps(result['all_scores'])}."
    )


def make_gradcam_overlay(pil_image, conv_model, classifier_model):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array.copy())

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_preprocessed)
        tape.watch(conv_output)
        predictions = classifier_model(conv_output)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base_img = np.array(img).astype("float32")
    overlay = heatmap_color * 0.4 + base_img
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay


# ---------------------------------------------------------------------------
# RAG assistant
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a heritage guide assistant for the National Heritage Preservation Trust (NHPT).
Answer the visitor's question using ONLY the information in the CONTEXT below.

Rules:
- Every factual claim must be attributable to a document in the context. After each claim, name the
  source document in parentheses, e.g. "(Source: Ashcombe Abbey)".
- If the context does not contain enough information to answer confidently, say so plainly instead
  of guessing — do not invent dates, names, or architectural details that are not in the context.
- If an image analysis result is included in the context, treat it as a preliminary classification
  from a computer vision model (with a confidence score) and explain it in plain language, noting
  the confidence level rather than stating it as certain fact.
- Keep answers concise (3-5 sentences) unless the visitor asks for more detail.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}
"""


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source_name', 'unknown')}] {d.page_content}" for d in docs)


def format_history(history, max_turns=4):
    recent = history[-max_turns:]
    return "\n".join(f"Visitor: {q}\nGuide: {a}" for q, a in recent) if recent else "(no prior turns)"


def ask(question, retriever, llm, chat_history, image_context=None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

    docs = retriever.invoke(question)
    context = format_docs(docs)
    if image_context:
        context = f"[Image analysis result] {image_context}\n\n{context}"

    chain_input = {
        "context": context if docs or image_context else "(no relevant documents found in the knowledge base)",
        "chat_history": format_history(chat_history),
        "question": question,
    }
    response = (prompt | llm | StrOutputParser()).invoke(chain_input)
    sources = sorted(set(d.metadata.get("source_name", "unknown") for d in docs))
    return response, sources


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🏛️ NHPT Heritage AI Assistant")
st.caption(
    "Upload a photo of a building to get its architectural style, a Grad-CAM explainability "
    "heatmap, and a grounded explanation from the heritage assistant (Gemma 3 12B, local via Ollama)."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cv_result" not in st.session_state:
    st.session_state.cv_result = None
if "image_context" not in st.session_state:
    st.session_state.image_context = None

with st.spinner("Starting up (first load only)..."):
    model, conv_model, classifier_model = load_cv_model()
    vectorstore = load_vectorstore()
    llm = load_llm()

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.35},
)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a building photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded photo", use_container_width=True)

        with st.spinner("Classifying architectural style..."):
            cv_result = predict_structured(pil_image, model)
            overlay = make_gradcam_overlay(pil_image, conv_model, classifier_model)

        st.session_state.cv_result = cv_result
        st.session_state.image_context = structured_to_context(cv_result)

        st.subheader(f"Predicted: {cv_result['predicted_style']}")
        st.progress(cv_result["confidence"])
        st.write(f"Confidence: **{cv_result['confidence']*100:.1f}%**")

        st.image(overlay, caption="Grad-CAM — where the model looked", use_container_width=True)

        with st.expander("Full class score distribution"):
            st.bar_chart(cv_result["all_scores"])

with col2:
    st.subheader("Ask the heritage assistant")

    if st.session_state.cv_result is not None:
        st.info(
            f"An image is loaded ({st.session_state.cv_result['predicted_style']}, "
            f"{st.session_state.cv_result['confidence']*100:.1f}% confidence). "
            "Questions below will include this in context automatically."
        )

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    user_question = st.chat_input("Ask about the uploaded building, or any NHPT heritage site...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking..."):
            answer, sources = ask(
                user_question,
                retriever=retriever,
                llm=llm,
                chat_history=st.session_state.chat_history,
                image_context=st.session_state.image_context,
            )

        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.chat_history.append((user_question, answer))

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
