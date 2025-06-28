import os, random, requests, pandas as pd, streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from rapidfuzz import process, fuzz

# — load .env & key (needed by backend only)
load_dotenv()

# — FastAPI URL
API_URL = "http://127.0.0.1:8000/generate-response/"

# — Load CSV
df = pd.read_csv("/Users/katerinatelegina/Library/Mobile Documents/com~apple~CloudDocs/RAG-bookchat/RAG-Chatbot/backend/books.csv")

# — Normalize Popularity to bool (“checked” or “true” → True; everything else → False)
df["Popularity"] = (
    df["Popularity"]
      .astype(str)
      .str.lower()
      .map({"checked": True, "true": True})
      .fillna(False)
      .astype(bool)
)
st.set_page_config(page_title="Book recommender", layout="wide")
st.markdown("""
  <style>
    /* hide default menu/footer */
    #MainMenu, footer { visibility: hidden; }
    /* give extra bottom padding so the input never overlaps content */
    .block-container {
      padding: 2rem;
      padding-bottom: 10rem;    /* ← increased from 2rem to 6rem */
    }
  </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1,1,4,1])
with col3:
    st.markdown(
      '<h1 style="color:Red; background:Lavender; text-align:center;">'
      'Book recommendation</h1>',
      unsafe_allow_html=True
    )
st.write("---"*20)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Ask Question", "About the App"],
        icons=["question-circle", "info-circle"],
        default_index=0
    )
    st.write("---")
    if st.button("Clear Chat History"):
        st.session_state.clear()

# — Session State
if "step" not in st.session_state:
    st.session_state.step = "start"
if "round" not in st.session_state:
    st.session_state.round = 0
if "last_six" not in st.session_state:
    st.session_state.last_six = pd.DataFrame()

def show_six(popular: bool):
    pool = df[df["Popularity"] == popular]
    six = pool.sample(min(6, len(pool)))
    st.session_state.last_six = six
    with st.chat_message("assistant"):
        st.write(f"Here are six {'popular' if popular else 'non-popular'} books:")
        for _, r in six.iterrows():
            st.write(f"Title: {r.Title}")
            st.write(f"Author: {r.Author}")
            st.write(f"Summary: {r.Description[:120]}…")
            st.write("---")
        st.write("Do you like any of these? (reply with a title or ‘no’)")

# 1) Start
if st.session_state.step == "start":
    if st.button("📚 I want to read something interesting", key="btn_browse_start"):
        st.session_state.step = "browse"
        st.session_state.round = 1
        show_six(popular=True)

    if st.button("💬 Ask me anything", key="btn_ask_anything"):
        st.session_state.step = "ask"

# 2) Browse Loop
if st.session_state.step == "browse":
    reply = st.chat_input("Your answer:")
    if reply:
        with st.chat_message("user"):
            st.markdown(reply)

       # normalize
        text = reply.strip()
        titles = st.session_state.last_six["Title"].tolist()
        # fuzzy‐find the best title match
        match, score, _ = process.extractOne(
            query=text,
            choices=titles,
            scorer=fuzz.token_sort_ratio,
        )
        matched = match if score >= 60 else None  # require at least 60% similarity
        if text in ("no","nah","no thanks","not really") and st.session_state.round < 3:
            st.session_state.round += 1
            show_six(popular=(st.session_state.round < 3))
        else:
            # User picked a title → ask backend
            # Did we match one of the shown titles?
            if matched:
                chosen = matched
            else:
                st.error("Sorry, I didn’t recognize that title. Please pick one of the six I showed.")
                st.stop()
            prompt = f"Recommend a book similar to '{chosen}'."
            with st.chat_message("assistant"):
                st.markdown("Looking up similar titles…")
            try:
                res = requests.post(API_URL, json={"query": prompt})
                res.raise_for_status()
                data = res.json()
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

            # Render final recommendations
            with st.chat_message("assistant"):
                plain = "\n".join(
                    line.lstrip("# ").rstrip()
                    for line in data["answer"].splitlines()
                )
                html = (
                    "<div style='font-size:14px; line-height:1.4;'>"
                    + plain.replace("\n", "<br>")
                    + "</div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                st.caption(f"Referred: *{data['sources']}*")

            # Reset for next session
            st.session_state.step = "start"
            st.session_state.round = 0