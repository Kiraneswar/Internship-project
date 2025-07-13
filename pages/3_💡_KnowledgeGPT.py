import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import pipeline
import datetime

@st.cache_resource
def init_firebase():
    import firebase_admin
    from firebase_admin import credentials, firestore

    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("firebase_config.json")
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.error(f"Firebase init failed: {e}")
            return None  # Fail safely

    return firestore.client()


db = init_firebase()

# ---------------- Configure Gemini ----------------
genai.configure(api_key="your gemini api key")

@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-1.5-flash-8b")

model = load_model()

# ---------------- Hugging Face Summarizer ----------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="Falconsai/text_summarization")

summarizer = load_summarizer()

# ---------------- Page Settings ----------------
st.set_page_config(page_title="Knowledge GPT", layout="wide")
if "user_uid" not in st.session_state:
    st.warning("You must log in to access Knowledge GPT.")
    st.stop()


# ---------------- Session State Init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "saving_chat" not in st.session_state:
    st.session_state.saving_chat = False
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "show_safety" not in st.session_state:
    st.session_state.show_safety = False
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "last_safety_score" not in st.session_state:
    st.session_state.last_safety_score = 100
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "flash_index" not in st.session_state:
    st.session_state.flash_index = 0
if "show_flashcard_maker" not in st.session_state:
    st.session_state.show_flashcard_maker = False
if "show_summarizer" not in st.session_state:
    st.session_state.show_summarizer = False

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.big-font { font-size:30px !important; }
.user-info {
    display: flex;
    align-items: center;
    font-size: 20px;
    font-weight: bold;
    padding-bottom: 10px;
}
.user-info img {
    border-radius: 50%;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.header("Knowledge GPT")

# ---------------- Features ----------------
col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.markdown('<p class="big-font"> 🧠 AI-Generated Explanations</p>', unsafe_allow_html=True)
with col2:
    st.markdown('<p class="big-font"> ❓ Quiz Generation Engine</p>', unsafe_allow_html=True)
with col3:
    st.markdown('<p class="big-font"> 📝 Flashcard Maker</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.markdown('<p class="big-font"> 📄 Text Summarizer</p>', unsafe_allow_html=True)
with col2:
    st.markdown('<p class="big-font"> 🔄 Feedback Loop + Adaptation</p>', unsafe_allow_html=True)
with col3:
    st.markdown('<p class="big-font"> 🛡️ Ethical AI Mode</p>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown(f"""
<div class="user-info">
    <img src="data:image/png;base64,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" width="40">
    <span>{st.session_state.get('user_name', 'Guest')}</span>
</div>
""", unsafe_allow_html=True)


    if not st.session_state.saving_chat:
        if st.button("💾 Save Chat"):
            st.session_state.saving_chat = True
    else:
        new_name = st.text_input("Enter name to save chat", placeholder="e.g. Photosynthesis")
        if st.button("✅ Confirm Save"):
            if new_name.strip():
                st.session_state.chat_history[new_name] = st.session_state.messages.copy()
                st.session_state.messages = []
                st.session_state.saving_chat = False
                st.success(f"Chat saved as '{new_name}'. Starting new chat.")
                        # 🔥 Save chat to Firebase
            try:
                user = st.session_state.get("user_name", "Guest")
                timestamp = datetime.datetime.now().isoformat()
                doc_ref = db.collection("chats").document()
                doc_ref.set({
                    "user": user,
                    "chat_name": new_name,
                    "timestamp": timestamp,
                    "messages": st.session_state.chat_history[new_name]
                })
                st.success("📤 Chat uploaded to Firebase.")
            except Exception as firebase_error:
                st.warning(f"⚠️ Failed to store chat in Firebase: {firebase_error}")
         


    st.markdown("### 📚 Previous Chats")
    for name, history in reversed(st.session_state.chat_history.items()):
        with st.expander(f"🗂 {name}"):
            for msg in history:
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"{icon} **{msg['role']}**: {msg['content']}")

# ---------------- Utility Functions ----------------
def analyze_safety(messages):
    keywords = ["kill", "hate", "hack", "bomb", "attack", "explode","homicide"]
    unsafe_count = sum(1 for m in messages if any(w in m["content"].lower() for w in keywords))
    total = len(messages)
    safe_percent = 100 - int((unsafe_count / total) * 100) if total > 0 else 100
    return safe_percent

def pie_chart_to_base64(safe, unsafe):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie([safe, unsafe], labels=["Safe", "Unsafe"], autopct='%1.1f%%', colors=["#4CAF50", "#F44336"])
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ---------------- Chat Display ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat Input ----------------
prompt = st.chat_input("Ask a question...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            chat = model.start_chat(history=[
                {"role": m["role"], "parts": [m["content"]]}
                for m in st.session_state.messages if m["role"] in ["user", "assistant"]
            ])
            result = chat.send_message(prompt).text
        except Exception as e:
            result = f"⚠️ Error: {str(e)}"

        placeholder.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})

# ---------------- Collapsible Features ----------------
st.markdown("---")

# --- FLASHCARD MAKER ---
if st.button("📘 Flashcard Maker", use_container_width=True):
    st.session_state.show_flashcard_maker = not st.session_state.show_flashcard_maker

if st.session_state.show_flashcard_maker:
    st.subheader("🎓 Generate Flashcards for Any Topic")

    def generate_flashcards(topic):
        prompt = f"""Create 10 educational flashcards for the topic '{topic}'. 
Each flashcard should be formatted as:
Q: <Term or Question>
A: <Short, clear definition or answer>"""
        try:
            response = model.generate_content(prompt)
            raw_cards = response.text.strip().split("\n")

            flashcards = []
            q, a = "", ""
            for line in raw_cards:
                if line.startswith("Q:"):
                    q = line[2:].strip()
                elif line.startswith("A:"):
                    a = line[2:].strip()
                    if q and a:
                        flashcards.append((q, a))
                        q, a = "", ""
            return flashcards[:10]
        except Exception as e:
            st.error(f"Error generating flashcards: {e}")
            return []

    with st.form("flashcard_form"):
        topic_input = st.text_input("Enter a topic", placeholder="e.g. Photosynthesis")
        submitted = st.form_submit_button("Generate Flashcards")
        if submitted and topic_input.strip():
            st.session_state.flashcards = generate_flashcards(topic_input.strip())
            st.session_state.flash_index = 0

    if st.session_state.flashcards:
        q, a = st.session_state.flashcards[st.session_state.flash_index]
        st.markdown(f"""
        <div style="background-color:black;padding:20px;border-radius:15px;box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h4 style="color:#2c3e50;">🃏 <b>Flashcard {st.session_state.flash_index + 1} of {len(st.session_state.flashcards)}</b></h4>
            <p><strong>Q:</strong> {q}</p>
            <p><strong>A:</strong> {a}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("➡️ Next Flashcard"):
                st.session_state.flash_index = (st.session_state.flash_index + 1) % len(st.session_state.flashcards)

# --- SAFETY ANALYSIS ---
if st.button("🛡️ Safety Analysis", use_container_width=True):
    st.session_state.show_safety = not st.session_state.show_safety

if st.session_state.show_safety:
    safe = analyze_safety(st.session_state.messages)
    unsafe = 100 - safe
    st.session_state.last_safety_score = safe
    img_base64 = pie_chart_to_base64(safe, unsafe)

    st.markdown(f"""
    <div style="padding:20px;background-color:#eef9f1;border-radius:15px;">
        <h4>Safety Analysis</h4>
        <p>Your chat is <strong>{safe}% safe</strong>.</p>
        <img src="data:image/png;base64,{img_base64}" width="200">
    </div>
    """, unsafe_allow_html=True)

    if safe < 40:
        st.warning("⚠️ Chat safety dropped below 40%. Please pause and contact someone nearby.")

# --- FEEDBACK SUMMARY ---
if st.button("💬 Feedback Summary", use_container_width=True):
    st.session_state.show_feedback = not st.session_state.show_feedback

if st.session_state.show_feedback:
    with st.form("feedback_form"):
        st.subheader("🗣️ Share Feedback About This Chat")
        feedback_text = st.text_area("Your feedback here...", value=st.session_state.feedback)
        submit_feedback = st.form_submit_button("Submit")
        if submit_feedback:
            if feedback_text.strip():
                st.session_state.feedback = feedback_text.strip()
                st.success("✅ Feedback saved.")
            else:
                st.warning("Please enter some feedback.")

# --- TEXT SUMMARIZER (Hugging Face) ---
if st.button("📄 Text Summarizer", use_container_width=True):
    st.session_state.show_summarizer = not st.session_state.show_summarizer

if st.session_state.show_summarizer:
    st.subheader("📚 Summarize Long Text with Falconsai Model")
    with st.form("summarization_form"):
        input_text = st.text_area("Paste or type content to summarize:", height=250)
        submitted = st.form_submit_button("📝 Summarize")
        if submitted and input_text.strip():
            with st.spinner("⏳ Generating summary..."):
                try:
                    summary = summarizer(input_text, max_length=500, min_length=50, do_sample=False)
                    st.success("✅ Summary generated:")
                    st.markdown(f"**🔍 Summary:**\n\n{summary[0]['summary_text']}")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
