# streamlit_app.py
import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="🖼️ Image Processing Chatbot with RAG", layout="wide")
st.title("🖼️ Image Processing Chatbot with RAG")

# API input
api_url = st.text_input("API URL", value="http://localhost:8000")

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inputs
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
user_prompt = st.text_input("Your question (about image or text)", key="user_prompt")

if st.button("Send") and user_prompt:
    answer = ""
    retrieved = []

    try:
        # CASE 1: Image provided
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            if user_prompt:
                response = requests.post(f"{api_url}/api/upload-image-prompt", files=files, data={"prompt": user_prompt})
            else:
                response = requests.post(f"{api_url}/api/upload-image", files=files)
        else:
            # CASE 2: Text-only query
            response = requests.post(f"{api_url}/api/prompt-text", json={"prompt": user_prompt})

        if response.ok:
            data = response.json()
            answer = data.get("response", "_No response_")
            retrieved = data.get("retrieved", [])
        else:
            answer = f"❌ Error: {response.status_code}"
    except Exception as e:
        answer = f"🚫 Exception: {e}"

    # Save to session state
    st.session_state.chat_history.append({
        "prompt": user_prompt,
        "retrieved": retrieved,
        "answer": answer
    })

# Display chat
st.markdown("---")
st.subheader("💬 Chat History with RAG")

for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['prompt']}")
    if chat["retrieved"]:
        st.markdown("**🔍 Top Matches from Database:**")
        for item in chat["retrieved"]:
            st.markdown(f"- **{item['food']}** (Score: {item['score']:.2f})")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")
