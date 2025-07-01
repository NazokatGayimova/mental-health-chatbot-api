import streamlit as st
import requests

st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Mental Health Support Chatbot")
st.write("This chatbot gives empathetic responses based on real mental health conversations.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    st.session_state.chat_history.append(("You", user_input))

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"message": user_input},
            timeout=10
        )
        if response.status_code == 200:
            bot_reply = response.json().get("response", "")
            if bot_reply.strip():
                st.session_state.chat_history.append(("Bot", bot_reply))
            else:
                st.session_state.chat_history.append(("Bot", "I'm here for you. Please tell me more."))
        else:
            st.session_state.chat_history.append(("Bot", "⚠️ Server error. Please try again."))
    except Exception as e:
        st.session_state.chat_history.append(("Bot", f"⚠️ Error: {e}"))

st.markdown("---")
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You**: {message}")
    else:
        st.markdown(f"**Bot**: {message}")

