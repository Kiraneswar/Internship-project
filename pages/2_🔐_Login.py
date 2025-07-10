import streamlit as st
import pyrebase

firebaseConfig = {
    "apiKey": st.secrets["firebase"]["apiKey"],
    "authDomain": st.secrets["firebase"]["authDomain"],
    "databaseURL": st.secrets["firebase"]["databaseURL"],
    "projectId": st.secrets["firebase"]["projectId"],
    "storageBucket": st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId": st.secrets["firebase"]["appId"]
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

st.title("Login to Knowledge GPT")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if not email or not password:
        st.warning("Please enter both email and password.")
    else:
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            user_data = db.child("users").child(uid).get().val()
            name = user_data.get("name", "User")

            st.session_state["user_uid"] = uid
            st.session_state["user_name"] = name
            st.session_state["user_email"] = email

            st.success(f"Welcome, {name}!")
            st.switch_page("pages/3_💡_KnowledgeGPT.py")
        except Exception as e:
            st.error("Login failed. Please check your credentials.")
            st.error(f"Error details: {e}")
