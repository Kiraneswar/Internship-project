import streamlit as st
import pyrebase

# Firebase config
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

st.title("Sign Up for Knowledge GPT")

name = st.text_input("Name")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

if st.button("Sign Up"):
    if not name or not email or not password:
        st.warning("Please fill in all fields.")
    elif password != confirm_password:
        st.error("Passwords do not match.")
    else:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']
            db.child("users").child(uid).set({"name": name, "email": email})

            # Store in session
            st.session_state["user_uid"] = uid
            st.session_state["user_name"] = name
            st.session_state["user_email"] = email

            st.success("Signup successful! Redirecting...")
            st.switch_page("pages/3_💡_KnowledgeGPT.py")
        except Exception as e:
            st.error(f"Signup failed: {e}")
