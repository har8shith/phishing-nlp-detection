import streamlit as st, requests, os
API_URL = os.getenv("PHISHING_API_URL", "http://localhost:8000/api")

def call_analyze_api(email_text: str, model: str):
    payload = {"email_text": email_text, "model": model}
    resp = requests.post(f"{API_URL}/analyze", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def main():
    st.title("Phishing Email Detection (NLP Project)")
    model_choice = st.selectbox("Select model", ["bert", "tfidf"])
    email_text = st.text_area("Paste email content here", height=250)
    if st.button("Analyze"):
        if not email_text.strip():
            st.warning("Please paste an email first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    result = call_analyze_api(email_text, model_choice)
                except Exception as e:
                    st.error(f"Error calling API: {e}")
                    return
            label = result["label"]
            prob = result["probability"]
            extracted = result.get("extracted_info", {})
            if label == "phishing":
                st.error(f"Prediction: PHISHING (score={prob:.3f})")
            else:
                st.success(f"Prediction: LEGITIMATE (score={prob:.3f})")
            st.subheader("Extracted Information")
            st.json(extracted or {})
    st.caption("Team: Banothu Harshith, Shashi Reddy, Ineesh Reddy, Bala Bhargav, Manideep Kaparthi")

if __name__ == "__main__":
    main()
