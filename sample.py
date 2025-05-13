from pyngrok import ngrok
import os

# Open a tunnel to port 8501 (Streamlit default)
public_url = ngrok.connect(8501)
print("Public URL:", public_url)

# Run your Streamlit app
os.system("streamlit run RAG_frontend.py")
