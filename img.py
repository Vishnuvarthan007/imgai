!pip install streamlit opencv-python pytesseract networkx pandas groq pyngrok
import streamlit as st
import cv2
import pytesseract
import networkx as nx
import pandas as pd
import json
import tempfile
from groq import Client
from PIL import Image

# Initialize AI Client
client = Client(api_key="YOUR_GROQ_API_KEY")

class CallFlowAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.extracted_data = {}

    def extract_text_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        text = pytesseract.image_to_string(blurred, config="--psm 6")
        self.extracted_data["raw_text"] = text[:1000]
        return text

    def parse_text_to_graph(self, extracted_text):
        lines = extracted_text.split("\n")
        for line in lines:
            if "->" in line:
                nodes = line.split("->")
                parent, child = nodes[0].strip(), nodes[1].strip()
                self.graph.add_edge(parent, child)

        self.extracted_data["graph"] = {
            "nodes": list(self.graph.nodes)[:50],
            "edges": list(self.graph.edges)[:50]
        }

    def explain_call_flow(self):
        if not self.graph.nodes:
            return "The call flow diagram is empty or could not be parsed."

        explanation = "Call flow starts at: " + list(self.graph.nodes)[0] + "\n"
        for node in list(self.graph.nodes)[:50]:
            next_steps = list(self.graph.successors(node))
            if next_steps:
                explanation += f"From '{node}', the call can go to: {', '.join(next_steps)}.\n"
        self.extracted_data["explanation"] = explanation[:1000]
        return explanation

    def generate_test_cases(self):
        test_cases = [{"Step": parent, "Expected Outcome": child} for parent, child in self.graph.edges]
        df = pd.DataFrame(test_cases)
        return df

    def chat_with_ai(self, query):
        request_data = json.dumps(self.extracted_data, indent=2)[:6000]
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You analyze call flow diagrams."},
                {"role": "user", "content": f"Analyze this call flow: {query}\n{request_data}"}
            ]
        )
        return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📞 Call Flow AI Chatbot")

uploaded_file = st.file_uploader("Upload Call Flow Diagram", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Diagram", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    bot = CallFlowAnalyzer()
    extracted_text = bot.extract_text_from_image(cv2.imread(temp_path))
    bot.parse_text_to_graph(extracted_text)

    st.subheader("📜 Extracted Call Flow Explanation")
    st.write(bot.explain_call_flow())

    if st.button("📋 Generate Test Cases"):
        df = bot.generate_test_cases()
        st.write(df)
        df.to_excel("test_cases.xlsx", index=False)
        with open("test_cases.xlsx", "rb") as f:
            st.download_button("Download Test Cases", f, "test_cases.xlsx")

query = st.text_input("💬 Ask about the Call Flow")
if query:
    response = bot.chat_with_ai(query)
    st.write(response)

