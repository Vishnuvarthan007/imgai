import streamlit as st
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\2355389\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe'
import cv2
import pytesseract
import networkx as nx
import pandas as pd
import json
import os
from groq import Client
from PIL import Image
import numpy as np

# Initialize Groq Client
client = Client(api_key="YOUR_GROQ_API_KEY")

class CallFlowAnalyzer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.extracted_data = {}
        self.ocr_config = "--psm 6"

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def extract_text_from_image(self, image):
        processed_image = self.preprocess_image(image)
        text = pytesseract.image_to_string(processed_image, config=self.ocr_config)
        self.extracted_data["raw_text"] = text[:1000]
        return text

    def parse_text_to_graph(self, extracted_text):
        lines = extracted_text.split("\n")
        for line in lines:
            if "->" in line:
                nodes = line.split("->")
                parent, child = nodes[0].strip(), nodes[1].strip()
                self.graph.add_edge(parent, child)
        return self.graph

    def generate_test_cases(self):
        test_cases = [{"Step": edge[0], "Expected Outcome": edge[1]} for edge in self.graph.edges]
        df = pd.DataFrame(test_cases)
        df.to_excel("test_cases.xlsx", index=False)
        return "test_cases.xlsx"

    def explain_call_flow(self):
        if not self.graph.nodes:
            return "The call flow diagram is empty or could not be parsed."
        
        explanation = "The call flow starts with: " + list(self.graph.nodes)[0] + "\n"
        for node in list(self.graph.nodes)[:10]:  
            next_steps = list(self.graph.successors(node))
            if next_steps:
                explanation += f"From '{node}', the call can proceed to {', '.join(next_steps)}.\n"
        return explanation

# Streamlit UI
st.title("AI Call Flow Analyzer")
st.sidebar.header("Upload a Call Flow Diagram")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

analyzer = CallFlowAnalyzer()

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Call Flow Diagram", use_column_width=True)

    extracted_text = analyzer.extract_text_from_image(image)
    analyzer.parse_text_to_graph(extracted_text)
    
    if st.button("Explain Call Flow"):
        st.text(analyzer.explain_call_flow())

    if st.button("Generate Test Cases"):
        file_path = analyzer.generate_test_cases()
        with open(file_path, "rb") as file:
            st.download_button("Download Test Cases", file, file_name="test_cases.xlsx")
