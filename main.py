import gradio as gr
from transformers import pipeline

# Load your model from Hugging Face
pipe = pipeline("text-classification", model="kawudiv/accessabilitySVC")

def classify(text):
    result = pipe(text)
    return f"{result[0]['label']} ({round(result[0]['score'] * 100, 2)}%)"

demo = gr.Interface(fn=classify, inputs="text", outputs="text", title="Accessibility Classifier")

if __name__ == "__main__":
    demo.launch()
