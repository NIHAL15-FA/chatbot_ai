import gradio as gr
from utils import match_description_to_images
from models import initialize_models

def gradio_search(description):
    initialize_models()
    matches = match_description_to_images(description)
    
    if matches:
        results = []
        for match in matches:
            image_info = f"{match['image_name']} - Similarity: {match['similarity']:.2f}"
            results.append((match['image_path'], image_info))
        return results
    else:
        return [("No matches found", None)]

# Gradio Interface
interface = gr.Interface(fn=gradio_search, inputs="text", outputs="gallery", title="Criminal Face Matching")
interface.launch()
