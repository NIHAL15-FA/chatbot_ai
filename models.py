import torch
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize models
resnet50 = None
classifier = None
sentence_model = None

def initialize_models():
    global resnet50, classifier, sentence_model
    if resnet50 is None:
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))

    if classifier is None:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    if sentence_model is None:
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def extract_image_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(img_tensor).squeeze().cpu().numpy()
    return features

def get_text_embedding(features):
    description = " ".join([f"{key}: {value}" for key, value in features.items() if value])
    return sentence_model.encode(description)

def apply_synonym_mapping(description):
    synonym_mapping = {
        'male': ['man', 'male', 'gentleman'],
        # Add other mappings...
    }
    for key, synonyms in synonym_mapping.items():
        for synonym in synonyms:
            description = re.sub(rf"\b{synonym}\b", key, description, flags=re.IGNORECASE)
    return description
