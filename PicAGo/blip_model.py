from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return model, processor

# Generate answer using BLIP
def generate_answer_with_blip(image_path, question):
    model, processor = load_blip_model()

    # Preprocess image
    image = Image.open(image_path).convert("RGB")

    # Prepare input for BLIP
    inputs = processor(image, question, return_tensors="pt").to(device)

    # Generate answer
    with torch.no_grad():
        out = model.generate(**inputs)

    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer
