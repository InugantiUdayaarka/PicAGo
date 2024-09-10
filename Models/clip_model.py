import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

# Process image and question with CLIP
def process_with_clip(image_path, question):
    model, preprocess = load_clip_model()

    # Preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize the question
    text = clip.tokenize([question]).to(device)

    # Encode both image and text
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Compute similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity
