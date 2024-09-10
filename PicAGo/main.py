import os
from models.clip_model import process_with_clip
from models.blip_model import generate_answer_with_blip

def main():
    # Take image path from the user
    image_path = input("Enter the path to your image: ")

    if not os.path.exists(image_path):
        print("Image file does not exist!")
        return

    # Take the question from the user
    question = input("Enter your question about the image: ")

    # Use CLIP to determine if the question is relevant (optional step)
    similarity = process_with_clip(image_path, question)
    print(f"CLIP Similarity Score: {similarity.item()}")

    # If similarity score is acceptable, proceed with BLIP answer generation
    if similarity.item() > 0.5:  # Threshold for similarity
        answer = generate_answer_with_blip(image_path, question)
        print(f"Answer: {answer}")
    else:
        print("The question might not be relevant to the image.")

if __name__ == "__main__":
    main()
