from flask import Flask, request, render_template
from models.clip_model import process_with_clip
from models.blip_model import generate_answer_with_blip

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form.get("question")
        image = request.files.get("image")

        if image and question:
            image_path = f"static/uploads/{image.filename}"
            image.save(image_path)

            # Process with CLIP and BLIP
            similarity = process_with_clip(image_path, question)
            if similarity.item() > 0.5:
                answer = generate_answer_with_blip(image_path, question)
            else:
                answer = "The question might not be relevant to the image."

            return render_template("index.html", answer=answer, image_path=image_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
