import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 1. Load the pre-trained model and its processor
# A "processor" prepares the data (image + text) for the model.
# We're using a "ViLT" (Vision-and-Language Transformer) model 
# fine-tuned for visual question answering.
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 2. Prediction
# Invoked on "Submit" button click
def answer_question(image, text):
    try:
        # 3. Prepare the inputs
        # The processor converts the raw image and text query into
        # the specific numerical format the model expects.
        encoding = processor(image, text, return_tensors="pt")

        # 4. Run the model
        # We pass the processed inputs to the model...
        outputs = model(**encoding)
        logits = outputs.logits

        # 5. Decode the answer
        # The model's raw output ("logits") is just a set of numbers.
        # We find the highest-scoring number (the "argmax") and use
        # the model's config to turn it back into a readable word.
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        
        return answer
        
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I had trouble processing that. Try a different image or question."

# 6. Create the Gradio web interface
# This one line of code builds the entire UI!
iface = gr.Interface(
    fn=answer_question,  # The function to call
    inputs=[
        gr.Image(type="pil"), # An image upload box (provides a PIL image)
        gr.Textbox(label="Ask a question about the image...") # A text input box
    ],
    outputs=gr.Textbox(label="Answer"), # A text output box
    title="🤖 Multimodal AI: Visual Question Answering",
    description="Upload an image and ask any question about it. (Model: dandelin/vilt-b32-finetuned-vqa)"
)

# 7. Launch the app!
iface.launch()