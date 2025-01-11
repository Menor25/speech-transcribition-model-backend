# Required Libraries
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
from flask import Flask, render_template, request, jsonify
import os

# Initialize Flask Application
app = Flask(__name__, template_folder='templates', static_folder='templates/static')

# Lazy Loading Functions
def get_speech_model():
    print("Loading Wave2Vec 2.0 Model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

def get_nlp_pipeline():
    print("Loading NLP Model...")
    nlp_pipeline = pipeline("text-classification", model="distilbert-base-uncased")
    return nlp_pipeline

# Function for Speech Recognition
def transcribe_audio(audio_path):
    processor, model = get_speech_model()
    # Load and preprocess the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform).squeeze()

    # Tokenize and predict
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Function for NLP Intent Recognition
def process_text(text):
    nlp_pipeline = get_nlp_pipeline()
    return nlp_pipeline(text)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']

    save_directory = "uploaded_audio"

    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, file.filename)
    file.save(file_path)

    # Speech Recognition
    transcription = transcribe_audio(file_path)

    # Render the transcription result on a new page
    return render_template('result.html', transcription=transcription)

@app.route('/predict_word', methods=['POST'])
def predict_word():
    data = request.get_json()
    context = data.get('context', "")
    last_word = context.split(" ")[-1] if context else ""

    # Fetch word predictions based on last_word
    predictions = get_word_predictions(last_word)  # Implement this function as needed
    return jsonify(predictions)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        rating = request.form['rating']
        feedback = request.form['feedback']

        # Log or store the feedback (you can save this in a database or file)
        print(f"Rating: {rating}, Feedback: {feedback}")

        # Redirect to a thank-you page or render a success message
        return render_template('thank_you.html', rating=rating, feedback=feedback)
    except Exception as e:
        return render_template('error.html', message="An error occurred while submitting your feedback.")

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
