# Required Libraries
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
from flask import Flask, request, jsonify
import os

# Initialize Flask Application
app = Flask(__name__)

# Global Cache for Models
_speech_model = None
_nlp_model = None

# Lazy Loading Functions
def get_speech_model():
    global _speech_model
    if _speech_model is None:
        print("Loading Wave2Vec 2.0 Model...")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        _speech_model = (processor, model)
    return _speech_model

def get_nlp_pipeline():
    global _nlp_model
    if _nlp_model is None:
        print("Loading NLP Model...")
        _nlp_model = pipeline("text-classification", model="distilbert-base-uncased")
    return _nlp_model

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
    return jsonify({"message": "Backend API is running. The frontend is hosted separately."})

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

    # Return the transcription result as JSON
    return jsonify({"transcription": transcription})

@app.route('/process_text', methods=['POST'])
def process_text_route():
    data = request.get_json()
    text = data.get('text', "")

    # NLP Processing
    nlp_result = process_text(text)

    return jsonify({"nlp_result": nlp_result})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        rating = request.form['rating']
        feedback = request.form['feedback']

        # Log or store the feedback (e.g., save to database or log file)
        print(f"Rating: {rating}, Feedback: {feedback}")

        # Return success response
        return jsonify({"message": "Feedback submitted successfully."})
    except Exception as e:
        return jsonify({"error": "An error occurred while submitting your feedback."}), 500

# Limit upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
