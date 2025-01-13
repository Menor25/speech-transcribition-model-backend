# Required Libraries
import torchaudio
import onnxruntime as ort
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
        print("Loading ONNX Speech Model...")
        _speech_model = ort.InferenceSession("wav2vec2.onnx")  # Path to ONNX model
    return _speech_model

def get_nlp_pipeline():
    global _nlp_model
    if _nlp_model is None:
        print("Loading ONNX NLP Model...")
        _nlp_model = ort.InferenceSession("distilbert.onnx")  # Path to ONNX model
    return _nlp_model

# Function for Speech Recognition
def transcribe_audio(audio_path):
    ort_session = get_speech_model()
    
    # Load and preprocess the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Limit the audio duration to 30 seconds
    max_duration = 30  # seconds
    if waveform.size(1) > sample_rate * max_duration:
        waveform = waveform[:, :sample_rate * max_duration]

    # Resample to 16kHz
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform).squeeze()

    # Prepare input for ONNX Runtime
    inputs = {"input": waveform.numpy()}
    outputs = ort_session.run(None, inputs)
    
    # Decode the transcription
    transcription = outputs[0]  # Adjust based on your ONNX model's output
    return transcription

# Function for NLP Intent Recognition
def process_text(text):
    ort_session = get_nlp_pipeline()
    
    # Prepare input for ONNX Runtime
    inputs = {"input_text": [text]}  # Adjust based on your ONNX model's input
    outputs = ort_session.run(None, inputs)
    
    return outputs[0]  # Adjust based on your ONNX model's output

# Flask Routes
@app.route('/')
def index():
    return jsonify({"message": "Backend API is running. The frontend is hosted separately."})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a valid audio file."}), 400

    save_directory = "uploaded_audio"

    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_path = os.path.join(save_directory, file.filename)
    file.save(file_path)

    try:
        # Speech Recognition
        transcription = transcribe_audio(file_path)
        return jsonify({"transcription": transcription})
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": "An error occurred during transcription."}), 500

@app.route('/process_text', methods=['POST'])
def process_text_route():
    data = request.get_json()
    text = data.get('text', "")

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # NLP Processing
        nlp_result = process_text(text)
        return jsonify({"nlp_result": nlp_result})
    except Exception as e:
        print(f"Error during NLP processing: {e}")
        return jsonify({"error": "An error occurred during text processing."}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        rating = request.form.get('rating')
        feedback = request.form.get('feedback')

        if not rating or not feedback:
            return jsonify({"error": "Both rating and feedback are required."}), 400

        # Log or store the feedback (e.g., save to a file or database)
        print(f"Rating: {rating}, Feedback: {feedback}")
        return jsonify({"message": "Feedback submitted successfully."})
    except Exception as e:
        print(f"Error during feedback submission: {e}")
        return jsonify({"error": "An error occurred while submitting your feedback."}), 500

# Limit upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080, threads=1)  # Limit threads to 1 to reduce memory usage
