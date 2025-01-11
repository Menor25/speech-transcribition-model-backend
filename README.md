# Backend for Speech Recognition and Text Processing App

This repository contains the **backend** of the Speech Recognition and Text Processing application. The backend handles audio file uploads, speech-to-text transcription, text processing, and feedback submission.

## Features
- Processes audio files for speech recognition using **Wav2Vec 2.0**.
- Performs text classification for intent recognition using **BERT**.
- Provides an API endpoint for word predictions to correct transcription errors.
- Handles feedback and ratings from users.
- Flask-powered API with HTML templates.

## Technologies Used
- **Python**: Backend logic.
- **Flask**: Web framework.
- **PyTorch**: For deep learning models.
- **Transformers**: Pre-trained models from Hugging Face.
- **Render**: For hosting.

## File Structure



## API Endpoints
- **POST `/transcribe`**: Upload an audio file and return its transcription.
- **POST `/predict_word`**: Suggest words for correcting transcription errors.
- **POST `/submit_feedback`**: Submit feedback and ratings for transcription results.

## Deployment
The backend is hosted on **Render** for reliable API hosting.

### Steps to Deploy
1. Push this repository to GitHub.
2. Link the repository to your Render account.
3. Specify the `requirements.txt` file for dependencies.
4. Set the entry point to `app.py`.

## Installation (Local)
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/backend.git
   cd backend
