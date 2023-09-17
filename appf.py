import html
import cv2
import imageio
import numpy as np
from flask import Flask, jsonify, render_template, request
from googleapiclient.discovery import build
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from transformers import AutoTokenizer, TFAutoModelForCausalLM

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('front.html')

@app.route('/report')
def report():
    return render_template('report.html')

# Route for the fake_news_finder page
@app.route('/fake_news_finder', methods=['GET', 'POST'])
def fake_news_finder():
    if request.method == 'POST':
        # Get the keyword and rate limit from the form
        keyword = request.form['keyword']
        rate_limit = int(request.form['rate_limit'])
        
        api_key = "AIzaSyBPHs1Pq49RrKiW1BIFl2uJHYrwa7cpeyY"
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        model_name = "gpt2"
        model = TFAutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Search YouTube for videos with the keyword
        search_response = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=rate_limit
        ).execute()

        videos = []
        for item in search_response["items"]:
            video_title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            channel_title = item["snippet"]["channelTitle"]
            published_at = item["snippet"]["publishedAt"]

            # Retrieve video details
            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            view_count = video_response["items"][0]["statistics"].get("viewCount", 0)
            like_count = video_response["items"][0]["statistics"].get("likeCount", 0)

            videos.append({
                'video_title': video_title,
                'video_id': video_id,
                'channel_title': channel_title,
                'published_at': published_at,
                'view_count': view_count,
                'like_count': like_count
            })

        # Retrieve keyword-related comments
        comments = []
        for video in videos:
            video_id = video['video_id']
            comment_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                searchTerms=keyword,
                maxResults=rate_limit
            ).execute()

            if "items" in comment_response:
                for comment in comment_response["items"]:
                    comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comment_text = html.unescape(comment_text)
                    comments.append(comment_text)

        prompt = f"Is it true that {keyword}?"
        inputs = tokenizer.encode(prompt, return_tensors="tf", add_special_tokens=True)

        # Generate response from GPT-2 model
        output = model.generate(inputs, max_length=50, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return render_template('fake_news_finder.html', videos=videos, comments=comments, response=response)

    return render_template('fake_news_finder.html', videos=None, comments=None, response=None)


# Route for the fake_video_detector page
@app.route('/fake_video_detector/', methods=['GET', 'POST'])
def fake_video_detector():
    if request.method == 'POST':
        if 'video-file' in request.files:
            video_file = request.files['video-file']
            temp_file_path = 'temp_video.mp4'
            video_file.save(temp_file_path)

            # Process the video
            duration = calculate_video_duration(temp_file_path)
            predictions = process_video(temp_file_path, model)

            if predictions:
                classification, confidence = classify_video(predictions)
                result = {
                    'classification': classification,
                    'confidence': confidence,
                    'duration': duration
                }
                return jsonify(result)  # Return the result as JSON
            else:
                return jsonify({'error': 'Error occurred during video processing.'})

        else:
            return jsonify({'error': 'No video file uploaded.'}), 400

    return render_template('fake_video_detector.html')


# Route for the working page
@app.route('/working/')
def working():
    return render_template('working.html')

# Route for the privacy policy page
@app.route('/privacy_policy/')
def privacy_policy():
    return render_template('privacy_policy.html')

def calculate_video_duration(video_path):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return None

    duration = video.get_meta_data()['duration']
    video.close()

    return duration

def process_video(video_path, model):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Unable to open the video file. {e}")
        return []

    frame_count = len(video)
    frame_height, frame_width = video.get_meta_data()['source_size'][:2]

    print(f"Video info: {frame_count} frames, {frame_width}x{frame_height} resolution")

    predictions = []

    try:
        for i, frame in enumerate(video):
            # Resize and preprocess the frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)

            # Make a prediction
            prediction = model.predict(frame)
            predictions.append(prediction[0][0])  # Append the prediction result (fake score)

    except Exception as e:
        print(f"Error occurred during video processing. {e}")
        return []

    return predictions

def classify_video(predictions, threshold=0.5):
    if not predictions:
        return 'unknown', 0

    fake_count = sum(1 for score in predictions if score > threshold)
    real_count = len(predictions) - fake_count

    fake_score = fake_count / len(predictions)
    real_score = real_count / len(predictions)

    if fake_score > real_score:
        return 'fake', fake_score
    else:
        return 'real', real_score
    
    
# Define the route to handle form submissions
@app.route('/report', methods=['POST'])
def report_user():
    if request.method == 'POST':
        # Get the submitted report user
        report_user = request.form.get('reportUser')

        # Save the report user to a file (you can use any format you prefer)
        with open('report_users.txt', 'a') as file:
            file.write(report_user + '\n')

        return "Report submitted successfully."

if __name__ == '__main__':
    # Load the pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.001), metrics=['accuracy'])

    app.run(debug=True)
