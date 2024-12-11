from flask import Flask, request, render_template
import numpy as np
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from sklearn.preprocessing import LabelEncoder

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Load pre-trained TensorFlow model
model = tf.keras.models.load_model('resume_classification_model.keras')

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def vectorize_with_word2vec(text, model):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        resume_text = request.form.get('resume_text', '')

        if resume_text.strip():
            # Preprocess the resume text
            cleaned_text = preprocess_text(resume_text)

            # Vectorize the cleaned text
            resume_vec = vectorize_with_word2vec(cleaned_text, word2vec_model)

            # Predict the category
            prediction = model.predict(np.array([resume_vec]))
            predicted_category = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])

            # Render the result
            return render_template('result.html', category=predicted_category[0])

        else:
            return render_template('index.html', error="Please enter a valid resume text.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, request, render_template, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# import pandas as pd
# import numpy as np
# import nltk
# import tensorflow as tf
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import gensim.downloader as api
# from sklearn.preprocessing import LabelEncoder

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# app = Flask(__name__)

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# word2vec_model = api.load("word2vec-google-news-300")
# model = tf.keras.models.load_model('resume_classification_model.keras')

# label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_text(text):
#     text = text.lower()
#     words = word_tokenize(text)
#     words = [word for word in words if word.isalpha()]
#     words = [word for word in words if word not in stop_words]
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(words)

# def vectorize_with_word2vec(text, model):
#     words = text.split()
#     vectors = [model[word] for word in words if word in model]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             print('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 resume_text = f.read()

            
#             cleaned_text = preprocess_text(resume_text)
#             resume_vec = vectorize_with_word2vec(cleaned_text, word2vec_model)

#             prediction = model.predict(np.array([resume_vec]))
#             predicted_category = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])
            
#             print(f"Cleaned Text: {cleaned_text}\nResume Vector: {resume_vec}\nPredicted category: {predicted_category}")

#             # random_category = np.random.choice(label_encoder.classes_)
#             # return render_template('result.html', category=random_category)
#             return render_template('result.html', category=predicted_category[0])

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


















# # from flask import Flask, request, render_template, redirect, url_for
# # from werkzeug.utils import secure_filename
# # import os
# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # from sklearn.preprocessing import LabelEncoder
# # import gensim.downloader as api


# # app = Flask(__name__)

# # # Load the pre-trained Word2Vec model and other necessary files
# # word2vec_model = api.load("word2vec-google-news-300")
# # model = tf.keras.models.load_model('resume_classification_model.keras')

# # # Load the label encoder (if applicable)
# # label_encoder = LabelEncoder()
# # label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# # # Dataset for preprocessed resume data
# # dataset = pd.read_csv('dataset.csv')  # Replace with actual path to your dataset
# # dataset.set_index('ID', inplace=True)  # Assuming 'ID' column is the unique identifier for resumes

# # UPLOAD_FOLDER = 'uploads'
# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     if request.method == 'POST':
# #         if 'file' not in request.files:
# #             print('No file part')
# #             return redirect(request.url)
# #         file = request.files['file']
# #         if file and allowed_file(file.filename):
# #             filename = secure_filename(file.filename)
# #             file_id = filename.split('.')[0]  # Assuming the file name (without extension) matches the ID in dataset

# #             # Check if the file ID exists in the dataset
# #             if file_id not in dataset.index:
# #                 return "Error: No matching record found for this resume ID in the dataset."

# #             # Get the preprocessed data (assuming it's already preprocessed in the dataset)
# #             preprocessed_text = dataset.loc[file_id, 'Resume_str']  # Assuming 'Resume_str' contains the preprocessed text

# #             # Vectorize the preprocessed text (no need to preprocess again)
# #             resume_vec = vectorize_with_word2vec(preprocessed_text, word2vec_model)

# #             # Predict the category using the model
# #             prediction = model.predict(np.array([resume_vec]))
# #             predicted_category = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])

# #             print(f"Predicted category: {predicted_category}")

# #             return render_template('result.html', category=predicted_category[0])

# #     return render_template('index.html')

# # def vectorize_with_word2vec(text, model):
# #     words = text.split()
# #     vectors = [model[word] for word in words if word in model]
# #     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# # if __name__ == '__main__':
# #     app.run(debug=True)














# import os
# import numpy as np
# from flask import Flask, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import gensim.downloader as api
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Initialize Flask app
# app = Flask(__name__)

# # Set upload folder and allowed file extensions
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# # Load the trained model and label encoder
# model = load_model('resume_classification_model.keras')
# label_classes = np.load('label_encoder_classes.npy', allow_pickle=True)
# word2vec_model = api.load("word2vec-google-news-300")

# # Preprocessing functions
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_text(text):
#     text = text.lower()
#     words = word_tokenize(text)
#     words = [word for word in words if word.isalpha()]
#     words = [word for word in words if word not in stop_words]
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(words)

# def vectorize_with_word2vec(text, model):
#     words = text.split()
#     vectors = [model[word] for word in words if word in model]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Read and preprocess the file content
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 content = f.read()
#         except Exception as e:
#             return jsonify({'error': f'Failed to read file content: {str(e)}'}), 500

#         cleaned_text = preprocess_text(content)
#         vectorized_text = vectorize_with_word2vec(cleaned_text, word2vec_model).reshape(1, -1)

#         # Predict category
#         prediction = model.predict(vectorized_text).argmax(axis=1)[0]
#         predicted_category = label_classes[prediction]

#         # Create category folder and move file
#         category_folder = os.path.join(app.config['UPLOAD_FOLDER'], predicted_category)
#         os.makedirs(category_folder, exist_ok=True)
#         new_path = os.path.join(category_folder, filename)
#         os.rename(filepath, new_path)

#         return jsonify({'message': 'File successfully processed', 'category': predicted_category}), 200

#     return jsonify({'error': 'Invalid file type'}), 400

# @app.route('/download/<category>/<filename>', methods=['GET'])
# def download_file(category, filename):
#     category_folder = os.path.join(app.config['UPLOAD_FOLDER'], category)
#     if not os.path.exists(category_folder):
#         return jsonify({'error': 'Category not found'}), 404

#     filepath = os.path.join(category_folder, filename)
#     if not os.path.exists(filepath):
#         return jsonify({'error': 'File not found'}), 404

#     return send_from_directory(category_folder, filename)

# if __name__ == '__main__':
#     app.run(debug=True)




# import csv
# import os
# from flask import Flask, request, redirect, render_template
# from werkzeug.utils import secure_filename
# import numpy as np

# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = './uploads'  # Set this to your desired upload directory

# def preprocess_text(text):
#     text = text.lower()
#     words = word_tokenize(text)
#     words = [word for word in words if word.isalpha()]
#     words = [word for word in words if word not in stop_words]
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(words)

# def vectorize_with_word2vec(text, model):
#     words = text.split()
#     vectors = [model[word] for word in words if word in model]
#     return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# # Assume `model` and `label_encoder` are preloaded models.
# model = None  # Replace with your model loading logic
# label_encoder = None  # Replace with your label encoder loading logic
# word2vec_model = None  # Replace with your Word2Vec model loading logic

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             print('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             # Extract ID from the file name (assuming no extension)
#             file_id = os.path.splitext(filename)[0]
#             # print(f"File ID: {file_id} and Path: {file_path} and name: {filename}")

#             # Read CSV and match ID
#             csv_file_path = './dataset.csv'  # Set to the path of your CSV file
#             record = None
#             try:
#                 with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
#                     reader = csv.DictReader(csvfile)
#                     for row in reader:
#                         # print(row)
#                         if row['ID'] == file_id:  # Replace 'id' with the column name in your CSV
#                             print("here...\n\n\n\n\n\n\n")
#                             record = row
#                             break
#             except Exception as e:
#                 print(f"Error reading CSV: {e}")
#                 return "Error reading CSV file."

#             if record:
#                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                     resume_text = f.read()

#                 cleaned_text = preprocess_text(resume_text)
#                 resume_vec = vectorize_with_word2vec(cleaned_text, word2vec_model)

#                 prediction = model.predict(np.array([resume_vec]))
#                 predicted_category = label_encoder.inverse_transform([prediction.argmax(axis=1)[0]])

#                 print(f"Matched Record: {record}\nCleaned Text: {cleaned_text}\nResume Vector: {resume_vec}\nPredicted category: {predicted_category}\nACTUAL: {record['category']}")

#                 return render_template('result.html', category=predicted_category[0], record=record)
#             else:
#                 print(f"No matching record found for ID: {file_id}")
#                 return "No matching record found in CSV."

#     return render_template('index.html')

# def allowed_file(filename):
#     # Add logic for checking allowed file extensions
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}

# if __name__ == '__main__':
#     app.run(debug=True)
