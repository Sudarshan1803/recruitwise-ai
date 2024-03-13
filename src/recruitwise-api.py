from flask import Flask, request, jsonify
import os
import base64
from azure.storage.blob import BlobServiceClient
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import resume_processor

# Initialize Flask app
app = Flask(__name__)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to generate embeddings for resumes
def generate_embeddings(resumes_directory):
    embeddings = []
    filenames = []

    # Iterate over resumes in the input directory
    for filename in os.listdir(resumes_directory):
        input_file_path = os.path.join(resumes_directory, filename)
        resume_processor.preprocess_file(input_file_path)
        preprocessed_text = resume_processor.preprocess_file(input_file_path)
        doc = nlp(preprocessed_text)
        embeddings.append(doc.vector)
        filenames.append(filename)
    
    return np.array(embeddings), filenames

# Function to fetch resumes from Azure Blob storage
def fetch_resumes_from_blob(account_url, container_name):
    input_directory = 'resumes'
    # Connect to Azure Blob storage
    # blob_service_client = BlobServiceClient(account_url=account_url)
    # container_client = blob_service_client.get_container_client(container_name)

    # resumes = []

    # List blobs in the container
    # for blob in container_client.list_blobs():
        # Download blob content and append to the resumes list
        # blob_content = blob_client.download_blob(blob)
        # resumes.append(blob_content.readall())
    
    # Hard Code Temp Dir for testing 
    # return resumes
    return input_directory

# Function to calculate cosine similarity between job description and resumes
def calculate_similarity(job_embedding, resume_embeddings):
    return cosine_similarity(job_embedding.reshape(1, -1), resume_embeddings)[0]

# Function to match job description with resumes
def match_job(job_description, resume_embeddings, filenames):
    # Preprocess job description
    preprocessed_job = resume_processor.preprocess_text(job_description)
    # Generate job embedding
    job_doc = nlp(preprocessed_job)
    job_embedding = job_doc.vector

    # Calculate cosine similarity
    similarity_scores = calculate_similarity(job_embedding, resume_embeddings)

    # Rank resumes based on similarity scores
    ranked_resumes = sorted(zip(filenames, similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes

# API endpoint to match job description with resumes
@app.route('/match_resumes', methods=['POST'])
def match_resumes():
    # Parse request parameters
    if not request.json or 'job_description' not in request.json or 'resume_blob_location' not in request.json:
        return jsonify({'error': 'Missing required parameters'}), 400

    job_description = request.json['job_description']
    resume_blob_location = request.json['resume_blob_location']

    # Default value for top_n
    top_n = request.json.get('top_n', 5)

    try:
        # Fetch resumes from Azure Blob storage
        resumes = fetch_resumes_from_blob(resume_blob_location, '')

        # Generate embeddings for resumes
        resume_embeddings, filenames = generate_embeddings(resumes)

        # Match job with resumes
        matched_resumes = match_job(job_description, resume_embeddings, filenames)

        # Limit the number of matched resumes based on top_n
        matched_resumes = matched_resumes[:top_n]

        # Prepare JSON response
        response_data = [{'filename': filename, 'score': f"{round(float(score) * 100, 2)}%"} for filename, score in matched_resumes]

        return jsonify({'matched_resumes': response_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main function
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
