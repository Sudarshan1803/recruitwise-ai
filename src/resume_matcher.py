import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

def preprocess_text(text):
    # Implement text preprocessing here (similar to what you did before)
    return text

def generate_embeddings(input_directory):
    embeddings = []
    filenames = []

    # Load the language model
    nlp = spacy.load("en_core_web_sm")

    # Iterate over resumes in the input directory
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            preprocessed_text = preprocess_text(text)
            doc = nlp(preprocessed_text)
            embeddings.append(doc.vector)
            filenames.append((filename.split('.')[0], filename))  # Store candidate name along with filename
    
    return np.array(embeddings), filenames

def calculate_similarity(job_embedding, resume_embeddings):
    return cosine_similarity(job_embedding.reshape(1, -1), resume_embeddings)[0]

def match_job(job_description, resume_embeddings, filenames):
    # Preprocess job description
    preprocessed_job = preprocess_text(job_description)
    # Generate job embedding
    nlp = spacy.load("en_core_web_sm")
    job_doc = nlp(preprocessed_job)
    job_embedding = job_doc.vector

    # Calculate cosine similarity
    similarity_scores = calculate_similarity(job_embedding, resume_embeddings)

    # Rank resumes based on similarity scores
    ranked_resumes = sorted(zip(filenames, similarity_scores), key=lambda x: x[1], reverse=True)

    return ranked_resumes

def main():
    # Define input directories
    job_description_file = "job_description_embeddings.txt"  # Update with your job description file
    resumes_directory = "preprocessed_resumes"  # Update with your preprocessed resumes directory

    # Generate embeddings for resumes
    resume_embeddings, filenames = generate_embeddings(resumes_directory)

    # Load job description
    with open(job_description_file, 'r', encoding='utf-8') as file:
        job_desc_text = file.read()

    # Match job with resumes
    matched_resumes = match_job(job_desc_text, resume_embeddings, filenames)

    # Print the top matched resumes
    print("Top Matched Resumes for the Following Job Description:\n")
    
    # Print job description
    print("Job Description:\n----------------")
    print(job_desc_text + "\n")
    
    # Print matched resumes
    print("Matched Resumes:\n----------------")
    print("| Candidate Name | Resume File  | Similarity Score (%) |")
    print("|----------------|--------------|-----------------------|")
    for (candidate_name, resume_file), score in matched_resumes[:5]:  # Adjust the number of top matches to display
        similarity_percentage = score * 100
        print(f"| {candidate_name:<15} | {resume_file:<12} | {similarity_percentage:.2f}                   |")

if __name__ == "__main__":
    main()
