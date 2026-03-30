import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MediaRecommender:
    def __init__(self, data_filepath='media_database.csv'):
        self.data_filepath = data_filepath
        self.data = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def load_data(self):
        """Loads the dataset and handles missing values or formatting issues."""
        if not os.path.exists(self.data_filepath):
            print(f"❌ Error: Could not find '{self.data_filepath}'.")
            return False
            
        try:
            # Read CSV and immediately fix missing values
            self.data = pd.read_csv(self.data_filepath)
            
            # Fix: Ensure 'description' column exists and replace NaN with empty strings
            if 'description' in self.data.columns:
                self.data['description'] = self.data['description'].fillna('').astype(str)
            else:
                print("❌ Error: CSV is missing a 'description' column.")
                return False

            print(f"✅ Loaded {len(self.data)} items from the database.")
            return True
        except Exception as e:
            print(f"❌ Error reading the CSV file: {e}")
            return False

    def build_engine(self):
        """Prepares the TF-IDF matrix for semantic matching."""
        if self.data is None or self.data.empty:
            return False
            
        try:
            # Build the TF-IDF representation of the text
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data['description'])
            print("✅ Intelligence Engine Initialized Successfully.")
            return True
        except Exception as e:
            print(f"❌ Initialization Failed: {e}")
            return False

    def get_recommendation(self, user_query):
        """Finds the most similar entry in the dataset to the user's input."""
        # Convert user input to a vector
        query_vec = self.vectorizer.transform([user_query])
        
        # Calculate Cosine Similarity
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get the index with the highest score
        best_match_idx = similarity_scores.argsort()[-1]
        score = similarity_scores[best_match_idx]
        
        # Fetch the title, type, and description of the best match
        title = self.data.iloc[best_match_idx]['title']
        m_type = self.data.iloc[best_match_idx]['type']
        
        return title, m_type, score

def main():
    # Make sure to point to your CSV file
    engine = MediaRecommender('media_database.csv')
    
    if engine.load_data() and engine.build_engine():
        print("\n" + "="*45)
        print("    SEMANTIC MEDIA MATCHING ENGINE    ")
        print("="*45)
        
        while True:
            print("\n💡 Tip: Describe a feeling, genre, or plot (Type 'exit' to quit)")
            user_input = input("✨ Your Request: ").strip()
            
            if user_input.lower() == 'exit':
                print("Exiting engine. Goodbye!")
                break
                
            if len(user_input) < 4:
                print("⚠️ Please provide a more detailed description.")
                continue
                
            # Perform the calculation
            title, media_type, confidence = engine.get_recommendation(user_input)
            
            print(f"\n🎯 Recommended {media_type}: {title}")
            print(f"📈 Match Confidence: {confidence:.2%}") # Formatted as a percentage
            
            if confidence < 0.15:
                print("🔍 Hint: Try mentioning themes like 'magic', 'future', or 'politics'.")
    else:
        print("\n⚠️ Program failed to initialize. Check your CSV and console for errors.")

if __name__ == "__main__":
    main()