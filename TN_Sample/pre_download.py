# pre_download_model.py
from sentence_transformers import SentenceTransformer

def download_model():
    """
    This function downloads and caches the specified SentenceTransformer model.
    Run this script once to avoid network issues in your main application.
    """
    model_name = 'all-MiniLM-L6-v2'
    print(f"Downloading and caching model: '{model_name}'...")
    print("This may take a few minutes depending on your internet connection.")

    try:
        # This line triggers the download and caches the model locally.
        SentenceTransformer(model_name)
        print("\n✅ Model downloaded and cached successfully!")
        print("You can now run your main verification script.")
    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please check your internet connection and firewall settings, then try again.")

if __name__ == "__main__":
    download_model()