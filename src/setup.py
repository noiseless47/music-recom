import subprocess
import os
from utils.verify_downloads import verify_downloads
from utils.logger import Logger
from utils.decorators import handle_errors

logger = Logger()

@handle_errors
def setup_project():
    # Install requirements
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    
    # Download and prepare emotion model
    print("\nPreparing emotion detection model...")
    subprocess.run(['python', 'src/train_emotion_model.py'])
    
    # Download and process music library
    print("\nPreparing music library...")
    subprocess.run(['python', 'src/prepare_music_library.py'])
    
    # Download and prepare genre classification model
    print("\nPreparing genre classification model...")
    subprocess.run(['python', 'src/genre_classifier.py'])
    
    # Download pre-trained models
    print("\nDownloading pre-trained models...")
    subprocess.run(['python', 'src/download_models.py'])
    
    # Verify downloads
    if verify_downloads():
        print("\nSetup complete! You can now run the app using:")
        print("streamlit run src/app.py")
    else:
        print("\nSetup incomplete. Please check the errors above.")

if __name__ == "__main__":
    setup_project() 