import os
import requests
import json
import tempfile
from moviepy import *
# from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip
from langchain.llms import OpenAI, HuggingFacePipeline
import openai
# For the local model approach
import torch
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import traceback
# Import dotenv to load environment variables
from dotenv import load_dotenv
from pydub import AudioSegment
import regex as re

###########################
# Load Environment Variables
###########################
load_dotenv()  # Loads variables from .env into os.environ
SERP_API_KEY = os.environ.get("SERP_API_KEY")
PIXELS_API_KEY = os.environ.get("PIXELS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

###########################
# 1. Trend Detection
###########################
class TrendDetectorSERP:
    """Uses Google SERP API to fetch trending topics."""
    def __init__(self, serp_api_key):
        self.serp_api_key = serp_api_key

    def get_trending_topics(self, query="latest trending news"):
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.serp_api_key,
            "q": query,
            "hl": "en",
            "tbm": "nws"  # news results
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            topics = [result['title'] for result in data.get("news_results", []) if 'title' in result]
            return topics
        else:
            print("Error fetching trending topics from SERP API:", response.status_code)
            return []

###########################
# 2. Script Generation using LangChain
###########################

# Approach 1: Using OpenAI API with LangChain
from langchain.chat_models import ChatOpenAI

class ScriptGeneratorLangchainOpenAI:
    """Generates video scripts using the OpenAI API via LangChain."""
    def __init__(self, openai_api_key):
        # Initialize the LangChain OpenAI LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key, 
            model="gpt-4o-mini"
        )
    
    def generate_script(self, trending_topic):
        prompt = f"Generate an engaging and factually accurate video script about: {trending_topic}"
        response = self.llm.invoke(prompt)  # Use invoke() instead of calling the object directly
        return response.content  # Extract the response text


# Approach 2: Using a local lightweight LLM with LangChain
class ScriptGeneratorLangchainLocal:
    """Generates video scripts using a local LLM via LangChain (e.g. GPT-2)."""
    def __init__(self, model_name="qwne2.5-7b:instruct"):
        # Load the tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Create a HuggingFace text-generation pipeline
        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, max_length=300)
        # Wrap the pipeline using LangChain's HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def generate_script(self, trending_topic):
        prompt = f"Generate an engaging and factually accurate video script about: {trending_topic}"
        script = self.llm(prompt)
        return script

###########################
# 3. Media Sourcing
###########################
class MediaSourcer:
    def __init__(self, pexels_api_key):
        self.pexels_api_key = pexels_api_key

    def get_images(self, query, count=10):
        url = "https://api.pexels.com/v1/search"
        headers = {
            "Authorization": self.pexels_api_key
        }
        params = {
            "query": query,
            "per_page": count
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            images = [img["src"]["medium"] for img in data.get("photos", [])]
            return images
        else:
            print("Error fetching images from Pexels:", response.status_code)
            return []

    def get_videos(self, query, count=3):
        url = "https://api.pexels.com/videos/search"
        headers = {
            "Authorization": self.pexels_api_key
        }
        params = {
            "query": query,
            "per_page": count
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            videos = [video["video_files"][0]["link"] for video in data.get("videos", [])]
            return videos
        else:
            print("Error fetching videos from Pexels:", response.status_code)
            return []

###########################
# 4. Video Assembly
###########################
class VideoAssembler:
    def __init__(self, output_dir="./videos", download_dir="./downloads", narration_dir="./narrations"):
        """Initialize the video assembler with structured directories."""
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.narration_dir = narration_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories if they do not exist
        for directory in [self.output_dir, self.download_dir, self.narration_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Initialize Coqui TTS Model
        self.tts_model = TTS("tts_models/en/ljspeech/tacotron2-DCA").to(self.device)  # Change model if needed

    def _download_file(self, url, filename):
        """Downloads a file and stores it in the downloads directory."""
        response = requests.get(url)
        file_path = os.path.join(self.download_dir, filename)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path

    def _split_text_into_chunks(self, text, max_words=40):
        """Splits text into smaller chunks based on word count to avoid TTS buffer issues."""
        sentences = re.split(r'(?<=[.!?]) +', text)  # Split at sentence boundaries
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            words = sentence.split()
            if len(current_chunk.split()) + len(words) > max_words:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    def _concatenate_audio_clips(self, audio_files, output_path):
        """Merges multiple audio files into a single file."""
        combined = AudioSegment.empty()
        
        for file in audio_files:
            audio = AudioSegment.from_wav(file)
            combined += audio  # Append each clip

        combined.export(output_path, format="wav")
        return output_path


    def _generate_narration(self, script, filename="narration.wav"):
        """Generates narration using Coqui TTS and saves it in the narrations directory."""
        narration_file = os.path.join(self.narration_dir, filename)
        text_chunks = self._split_text_into_chunks(script, max_words=40)  # Reduce length per call
        temp_files = []

        try:
            # Run TTS
            # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
            # Text to speech list of amplitude values as output
            for i, chunk in enumerate(text_chunks):
                chunk_cleaned = re.sub(r'[^a-zA-Z0-9\s,.\-!?;:(){}[\]"\']', '', chunk)
                chunk_cleaned = chunk_cleaned.replace("\n", " ").replace("\r", " ")
                temp_file = os.path.join(self.narration_dir, f"temp_chunk_{i}.wav")
                self.tts_model.tts_to_file(text=chunk_cleaned, file_path=temp_file)
                temp_files.append(temp_file)
            # Combine all chunks into one final audio file
            combined_audio = self._concatenate_audio_clips(temp_files, narration_file)
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()  # Prints the full stack trace
        return narration_file


    def assemble_video(self, script, image_urls, video_clip_urls, bg_music_path=None):
        """Assembles a video from images, text clips, and video clips with narration and background music."""
        img_clips = []
        video_clips = []

        # Download images and store file paths
        for idx, url in enumerate(image_urls):
            try:
                print(f"Downloading image {idx+1}...")
                image_file = self._download_file(url, f"image_{idx}.jpg")
                # Create an image clip using MoviePy's ImageClip (fallback to VideoFileClip if needed)
                img_clip = ImageClip(image_file, duration=3)
                img_clips.append(img_clip)
            except Exception as e:
                print("Error processing image:", e)
        
        # Download videos and store file paths
        for idx, url in enumerate(video_clip_urls):
            try:
                print(f"Downloading video {idx+1}...")
                video_file = self._download_file(url, f"video_{idx}.mp4")
                video_clip = VideoFileClip(video_file)
                video_clips.append(video_clip)
            except Exception as e:
                print("Error processing video:", e)


        # Create text overlay clip
        try:
            txt_clip = TextClip(
                font="DejaVuSans-Bold",
                text=script[:150] + "...",
                font_size=24,
                color='white',
                size=(640, 480),
                method='caption',
                duration=3
            )
        except Exception as e:
            print("Error creating text clip:", e)
            txt_clip = None
    
        # Generate narration using Coqui TTS
        try:
            narration_file = self._generate_narration(script)
            narration_audio = AudioFileClip(narration_file)
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()
            return None
    
        # Load background music if provided
        bg_music = None
        if bg_music_path and os.path.exists(bg_music_path):
            bg_music = AudioFileClip(bg_music_path).volumex(0.3)  # Lower background music volume
    
        # Combine audio tracks
        audio_tracks = [narration_audio]
        if bg_music:
            audio_tracks.append(bg_music)
    
        final_audio = CompositeAudioClip(audio_tracks)
    
        # Combine all clips (images + videos)
        clips = img_clips + video_clips
        if txt_clip:
            clips.append(txt_clip)
    
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.audio = final_audio  # Add the audio to the video
    
        # Save the video in the `videos` directory
        output_file = os.path.join(self.output_dir, "final_video.mp4")
        print(f"Rendering video with narration to {output_file}... This may take a few moments.")
        final_clip.write_videofile(output_file, fps=24, audio_codec="aac")
    
        return output_file  # Return saved video path


###########################
# 5. Content Moderation
###########################
class ContentModerator:
    def moderate_video(self, video_file):
        # For demonstration, this method always returns True.
        print("Moderating video:", video_file)
        return True

###########################
# 6. Social Media Publisher
###########################
class SocialMediaPublisher:
    def publish_video(self, video_file, caption):
        # Stub: In production, integrate with social media APIs.
        print(f"Publishing video: {video_file}")
        print(f"With caption: {caption}")
        return True

###########################
# Main Pipeline Orchestration
###########################
def main():
    # Replace these API keys with your actual credentials.
    # 1. Trend Detection using Google SERP API
    trend_detector = TrendDetectorSERP(SERP_API_KEY)
    trending_topics = trend_detector.get_trending_topics(query="latest trending news")
    # trending_topics = None
    if trending_topics:
        trending_topic = trending_topics[0]
        print("Trending topic:", trending_topic)
    else:
        trending_topic = "General News"
        print("No trending topics found; using default topic:", trending_topic)

    # 2. Script Generation
    # Choose which approach to use for script generation.
    use_openai = True  # Set to False to use the local LLM approach.
    if use_openai:
        script_generator = ScriptGeneratorLangchainOpenAI(OPENAI_API_KEY)
    else:
        script_generator = ScriptGeneratorLangchainLocal(model_name="qwne2.5-7b:instruct")
    
    script = script_generator.generate_script(trending_topic)
    # script = "This is a sample script for the video."
    print("Generated script:\n", script)

    # 3. Media Sourcing
    media_sourcer = MediaSourcer(PIXELS_API_KEY)
    image_urls = media_sourcer.get_images(trending_topic)
    video_clip_urls = media_sourcer.get_videos(trending_topic)
    print("Sourced", len(image_urls), "images and", len(video_clip_urls), "video clips.")

    # 4. Video Assembly
    video_assembler = VideoAssembler()
    video_file = video_assembler.assemble_video(script, image_urls, video_clip_urls)
    if video_file:
        print("Video assembled successfully:", video_file)
    else:
        print("Video assembly failed.")
        return

    # 5. Content Moderation
    moderator = ContentModerator()
    if not moderator.moderate_video(video_file):
        print("Video failed moderation. Aborting publish.")
        return
    print("Video passed moderation.")

    # 6. Publish Video on Social Media
    publisher = SocialMediaPublisher()
    caption = script[:100]  # Use the first 100 characters of the script as a caption.
    publisher.publish_video(video_file, caption)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
