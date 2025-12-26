# Scout
**Artifical Intellegence powered Visual search for local image library**

Scout uses Advanced AI (CLIP embeddings) to let you seacah through your images using natural language (keywords/tags). No manual tags or organization. Just index a folder of images into the application and search a keyword to see the images in the folder appear from most relevant to the search.

Originally, this was build for local battlemap collections for D&D games. This was something I needed with other 10K of battlemap images and no possible way to easily and quickly find exactly what I want in it. Scout works vrilliantly with any image collection: photos, game assets, design resources, and more.

---

# What Scout does
Scout doesn't just match keywords to images automatically - it understand what is in all images. Search for concepts, colours, moods, scenes, objects and more.

- Searches 100,000+ images in under 2 seconds
- Real time results as you type
- Uses thumbnail generation and preview for better and faster performance.

## Learning System
Mark results as correct/incorrect to train Scout to understand when an image result on a search isn't applicable. For example - you search the keyword "Dog" and in your results you get one cat image out of thousands of dog images. Marking cat as incorrect will remove it from the search results and ensure any future search results that "cat" isn't applicable to won't show up.

This is a form of correction when it goes things wrong.

# Features
## Search
- **Filename Matching**: Combines AI with traditional keyword search
- **Dynamic search**: see results upddate as you type
- **Learning system**: correct/incorrect train search accuracy.
- **FAISS Acceleration**: Automatic performance acceleration for 10,000 images indexed or more.

## Image Management
- **Drag and drop Indexing**: Drop folders or files to index instantly
- **Multi-folder support**: index any number of folders from anywhere on your system or connected storage.
- **Folder Manager**: View, refresh or remove indexed folders and associated images.
- **Smart Deduplication**: Automatically skip already indexed images
- **Deletion protection**: Remove images from the index without deleteing them from your device.

## User interface
- **Thumbnail Caching**: Fast browing with WEBP/JPEG thumbnails
- **Hover preview**: see larger previews of search results without opening files
- **Multi-select** - select multiple images with `Ctrl+click` allowing for bulk actions.
- **Theme**: Uses system theme for dark/light mode.
- **Preview image**: left click on image in search results to open in default operating system's image viewer.

## Performance
- **Lazy Loading**: Only loads visible thumbnails (plus a small buffer).
- **Batch processing**: Effecient multi-threading indexing.
- **Memory Management**: Optimized for large collections (tested with 135,000 images but more is possible depending on system resources).

# How it works
Scout uses CLIP (Contrastive Language-Image Pre-training), an AI model from OpenAI that understands the relationship between images and text. CLIP is automatically downloaded and ran on your device so no cloud or data over internet transfers, complete security on your device.

## The process:
1. **Indexing**: Scout processes each image through CLIP, creating a embedding.
2. **Storage**: Embeddings are stored locally along with image paths.
3. **Search**: Your text query is converted to an embedding.
4. **Matching**: Scout finds images with similar embeddings using Cosine Similarity.
5. **Ranking**: Results are sorted by relevance, combining AI similarity with filename matches

## FAISS Acceleration
For collections over 10,000 images, Scout automatically uses FAISS (Facebook AI Similarity Search) to dramatically speed up searches:
- 10K-50K images: IVFFlat index (~5-10x faster)
- 50K+ images: IVFPQ index (~10-50x faster)

# System Requirements
## Minimum Requirements (for small collections <5K images)
- **OS**: Windows 11 (64-bit)
- **RAM**: 4GB
- **CPU**: Intel i3 / AMD Ryzen 3 (duel-core)
- **Storage**: 2GB free + image storage
- **Python**: 3.11 or greater (works with 3.13)

## Recommended (10K-50K images)
- **OS**: Windows 11 (64-bit)
- **RAM**: 8GB
- **CPU**: Intel i5 / AMD Ryzen 5
- **Storage**: SSD with 5GB+ free + image storage
- **Python**: 3.11 or greater (works with 3.13)

## Heavy use (50K+ images)
- **OS**: Windows 11 (64-bit)
- **RAM**: 16GB+
- **CPU**: Intel i7 / AMD Ryzen 7
- **Storage**: SSD with 10GB+ free + image storage
- **Python**: 3.11 or greater (works with 3.13)
- **GPU**: NVIDIA GPU with CUDA for faster indexing

# Installation
## Prerequisites
- **Python** 3.11 or 3.13 (tested versions)
- **Windows** 11 (64-bit)

### Quick Install

1. **Clone the repository**
```bash
git clone https://github.com/Jamster3000/scout.git
cd scout
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Scout**
```bash
# Using the included batch file (recommended)
run_scout.bat

# Or directly with Python
python Scout.py
```

## Quick Start

### 1. Launch Scout
Run `run_scout.bat` or `python Scout.py`

### 2. Index Your Images
- Click **"Index Images"** → Select folder
- Or **drag & drop** folders directly into Scout
- Progress shown with progress bar and notification.

### 3. Search!
Type what you're looking for:
- "forest clearing with sunlight"
- "red sports car front view"
- "cute cat sleeping"
- Or using singular words as well

### 4. Refine Results
- **Correct**: Mark as correct (improves future searches)
- **Incorrect**: Mark as incorrect (hides from results)
- **Right-click**: Context menu (open, delete, etc.)

# Building from source

I've tried building and compiling Scout to a standalone .exe using PyInstaller but due to **PyTorch** and it's dependencies. PyTorch's complex achitecture and libraries make it incompatible with PyInstaler's bundling process.

# Tested Python Versions
Other versions besides below may work but haven't directly been tested.

- **Python 3.11**
- **Python 3.13**

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
