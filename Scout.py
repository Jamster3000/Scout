import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import pickle
import numpy as np
import orjson
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import asyncio
import aiofiles
from pathlib import Path
import threading
from typing import Any, List, Dict, Optional
import warnings
import time
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
import queue
import faiss
from plyer import notification
import pyautogui
import ctypes
import shutil
from collections.abc import Sequence

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*MessageFactory.*GetPrototype.*')
Image.MAX_IMAGE_PIXELS = None

#customtkinter settings
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("green")

def save_json(filepath: str, data: Dict) -> None:
    """Save JSON data. ORjson for faster speeds and no indent to improve read and write times"""
    with open(filepath, 'wb') as f:
        f.write(orjson.dumps(data))

def load_json(filepath: str) -> Dict:
    """Loads JSON data using orjaon for faster speeds"""
    with open(filepath, 'rb') as f:
        return orjson.loads(f.read())

async def process_thumbnail(path: str, thumb_folder: str, cache_size: int, thumb_format:str="webp"):
    """Generate a thumbnail for a single image using async"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _generate_thumbnail, path, thumb_folder, cache_size, thumb_format)
        return result
    except Exception as e:
        return None

def _generate_thumbnail(path: str, thumb_folder: str, cache_size: int, thumb_format:str="webp") -> bool:
        """Generates thumbnails from a given image called in an async wrapper"""
        try:
            original_img = Image.open(path)

            #Handle the possibility of an image having transparency
            if original_img.mode == 'RGBA':
                background = Image.new("RGB", original_img.size, '#3a3a3a')
                background.paste(original_img, mask=original_img.split()[3])
                original_img = background
            elif original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')

            #Create the actual thumbnail
            thumb = original_img.copy()
            thumb.thumbnail((cache_size, cache_size), Image.Resampling.LANCZOS)

            ext = '.webp' if thumb_format == 'webp' else '.jpg'
            thumb_name = str(abs(hash(path)) % 10**10) + ext
            thumb_path = os.path.join(thumb_folder, thumb_name)

            if thumb_format == 'webp':
                thumb.save(thumb_path, 'WEBP', quality=85, method=4)
            else:
                thumb.save(thumb_path, 'JPEG', quality=85, optimize=True)

            return True
        except Exception as e:
            return None

async def process_thumbnails_batch(paths: str, thumb_folder: str, cache_size: int, batch_size:int=10, thumb_format:str='webp'):
    """Process multiple thumbnails with async batching"""
    results = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        tasks = [process_thumbnail(p, thumb_folder, cache_size, thumb_format) for p in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    return results

def lazy_import_sentence_transformers():
    """Lazy import for SentenceTransformer to speed up startup allowing the GUI to appear almost instantly"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

class SearchCache:
    def __init__(self, max_size:int=100) -> None:
        self.cache = {}
        self.max_size = max_size
        self.index_version = None

    def get(self, query: str) -> List[Dict]:
        """"""
        return self.cache.get(query.lower().strip())

    def set(self, query: str, results: List[Dict]) -> None:
        key = query.lower().strip()
        self.cache[key] = results
        if len(self.cache) > self.max_size:
            self.cache.pop(next(iter(self.cache)))

    def clear(self) -> None:
        self.cache.clear()

    def invalidate_if_needed(self, current_version: str) -> None:
        if self.index_version != current_version:
            self.clear()
            self.index_version = current_version

class ImageLoader:
    def __init__(self, thumbnail_size:int=300, cache_folder:str="images/thumbnails") -> None:
        self.thumbanil_size = thumbnail_size
        self.cache = {}
        self.cache_folder = cache_folder
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.active = True
        self.callback_cache = queue.Queue()
        self.load_lock = threading.Lock()

    def load_thumbnail(self, path:str, callback) -> None:
        if path in self.cache:
            callback(self.cache[path])
            return

        self.executor.submit(self._load_worker, path, callback)

    def _load_worker(self, path: str, callback) -> None:
        """Worker thread to load thumbnail"""
        try:
            if not self.active:
                return

            thumb_hash = str(abs(hash(path)) % 10**10)
            thumb_path_webp = os.path.join(self.cache_folder, thumb_hash + '.webp')
            thumb_path_jpg = os.path.join(self.cache_folder, thumb_hash + '.jpg')

            with self.load_lock:
                if os.path.exists(thumb_path_webp): #load webp thumbnail
                    img = Image.open(thumb_path_webp)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.load()
                elif os.path.exists(thumb_path_jpg): #load jpeg thumbnail
                    img = Image.open(thumb_path_jpg)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.load()
                else: #thumbnail doesn't exist, regenerate it
                    img = Image.open(path)

                    if hasattr(img, 'draft') and max(img.size) > 2048:
                        try:
                            img.draft('RGB', (self.thumbanil_size * 2, self.thumbanil_size * 2))
                        except:
                            pass

                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')

                    img.thumbnail((self.thumbanil_size, self.thumbanil_size), Image.Resampling.NEAREST)
                    img.load()
            
            self.cache[path] = img
            self.callback_cache.put((callback, img))
        except Exception as e:
            pass

    def process_callbacks(self, root:ctk.CTk) -> None:
        """Process queued callbacks on the main thread"""
        try:
            while not self.callback_cache.empty():
                try:
                    callback, img = self.callback_cache.get_nowait()
                    callback(img)
                except queue.Empty:
                    break
                except:
                    pass
        except:
            pass

        if self.active:
            root.after(16, lambda: self.process_callbacks(root))

    def clear(self) -> None:
        """Clear the image cache and stop loading"""
        self.cache.clear()

    def shutdown(self) -> None:
        self.active = False
        self.executor.shutdown(wait=False)

class ScoutExplorer:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Scout")
        self.root.iconbitmap("Scout.ico")

        appID = 'Scout'  # Shows in notifications and taskbar
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appID)

        self.root.update()
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        pyautogui.hotkey('winleft', 'up')

        #state variables
        self.model = None
        self.embeddings = None
        self.faiss_index = None
        self.image_paths = []
        self.image_paths_set = set()
        self.metadata = {}
        self.current_results = []
        self.settings = self.load_settings()
        self.search_cache = SearchCache()
        self.image_loader = ImageLoader(self.settings['thumbnail_size'])
        self.search_debounce_timer = None
        self.negative_examples = {}
        self.positive_examples = {}

        #disable states
        self.displayed_count = 0
        self.load_batch_size = 40
        self.is_loading_more = False
        self.card_widgets = {}
        self.selected_images = set() 
        self.hover_preview_window = None
        self.hover_timer = None
        self.photo_refs = {}
        self.loaded_cards = set()

        self.create_ui()

        self.setup_shortcuts()

        self.start_callback_processor()

        #lood the index on a separate thread
        threading.Thread(target=self.load_index, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self) -> None:
        """Handle the closing of the application"""
        self.image_loader.shutdown()
        self.root.destroy()

    def start_callback_processor(self) -> None:
        """Start the callback processor loop"""
        self.image_loader.process_callbacks(self.root)

    def setup_shortcuts(self) -> None:
        """setup all the keyboard shortcuts"""
        self.root.bind('<Control-f>', lambda e: self.search_entry.focus())
        self.root.bind('<Escape>', lambda e: (self.search_entry.delete(0, 'end'), self.show_initial_message()))
        self.root.bind('<F1>', lambda e: self.show_shortcuts())
        self.root.bind('<Delete>', lambda e: self.delete_selected())

    def regenerate_thumbnails(self, new_size:int) -> None:
        """Regenerate all cached thumbnails at a new size whilst avoiding direct re-indexing"""
        if not self.image_paths:
            messagebox.showinfo("No Index", "No images indexed yet!")
            return

        self.progress_bar.pack(side='left', padx=15)
        self.progress_bar.set(0)
        self.status_label.configure(text=f"Regenerating {len(self.image_paths)} thumbnails at {new_size}x{new_size}...")

        threading.Thread(target=self._regenerate_thumbnails_thread, args=(new_size,), daemon=True).start()

    def _regenerate_thumbnails_thread(self, new_size:int) -> None:
        """Seperate threat to regenereate all the thumbnails"""

        try:
            thumb_folder = os.path.join(self.settings['index_folder'], 'thumbnails')

            #delete the old thumbnails
            self.root.after(0, lambda: self.status_label.configure(text="Deleting old thumbnails..."))

            if os.path.exists(thumb_folder):
                try:
                    shutil.rmtree(thumb_folder)
                except PermissionError:
                    #individual files could be locked, try deleteing manually one by one
                    for root, dirs, files in os.walk(thumb_folder):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except:
                                pass
                    #see if the empty directory can be removed
                    try:
                        shutil.rmtree(thumb_folder)
                    except: 
                        pass

            os.makedirs(thumb_folder, exist_ok=True)

            #run thumbnail generation with async
            completed = asyncio.run(self._regenerate_thumbnails(thumb_folder, new_size))

            #clear image cache to ensure that it uses new thumbnails instead
            self.image_loader.clear()

            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(text=f"✓ Regenerated {completed} thumbnails at {new_size}x{new_size}"))

            notification.notify(
                        title='Scout',
                        message=f'Thumbnails regenerated! {completed} images at {new_size}x{new_size}',
                        app_name='Scout',
                        timeout=5
            )

        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(text=f"❌ Regeneration failed: {str(e)}"))

    async def _regenerate_thumbnails(self, thumb_folder:str, new_size:int) -> int:
        """Generate thumbnails using async"""
        total = len(self.image_paths)
        completed = 0
        
        self.root.after(0, lambda: self.status_label.configure(text=f"Generating thumbnails at {new_size}x{new_size}..."))
        
        concurrent_batch = self.settings.get('processing_batch_size', 10)
        
        for i in range(0, total, concurrent_batch):
            batch_paths = self.image_paths[i:i + concurrent_batch]
            
            tasks = [self._process_single_thumbnail(path, thumb_folder, new_size) 
                    for path in batch_paths]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            completed += sum(1 for r in results if r is True)
            
            progress = completed / total
            self.root.after(0, lambda p=progress: self.progress_bar.set(p))
            self.root.after(0, lambda c=completed, t=total: 
                           self.status_label.configure(text=f"Regenerated {c}/{t} thumbnails at {new_size}x{new_size}"))
        
        return completed

    async def _process_single_thumbnail(self, path:str, thumb_folder:str, new_size:int) -> bool: #TODO: Ensure this is right return type
        """Process a single thumbnail using async"""
        try:
            if not os.path.exists(path):
                return False
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._create_thumbnail_sync, path, thumb_folder, new_size)
            return result
        except Exception as e:
            return False

    async def _generate_thumbnails_batch_async(self, paths, thumb_folder, cache_size, batch_size):
        """Generate thumbnails for a batch of paths asynchronously"""
        thumb_format = self.settings.get('thumbnail_format', 'webp')
        tasks = [self._process_single_thumbnail(path, thumb_folder, cache_size) for path in paths]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _create_thumbnail_sync(self, path:str, thumb_folder:str, cache_size:int) -> bool:
        """Synchronous thumbnail creation - called inside an async wrapper"""
        try:
            thumb_format = self.settings.get('thumbnail_format', 'webp')
            
            original_img = Image.open(path)
            
            # Handle transparency
            if original_img.mode == 'RGBA':
                background = Image.new('RGB', original_img.size, '#3a3a3a')
                background.paste(original_img, mask=original_img.split()[3])
                original_img = background
            elif original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            thumb = original_img.copy()
            thumb.thumbnail((cache_size, cache_size), Image.Resampling.LANCZOS)
            
            ext = '.webp' if thumb_format == 'webp' else '.jpg'
            thumb_name = str(abs(hash(path)) % 10**10) + ext
            thumb_path = os.path.join(thumb_folder, thumb_name)
            
            if thumb_format == 'webp':
                thumb.save(thumb_path, 'WEBP', quality=85, method=4)
            else:
                thumb.save(thumb_path, 'JPEG', quality=85, optimize=True)
            
            return True
        except Exception as e:
            return False
    
    async def _load_thumbnail_async(self, path:str, thumb_folder:str, new_size:int) -> bool:
        """Load in thumbnails for display - create where they don't exist"""
        try:
            if not os.path.exists(path):
                return False
            
            thumb_format = self.settings.get('thumbnail_format', 'webp')
            
            loop = asyncio.get_event_loop()
            img_data = await loop.run_in_executor(None, self._load_and_resize_thumbnail, path, new_size)
            
            if img_data is None:
                return False
            
            ext = '.webp' if thumb_format == 'webp' else '.jpg'
            thumb_name = str(abs(hash(path)) % 10**10) + ext
            thumb_path = os.path.join(thumb_folder, thumb_name)
            
            def save_thumb():
                if thumb_format == 'webp':
                    img_data.save(thumb_path, 'WEBP', quality=85, method=4)
                else:
                    img_data.save(thumb_path, 'JPEG', quality=85, optimize=True)
            
            await loop.run_in_executor(None, save_thumb)
            
            return True
            
        except Exception as e:
            return False

    def _load_and_resize_thumbnail(self, path: str, new_size: int) -> Optional[Image.Image]:
        """Load and resize image"""
        try:
            original_img = Image.open(path)
            
            if original_img.mode == 'RGBA':
                background = Image.new('RGB', original_img.size, '#3a3a3a')
                background.paste(original_img, mask=original_img.split()[3])
                original_img = background
            elif original_img.mode != 'RGB':
                original_img = original_img.convert('RGB')
            
            thumb = original_img.copy()
            thumb.thumbnail((new_size, new_size), Image.Resampling.LANCZOS)
            
            return thumb
        except Exception as e:
            return None
            
            notification.notify(
                title='Scout',
                message=f'Thumbnails regenerated! {completed} images at {new_size}x{new_size}',
                app_name='Scout',
                timeout=5
            )
        
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(text=f"❌ Regeneration failed: {str(e)}"))
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: messagebox.showerror("Error", f"Regeneration failed: {e}"))

    def show_shortcuts(self) -> None:
        """Show keyboard shortcuts help"""

        help_window = ctk.CTkToplevel(self.root)
        if os.path.exists("Scout.ico"):
            help_window.iconbitmap("Scout.ico")
        help_window.title("Keyboard Shortcuts")
        help_window.geometry("500x500")
        help_window.grab_set()
        
        ctk.CTkLabel(help_window, text="Keyboard Shortcuts", 
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        shortcuts_frame = ctk.CTkFrame(help_window)
        shortcuts_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        shortcuts = [
            ("Ctrl+F", "Focus search box"),
            ("Escape", "Clear search and results"),
            ("F1", "Show this help dialog"),
            ("Delete", "Delete selected images from database"),
            ("Ctrl+Click", "Select/deselect image (multi-select)"),
            ("Right Click", "Show context menu for image"),
            ("Middle Click", "Mark image as incorrect"),
            ("Ctrl+Middle Click", "Mark image as correct"),
        ]
        
        for key, desc in shortcuts:
            row = ctk.CTkFrame(shortcuts_frame, fg_color='transparent')
            row.pack(fill='x', pady=8, padx=10)
            
            ctk.CTkLabel(row, text=key, font=ctk.CTkFont(size=13, weight="bold"), 
                         width=120, anchor='w').pack(side='left')
            ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=12), 
                         anchor='w').pack(side='left', padx=15)
        
        ctk.CTkButton(help_window, text="Close", command=help_window.destroy, 
                      width=100, height=35).pack(pady=20)

    def load_settings(self) -> dict:
        """Loads settings from the json settings file"""

        settings_file = 'scout_settings.json'
        defaults = {
            'index_folder': 'scout_index',
            'images_folder': '',
            'clip_model': 'clip-ViT-B-32',
            'thumbnail_size': 300,
            'hover_preview_size': 800,
            'dynamic_search': True,
            'dynamic_delay': 500,
            'processing_batch_size': 10,
            'thumbnail_format': 'webp'
        }
        
        if os.path.exists(settings_file):
            try:
                return {**defaults, **load_json(settings_file)}
            except:
                return defaults
        return defaults
    
    def save_settings(self) -> None:
        """Saves settings to the json settings file"""

        save_json('scout_settings.json', self.settings)

    def create_ui(self) -> None:
        # Top bar
        top_frame = ctk.CTkFrame(self.root, height=120, corner_radius=0)
        top_frame.pack(fill='x', padx=0, pady=0)
        top_frame.pack_propagate(False)
        
        # Title
        title_frame = ctk.CTkFrame(top_frame, fg_color='transparent')
        title_frame.pack(side='left', padx=20, anchor='w')
        
        ctk.CTkLabel(title_frame, text="S C O U T", 
            font=ctk.CTkFont(size=28, weight="bold")).pack(anchor='w')
        
        # Index and folder management buttons
        button_frame = ctk.CTkFrame(top_frame, fg_color='transparent')
        button_frame.pack(side='left', padx=50)
        
        self.index_btn = ctk.CTkButton(button_frame, text="Index Images", command=self.open_indexer, width=140, height=40, font=ctk.CTkFont(size=14, weight="bold"))
        self.index_btn.pack(side='left', padx=(0, 10))
        
        self.refresh_btn = ctk.CTkButton(button_frame, text="🔄 Refresh All", command=self.refresh_all_folders, width=120, height=40, font=ctk.CTkFont(size=13, weight="bold"))
        self.refresh_btn.pack(side='left', padx=(0, 10))
        
        self.manage_btn = ctk.CTkButton(button_frame, text="📁 Manage Folders", command=self.open_folder_manager, width=140, height=40, font=ctk.CTkFont(size=13, weight="bold"))
        self.manage_btn.pack(side='left')
        
        # Search and buttons
        right_frame = ctk.CTkFrame(top_frame, fg_color='transparent')
        right_frame.pack(side='right', padx=20, anchor='e')
        
        self.search_entry = ctk.CTkEntry(right_frame, placeholder_text="Search Images...", width=400, height=40, font=ctk.CTkFont(size=14))
        self.search_entry.pack(side='left')
        self.search_entry.bind('<KeyRelease>', self.on_search_keypress)
        
        ctk.CTkButton(right_frame, text="⚙", command=self.open_settings, width=40, height=40, font=ctk.CTkFont(size=20)).pack(side='left', padx=(10, 0))
        ctk.CTkButton(right_frame, text="?", command=self.show_shortcuts, width=40, height=40, font=ctk.CTkFont(size=20)).pack(side='left', padx=(5, 0))
        
        # Status with progress bar
        status_frame = ctk.CTkFrame(self.root, height=50, corner_radius=0)
        status_frame.pack(fill='x', padx=0, pady=0)
        
        status_content = ctk.CTkFrame(status_frame, fg_color='transparent')
        status_content.pack(fill='x', padx=20, pady=5)
        
        self.status_label = ctk.CTkLabel(status_content, text="Loading index...", font=ctk.CTkFont(size=12))
        self.status_label.pack(side='left')
        
        self.progress_bar = ctk.CTkProgressBar(status_content, width=200, height=8)
        self.progress_bar.pack(side='left', padx=15)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()  # Hidden by default
        
        # Results
        self.results_frame = ctk.CTkScrollableFrame(self.root, corner_radius=0, fg_color='transparent')
        self.results_frame.pack(fill='both', expand=True, padx=0, pady=0)
        
        # Enable drag and drop
        self.setup_drag_drop()
        
        self.setup_mousewheel()
        
        # Footer
        footer_frame = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        footer_frame.pack(fill='x', side='bottom')
        
        self.footer_label = ctk.CTkLabel(footer_frame, text="Ready", font=ctk.CTkFont(size=11))
        self.footer_label.pack(pady=5)
        
        self.show_initial_message()
        
        # Start lazy load checker
        self.check_load_more_periodically()

    def setup_drag_drop(self) -> None:
        """Setup drag and drop for images/folders using windnd"""
        try:
            #try to import windnd if it can't be it's an optional function 
            #that isn't required for core functionality
            import windnd
            
            windnd.hook_dropfiles(self.root, func=self.on_drop)
            try:
                windnd.hook_dropfiles(self.results_frame, func=self.on_drop)
            except:
                pass
                
        except ImportError:
            pass

    def on_drop(self, files:bytes | str | Sequence[bytes | str]) -> None:
        """Handle dropped files/folders"""
        try:
            # files is a list of file paths (bytes in windnd)
            if isinstance(files, bytes):
                files = files.decode('utf-8')
            
            if isinstance(files, str):
                files = [files]
            elif isinstance(files, (list, tuple)):
                files = [f.decode('utf-8') if isinstance(f, bytes) else f for f in files]
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            folders_to_index = []
            files_to_index = []
            
            for item in files:
                if os.path.isdir(item):
                    folders_to_index.append(item)
                elif os.path.isfile(item):
                    ext = os.path.splitext(item)[1].lower()
                    if ext in image_extensions:
                        files_to_index.append(item)
            
            if not folders_to_index and not files_to_index:
                messagebox.showinfo("No Images", "No image files or folders were dropped")
                return
            
            if folders_to_index:
                msg = f"Index {len(folders_to_index)} folder(s)?"
                if files_to_index:
                    msg += f"\n(Plus {len(files_to_index)} individual files)"
            else:
                msg = f"Index {len(files_to_index)} image file(s)?"
            
            if messagebox.askyesno("Index Dropped Items", msg):
                # Index folders
                for folder in folders_to_index:
                    self.quick_index(folder)
                
                # Then index individual files
                if files_to_index:
                    self.quick_index_files(files_to_index)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to process dropped items: {e}")

    def setup_mousewheel(self) -> None:
        """Setup mouse wheel scrolling"""
        canvas = self.results_frame._parent_canvas
        canvas.configure(yscrollincrement=20)
        
        def on_mousewheel(event):
            if hasattr(event, 'delta') and event.delta:
                canvas.yview_scroll(-1 * int(event.delta / 30), "units")
            elif event.num == 4:
                canvas.yview_scroll(-4, "units")
            elif event.num == 5:
                canvas.yview_scroll(4, "units")
            return "break"
        
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", on_mousewheel)
        canvas.bind("<Button-5>", on_mousewheel)

    def check_load_more_periodically(self) -> None:
        """Check if we need to load more results"""
        try:
            if not self.is_loading_more and self.current_results:
                if self.displayed_count < len(self.current_results):
                    canvas = self.results_frame._parent_canvas
                    yview = canvas.yview()
                    
                    # Load more when 70% scrolled
                    if yview[1] > 0.7:
                        self.load_more_results()
        except:
            pass
        finally:
            self.root.after(500, self.check_load_more_periodically)

    def show_initial_message(self) -> None:
        self.clear_results()
        msg_frame = ctk.CTkFrame(self.results_frame, fg_color='transparent')
        msg_frame.pack(expand=True, pady=150)
        
        if self.model is None and os.path.exists(self.settings['index_folder']):
            ctk.CTkLabel(msg_frame, text="⏳ Loading AI Model...", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=10)
            ctk.CTkLabel(msg_frame, text="Please wait, this may take a minute", font=ctk.CTkFont(size=14)).pack()
        else:
            ctk.CTkLabel(msg_frame, text="Search for images or click 'Index Images' to add more", font=ctk.CTkFont(size=16)).pack()
    
    def on_search_keypress(self, event=None) -> None:
        if not self.settings['dynamic_search']:
            return
        
        if self.search_debounce_timer:
            self.root.after_cancel(self.search_debounce_timer)
        
        self.search_debounce_timer = self.root.after(self.settings['dynamic_delay'], self.perform_search)

    def load_index(self) -> None:
        try:
            index_folder = self.settings['index_folder']
            
            if not os.path.exists(index_folder):
                self.root.after(0, lambda: self.status_label.configure(text="No index found. Click 'Index Images' to get started."))
                return
            
            self.root.after(0, lambda: self.status_label.configure(text=f"Loading {self.settings['clip_model']}..."))
            
            SentenceTransformer = lazy_import_sentence_transformers()
            
            #import torch here to improve initial GUI opening
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = SentenceTransformer(
                self.settings['clip_model'],
                device=device,
                cache_folder=None
            )
            
            self.model.eval()
            if device == 'cpu':
                torch.set_num_threads(4)
            
            with open(os.path.join(index_folder, 'embeddings.pkl'), 'rb') as f:
                self.embeddings = pickle.load(f)
            
            self.image_paths = load_json(os.path.join(index_folder, 'image_paths.json'))
            self.image_paths_set = set(self.image_paths) 
            self.metadata = load_json(os.path.join(index_folder, 'metadata.json'))
            
            neg_file = os.path.join(index_folder, 'negative_examples.json')
            if os.path.exists(neg_file):
                self.negative_examples = load_json(neg_file)
            
            pos_file = os.path.join(index_folder, 'positive_examples.json')
            if os.path.exists(pos_file):
                self.positive_examples = load_json(pos_file)
            
            self.search_cache.invalidate_if_needed(self.metadata.get('indexed_at', time.time()))
            
            # Build FAISS index for fast search on large datasets (10K+)
            self.faiss_index = self._build_faiss_index()
            
            self.root.after(0, lambda: self.status_label.configure(text=f"Loaded {len(self.image_paths)} images"))
            self.root.after(0, lambda: self.show_initial_message())  # Refresh message now that model is loaded
            
        except Exception as ex:
            self.root.after(0, lambda: self.status_label.configure(text=f"Error: {str(ex)}"))

    def _build_faiss_index(self) -> faiss.Index | None:
        """Build FAISS index for fast similarity search on large datasets"""
        if self.embeddings is None:
            return None
        
        num_images = len(self.image_paths)
        
        # Only use FAISS for datasets with 10K+ images, otherwise it would be ineffecient
        if num_images < 10000:
            return None
        
        try:
            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.embeddings.astype('float32')
            faiss.normalize_L2(embeddings_normalized)
            
            dimension = embeddings_normalized.shape[1]
            
            if num_images < 50000:
                # For 10K-50K: Use IVFFlat (good balance of speed and accuracy)
                nlist = min(int(np.sqrt(num_images)), 256)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                # For 50K+: Use IVF with Product Quantization (more compression)
                nlist = min(int(np.sqrt(num_images)), 512)
                m = 32  # Number of subquantizers
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            
            index.train(embeddings_normalized)
            index.add(embeddings_normalized)
            
            index.nprobe = min(32, nlist)
            
            return index
            
        except Exception as e:
            return None

    def perform_search(self) -> None:
        query = self.search_entry.get().strip()
        if not query or not self.model:
            if not query:
                self.show_initial_message()
            return

        cached = self.search_cache.get(query)
        if cached:
            self.display_results(cached, from_cache=True)
            return

        self.footer_label.configure(text="Searching...")
        threading.Thread(target=self._search_thread, args=(query,), daemon=True).start()

    def _search_thread(self, query:str) -> None:
        """Searching with filename matching"""
        try:
            start_time = time.time()
            
            query_embedding = self.model.encode([query])[0]
            
            if self.faiss_index is not None:
                query_normalized = query_embedding.astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_normalized)
                
                # Search for top 1000 results
                k = min(1000, len(self.image_paths))
                distances, indices = self.faiss_index.search(query_normalized, k)
                
                # Convert FAISS results to similarity scores (distances are inner products = cosine similarity)
                image_similarities = np.zeros(len(self.image_paths))
                for i, idx in enumerate(indices[0]):
                    if idx >= 0:  # FAISS uses -1 for missing results
                        image_similarities[idx] = distances[0][i]
            else:
                # NumPy search - for smaller datasets or when FAISS unavailable
                image_similarities = np.dot(self.embeddings, query_embedding)
            
            filename_scores = np.zeros(len(self.image_paths))
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Batch process filenames
            for idx, path in enumerate(self.image_paths):
                filename = os.path.basename(path).lower()
                filename_no_ext = os.path.splitext(filename)[0]
                
                if query_lower in filename:
                    filename_scores[idx] = 0.5
                    continue
                
                filename_words = set(filename_no_ext.replace('_', ' ').replace('-', ' ').split())
                if query_words & filename_words:
                    overlap = len(query_words & filename_words) / len(query_words)
                    filename_scores[idx] = overlap * 0.3
            
            # Combine scores
            combined_scores = image_similarities + filename_scores
            
            # Apply positive/negative examples
            if query in self.positive_examples:
                for img_path in self.positive_examples[query]:
                    try:
                        idx = self.image_paths.index(img_path)
                        combined_scores[idx] *= 1.5
                    except ValueError:
                        pass
            
            if query in self.negative_examples:
                for img_path in self.negative_examples[query]:
                    try:
                        idx = self.image_paths.index(img_path)
                        combined_scores[idx] *= 0.1
                    except ValueError:
                        pass
            
            sorted_indices = np.argsort(combined_scores)[::-1]
            
            # Build results
            results = []
            for idx in sorted_indices:
                score = float(combined_scores[idx])
                if score > 0.25: #only show 25% of the more relevant results
                    results.append({
                        'path': self.image_paths[idx],
                        'score': score,
                        'filename': os.path.basename(self.image_paths[idx])
                    })
            
            self.search_cache.set(query, results)
            elapsed = time.time() - start_time
            
            search_method = "FAISS" if self.faiss_index is not None else "NumPy"
            
            self.root.after(0, lambda: self.display_results(results, search_time=elapsed))
            self.root.after(0, lambda: self.footer_label.configure(
                text=f"Showing {len(results)} results in {elapsed:.2f}s"
            ))
            
        except Exception as ex:
            self.root.after(0, lambda: self.footer_label.configure(text=f"Error: {str(ex)}"))
    
    def clear_results(self) -> None:
        """Clear current results from the UI"""

        self.card_widgets.clear()
        self.selected_images.clear()
        self.photo_refs.clear()
        self.loaded_cards.clear() 
        try:
            for widget in list(self.results_frame.winfo_children()):
                try:
                    widget.destroy()
                except:
                    pass
        except:
            pass
    
    def display_results(self, results: List[Dict], from_cache:bool=False, preserve_scroll:bool=False, search_time=None) -> None:
        """Display search results in the UI"""

        self.current_results = results
        self.displayed_count = 0
        self.is_loading_more = False
        
        scroll_pos = 0
        if preserve_scroll:
            try:
                canvas = self.results_frame._parent_canvas
                scroll_pos = canvas.yview()[0]
            except:
                pass
        
        self.clear_results()
        
        if not preserve_scroll:
            try:
                canvas = self.results_frame._parent_canvas
                canvas.yview_moveto(0)
            except:
                pass
        
        if not results:
            ctk.CTkLabel(self.results_frame, text="No results found.", font=ctk.CTkFont(size=14)).pack(pady=100)
            return
        
        if from_cache:
            time_str = f" in {search_time:.2f}s" if search_time else ""
            self.footer_label.configure(text=f"Showing {len(results)} results{time_str}")
        
        self.grid_container = ctk.CTkFrame(self.results_frame, fg_color='transparent')
        self.grid_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        cols = 4
        for i in range(cols):
            self.grid_container.grid_columnconfigure(i, weight=1, uniform='col')
        
        self.load_more_results()
        
        if preserve_scroll and scroll_pos > 0:
            try:
                canvas = self.results_frame._parent_canvas
                self.root.after(100, lambda: canvas.yview_moveto(scroll_pos))
            except:
                pass
    
    def load_more_results(self) -> None:
        """Load more result cards with lazy loading and prioritization"""

        if self.is_loading_more:
            return
        
        self.is_loading_more = True
        
        start = self.displayed_count
        end = min(start + self.load_batch_size, len(self.current_results))
        
        self._load_cards_prioritized(start, end)
    
    def _load_cards_prioritized(self, start:int, end:int) -> None:
        """Load visible cards first, then buffer cards"""

        VISIBLE_CARDS = 8
        
        visible_end = min(start + VISIBLE_CARDS, end)
        cols = 4
        
        for i in range(start, visible_end):
            result = self.current_results[i]
            row = i // cols
            col = i % cols
            self.create_result_card(result, row, col)
        
        self.displayed_count = visible_end
        
        if visible_end < end:
            self.root.after(50, lambda: self._load_buffer_cards(visible_end, end))
        else:
            self.is_loading_more = False
            self._update_footer_count()
    
    def _load_buffer_cards(self, start:int, end:int, index=None) -> None:
        """Load buffer cards in small chunks"""

        if index is None:
            index = start
        
        if index >= end:
            self.is_loading_more = False
            self._update_footer_count()
            return
        
        chunk_end = min(index + 4, end)
        cols = 4
        
        for i in range(index, chunk_end):
            result = self.current_results[i]
            row = i // cols
            col = i % cols
            self.create_result_card(result, row, col)
        
        self.displayed_count = chunk_end
        
        if chunk_end < end:
            self.root.after(20, lambda: self._load_buffer_cards(start, end, chunk_end))
        else:
            self.is_loading_more = False
            self._update_footer_count()
    
    def _update_footer_count(self) -> None:
        """Update footer with current count"""

        if self.displayed_count < len(self.current_results):
            self.footer_label.configure(text=f"Showing {self.displayed_count} of {len(self.current_results)} results")
        else:
            self.footer_label.configure(text=f"Showing all {len(self.current_results)} results")
    
    def create_result_card(self, result: Dict, row: int, col: int) -> None:
        card_id = f"{row}_{col}"
        img_path = result['path']
        
        card = ctk.CTkFrame(self.grid_container, corner_radius=8)
        card.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
        
        # Store card reference by BOTH card_id and path for easier lookup
        self.card_widgets[card_id] = card
        self.card_widgets[img_path] = card
        
        # Bind mouse buttons
        card.bind('<Button-3>', lambda e: self.show_context_menu(e, result)) 
        card.bind('<Button-2>', lambda e: self.mark_incorrect(result['path']))
        card.bind('<Control-Button-2>', lambda e: self.mark_correct(result['path']))
        
        if result['path'] in self.image_loader.cache:
            img = self.image_loader.cache[result['path']]
            self.display_card_content(card_id, img, result)
            self.loaded_cards.add(card_id) 
        else:
            placeholder = ctk.CTkFrame(card, width=300, height=300, fg_color='gray30')
            placeholder.pack(padx=8, pady=8)
            
            self.image_loader.load_thumbnail(
                result['path'], 
                lambda img, cid=card_id, r=result: self.on_thumbnail_ready(cid, img, r)
            )
    
    def on_thumbnail_ready(self, card_id: int, img: Image.Image, result: dict) -> None:
        """Called when thumbnail is loaded - only render once"""
        try:
            # Skip if already loaded
            if card_id in self.loaded_cards:
                return
            
            if card_id not in self.card_widgets:
                return
            
            card = self.card_widgets[card_id]
            
            if not card.winfo_exists():
                return
            
            # Clear placeholder
            for widget in list(card.winfo_children()):
                try:
                    if widget.winfo_exists():
                        widget.destroy()
                except:
                    pass
            
            # Display content
            self.display_card_content(card_id, img, result)
            
            # Mark as loaded
            self.loaded_cards.add(card_id)
            
        except Exception as e:
            pass
    
    def display_card_content(self, card_id:int, img:Image.Image, result:dict):
        """Display card content - only called once per card"""
        try:
            # Double-check not already loaded
            if card_id in self.loaded_cards:
                return
            
            if card_id not in self.card_widgets:
                return
            
            card = self.card_widgets[card_id]
            
            if not card.winfo_exists():
                return
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, '#3a3a3a')
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            photo = ImageTk.PhotoImage(img, master=self.root)
            
            self.photo_refs[card_id] = photo
            card_color = '#3a3a3a'
            
            img_label = tk.Label(card, image=photo, cursor='hand2', bg=card_color)
            img_label.image = photo
            img_label.pack(padx=8, pady=8)
            
            # Mouse bindings
            img_label.bind('<Button-1>', lambda e: self.toggle_selection(result['path'], card) if e.state & 0x0004 else self.open_image(result['path']))
            img_label.bind('<Button-3>', lambda e: self.show_context_menu(e, result))
            img_label.bind('<Button-2>', lambda e: self.mark_incorrect(result['path'])) 
            img_label.bind('<Control-Button-2>', lambda e: self.mark_correct(result['path']))
            img_label.bind('<Enter>', lambda e: self.show_hover_preview(e, result['path']))
            img_label.bind('<Leave>', lambda e: self.hide_hover_preview())
            
            info_frame = ctk.CTkFrame(card, fg_color='transparent')
            info_frame.pack(fill='x', padx=10, pady=(0, 10))
            
            filename = result['filename']
            if len(filename) > 25:
                filename = filename[:22] + "..."
            
            ctk.CTkLabel(info_frame, text=filename, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor='w')
            ctk.CTkLabel(info_frame, text=f"{result['score']*100:.1f}%", font=ctk.CTkFont(size=10)).pack(anchor='w', pady=(2, 0))
            
            if result['path'] in self.selected_images:
                card.configure(border_width=3, border_color='#1f6aa5')
        except:
            pass
    
    def toggle_selection(self, img_path:str, card:ctk.CTkFrame) -> None:
        """Toggle image selection"""

        if img_path in self.selected_images:
            self.selected_images.remove(img_path)
            card.configure(border_width=0)
        else:
            self.selected_images.add(img_path)
            card.configure(border_width=3, border_color='#1f6aa5')
    
    def show_hover_preview(self, event, img_path:str) -> None:
        """Show large preview on hover - faster and more reliable"""

        # Cancel any existing timer
        if self.hover_timer:
            self.root.after_cancel(self.hover_timer)
            self.hover_timer = None
        
        # Schedule new preview
        self.hover_timer = self.root.after(800, lambda: self._create_preview_safe(event, img_path))
    
    def _create_preview_safe(self, event, img_path:str) -> None:
        """Wrapper to handle preview creation safely"""

        try:
            self._create_preview(event, img_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def hide_hover_preview(self, event=None) -> None:
        """Hide hover preview"""

        if self.hover_timer:
            self.root.after_cancel(self.hover_timer)
            self.hover_timer = None
        
        if self.hover_preview_window:
            self.hover_preview_window.destroy()
            self.hover_preview_window = None
    
    def _create_preview(self, event, img_path:str) -> None:
        """Create the preview window with smart positioning"""

        try:
            if self.hover_preview_window:
                try:
                    self.hover_preview_window.destroy()
                except:
                    pass
                self.hover_preview_window = None
            
            if not os.path.exists(img_path):
                return
            
            size = self.settings.get('hover_preview_size', 800)
            
            try:
                img = Image.open(img_path)
            except Exception as e:
                return
            
            try:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
            except Exception as e:
                return
            
            # Thumbnail
            try:
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
            except Exception as e:
                return
            
            try:
                self.hover_preview_window = ctk.CTkToplevel(self.root)
                if os.path.exists("Scout.ico"):
                    self.hover_preview_window.iconbitmap("Scout.ico")
                self.hover_preview_window.overrideredirect(True)
                self.hover_preview_window.attributes('-topmost', True)
            except Exception as e:
                return
            
            #screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            img_width, img_height = img.size
            
            # Get mouse position
            try:
                mouse_x = event.x_root
                mouse_y = event.y_root
            except:
                mouse_x = self.root.winfo_pointerx()
                mouse_y = self.root.winfo_pointery()
            
            # Calculate position with better logic
            # Try to place to the right and below the cursor first
            x = mouse_x + 20
            y = mouse_y + 20
            
            # If would go off right edge, place to the left of cursor instead
            if x + img_width > screen_width - 20:
                x = mouse_x - img_width - 20
                # If still off screen, just position from right edge
                if x < 20:
                    x = screen_width - img_width - 20
            
            # If would go off bottom edge, place above cursor instead
            if y + img_height > screen_height - 20:
                y = mouse_y - img_height - 20
                # If still off screen, just position from bottom edge
                if y < 20:
                    y = screen_height - img_height - 20
            
            x = max(20, min(x, screen_width - img_width - 20))
            y = max(20, min(y, screen_height - img_height - 20))
            
            try:
                self.hover_preview_window.geometry(f"+{x}+{y}")
            except Exception as e:
                return
            
            try:
                photo = ImageTk.PhotoImage(img, master=self.hover_preview_window)
                label = tk.Label(self.hover_preview_window, image=photo, bg='#2b2b2b', bd=2, relief='solid')
                label.image = photo
                label.pack()
            except Exception as e:
                if self.hover_preview_window:
                    self.hover_preview_window.destroy()
                    self.hover_preview_window = None
                return
            
            try:
                self.hover_preview_window.update_idletasks()
                self.hover_preview_window.lift()
                self.hover_preview_window.focus_force()
            except Exception as e:
                pass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.hover_preview_window:
                try:
                    self.hover_preview_window.destroy()
                except:
                    pass
                self.hover_preview_window = None
    
    def show_context_menu(self, event, result: Dict) -> None:
        """Show context menu for a result"""

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Open", command=lambda: self.open_image(result['path']))
        menu.add_command(label="Show in Folder", command=lambda: self.show_in_folder(result['path']))
        menu.add_separator()
        menu.add_command(label="Find Similar Images", command=lambda: self.find_similar(result['path']))
        menu.add_separator()
        menu.add_command(label="Mark as correct (Ctrl+Middle Click)", command=lambda: self.mark_correct(result['path']))
        menu.add_command(label="Mark as incorrect (Middle Click)", command=lambda: self.mark_incorrect(result['path']))
        menu.add_separator()
        menu.add_command(label="Delete from database", command=lambda: self.delete_single(result['path']))
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def find_similar(self, img_path: str) -> None:
        """Find images similar to the selected image"""

        try:
            idx = self.image_paths.index(img_path)
            img_embedding = self.embeddings[idx]
            
            img_embedding_norm = img_embedding / (np.linalg.norm(img_embedding) + 1e-8)
            
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            embeddings_norm = self.embeddings / (norms + 1e-8)
            
            # Calculate cosine similarities
            similarities = np.dot(embeddings_norm, img_embedding_norm)
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for i in sorted_indices:
                if self.image_paths[i] == img_path:
                    continue
                if not os.path.exists(self.image_paths[i]):
                    continue
                
                score = float(similarities[i])
                # Lower threshold to 0.3 to get more similar images
                if score > 0.3:
                    results.append({
                        'path': self.image_paths[i],
                        'score': score,
                        'filename': os.path.basename(self.image_paths[i])
                    })
                
                if len(results) >= 100:
                    break
            
            # Update search box and display
            self.search_entry.delete(0, 'end')
            self.search_entry.insert(0, f"Similar to: {os.path.basename(img_path)}")
            self.display_results(results)
            self.footer_label.configure(text=f"Found {len(results)} similar images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find similar images: {e}")
    
    def mark_correct(self, path: str) -> None:
        """Mark image as correct/positive example for current query"""

        query = self.search_entry.get().strip()
        if not query:
            return
        
        if query not in self.positive_examples:
            self.positive_examples[query] = []
        
        if path not in self.positive_examples[query]:
            self.positive_examples[query].append(path)
        
        if query in self.search_cache.cache:
            del self.search_cache.cache[query]
        
        try:
            pos_file = os.path.join(self.settings['index_folder'], 'positive_examples.json')
            save_json(pos_file, self.positive_examples)
        except Exception as e:
            pass
        
        # Green flash boarder as a visual indicator to the image being marked as correct
        if path in self.card_widgets:
            try:
                card = self.card_widgets[path]
                original_border_width = card.cget('border_width')
                original_border_color = card.cget('border_color')
                
                card.configure(border_width=3, border_color='#00ff00')
                
                self.root.after(500, lambda: card.configure(
                    border_width=original_border_width,
                    border_color=original_border_color
                ))
            except:
                pass
        
        self.footer_label.configure(text=f"✓ Marked as correct for '{query}' (will boost in next search)")
    
    def _save_positive_examples(self) -> None:
        """Background save of positive examples (not used anymore)"""
        
        try:
            pos_file = os.path.join(self.settings['index_folder'], 'positive_examples.json')
            save_json(pos_file, self.positive_examples)
        except:
            pass
    
    def mark_incorrect(self, path: str) -> None:
        """Mark image as incorrect/negative example for current query"""

        query = self.search_entry.get().strip()
        if not query:
            return
        
        if query not in self.negative_examples:
            self.negative_examples[query] = []
        
        if path not in self.negative_examples[query]:
            self.negative_examples[query].append(path)
        
        query_key = query.lower().strip()
        if query_key in self.search_cache.cache:
            del self.search_cache.cache[query_key]
        
        try:
            neg_file = os.path.join(self.settings['index_folder'], 'negative_examples.json')
            save_json(neg_file, self.negative_examples)
        except Exception as e:
            pass
        
        #Destroy the card widget instead of recreating the whole grid        
        if path in self.card_widgets:
            try:
                card = self.card_widgets[path]
                card.destroy()
                
                keys_to_delete = [k for k, v in self.card_widgets.items() if v == card]
                for key in keys_to_delete:
                    del self.card_widgets[key]
                
                self.current_results = [r for r in self.current_results if r['path'] != path]
                
                self.footer_label.configure(text=f"Showing {len(self.current_results)} results (marked 1 incorrect)")
            except Exception as e:
                import traceback
                traceback.print_exc()
    
    def delete_single(self, img_path:str) -> None:
        """Delete single image from database"""

        self.selected_images = {img_path}
        self.delete_selected()
    
    def delete_selected(self) -> None:
        """Delete selected images from database"""

        if not self.selected_images:
            return
        
        count = len(self.selected_images)
        if not messagebox.askyesno("Confirm Delete", 
                                   f"Remove {count} image(s) from database?\n\n(Files will NOT be deleted from disk)"):
            return
        
        # Track deleted images
        deleted_by_user_file = os.path.join(self.settings['index_folder'], 'deleted_by_user.json')
        deleted_by_user = set()
        if os.path.exists(deleted_by_user_file):
            deleted_by_user = set(load_json(deleted_by_user_file))
        
        # Build list of indices to remove
        indices_to_remove = []
        removed_paths = []
        
        for img_path in list(self.selected_images):
            try:
                idx = self.image_paths.index(img_path)
                indices_to_remove.append(idx)
                removed_paths.append(img_path)
                deleted_by_user.add(img_path)
            except ValueError:
                pass
        
        if not indices_to_remove:
            return
        
        #Use boolean mask instead of np.delete (Much faster)
        mask = np.ones(len(self.image_paths), dtype=bool)
        mask[indices_to_remove] = False
        self.embeddings = self.embeddings[mask]
        
        # Remove paths in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            self.image_paths.pop(idx)
        
        # Save list of user-deleted images
        save_json(deleted_by_user_file, list(deleted_by_user))
        self._save_index()
        
        self.search_cache.clear()
        
        # Remove cards from UI without recreating grid
        for img_path in removed_paths:
            if img_path in self.card_widgets:
                try:
                    card = self.card_widgets[img_path]
                    card.destroy()
                    
                    keys_to_delete = [k for k, v in self.card_widgets.items() if v == card]
                    for key in keys_to_delete:
                        del self.card_widgets[key]
                except Exception as e:
                    pass
        
        if self.current_results:
            self.current_results = [r for r in self.current_results if r['path'] not in removed_paths]
        
        self.selected_images.clear()
        
        if self.current_results:
            self.footer_label.configure(text=f"Showing {len(self.current_results)} results (deleted {len(removed_paths)} from DB)")
        else:
            self.footer_label.configure(text=f"Deleted {len(removed_paths)} images from database")
    
    def _save_index(self) -> None:
        """Save current index to disk"""

        try:
            output = self.settings['index_folder']
            
            with open(os.path.join(output, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            save_json(os.path.join(output, 'image_paths.json'), self.image_paths)
            
            folder_counts = {}
            for path in self.image_paths:
                folder = os.path.dirname(path)
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            save_json(os.path.join(output, 'metadata.json'), {
                'total_images': len(self.image_paths),
                'model': self.settings['clip_model'],
                'indexed_folder': self.metadata.get('indexed_folder', ''),
                'indexed_at': time.time(),
                'folder_counts': folder_counts 
            })
            
            self.metadata['total_images'] = len(self.image_paths)
            self.metadata['folder_counts'] = folder_counts
            
            # Rebuild FAISS index if dataset crossed 10K threshold or index exists
            if len(self.image_paths) >= 10000 or self.faiss_index is not None:
                self.faiss_index = self._build_faiss_index()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save index: {e}")
    
    def open_image(self, path: str) -> None:
        """Open image with default system viewer"""

        os.startfile(path)
    
    def show_in_folder(self, path: str) -> None:
        """Show image in system file explorer"""

        os.system(f'explorer /select,"{path}"')
    
    def quick_index(self, folder) -> None:
        """Quick index a folder without opening the indexer dialog"""

        threading.Thread(target=self._quick_index_thread, args=(folder,), daemon=True).start()
        self.status_label.configure(text=f"Indexing {os.path.basename(folder)}...")
    
    def quick_index_files(self, files) -> None:
        """Quick index individual files"""

        threading.Thread(target=self._quick_index_files_thread, args=(files,), daemon=True).start()
        self.status_label.configure(text=f"Indexing {len(files)} files...")
    
    def _quick_index_thread(self, folder) -> None:
        """Background thread for quick folder indexing"""

        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            image_files = [str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in image_extensions]
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showinfo("No Images", f"No images found in {os.path.basename(folder)}"))
                return
            
            existing_paths = set(self.image_paths) if self.image_paths else set()
            new_image_files = [f for f in image_files if f not in existing_paths]
            
            if not new_image_files:
                self.root.after(0, lambda: self.status_label.configure(text="All images already indexed"))
                return
            
            self._index_images_quick(new_image_files, folder)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
    
    def _quick_index_files_thread(self, files) -> None:
        """Background thread for quick file indexing"""

        try:
            existing_paths = set(self.image_paths) if self.image_paths else set()
            new_files = [f for f in files if f not in existing_paths]
            
            if not new_files:
                self.root.after(0, lambda: self.status_label.configure(text="All files already indexed"))
                return
            
            self._index_images_quick(new_files, "Dropped Files")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
    
    def _index_images_quick(self, image_files:list, source_name:str) -> None:
        """Quick indexing without progress dialog"""

        try:
            self.root.after(0, lambda: self.progress_bar.pack(side='left', padx=15))
            self.root.after(0, lambda: self.progress_bar.set(0))
            
            SentenceTransformer = lazy_import_sentence_transformers()
            
            if not self.model:
                self.model = SentenceTransformer(self.settings['clip_model'])
            
            batch_size = 32
            total = len(image_files)
            
            # Pre-allocate numpy array
            embedding_dim = 512
            all_embeddings = np.zeros((total, embedding_dim), dtype=np.float32)
            valid_paths = []
            valid_count = 0
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                chunk_size = 256
                
                for chunk_start in range(0, total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total)
                    chunk_files = image_files[chunk_start:chunk_end]
                    
                    futures = {executor.submit(self._load_for_index, path): path for path in chunk_files}
                    
                    batch_images = []
                    batch_paths = []
                    
                    for future, path in futures.items():
                        try:
                            img = future.result()
                            if img is not None:
                                batch_images.append(img)
                                batch_paths.append(path)
                        except:
                            pass
                        
                        # Encode when batch is full
                        if len(batch_images) >= batch_size:
                            embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                            
                            batch_len = len(embeddings)
                            all_embeddings[valid_count:valid_count + batch_len] = embeddings
                            valid_paths.extend(batch_paths)
                            valid_count += batch_len
                            
                            batch_images = []
                            batch_paths = []
                            
                            progress = valid_count / total
                            self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                            self.root.after(0, lambda c=valid_count, t=total: 
                                          self.status_label.configure(text=f"Indexed {c}/{t} from {source_name}"))
                    
                    # Process remaining in chunk
                    if batch_images:
                        embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                        
                        batch_len = len(embeddings)
                        all_embeddings[valid_count:valid_count + batch_len] = embeddings
                        valid_paths.extend(batch_paths)
                        valid_count += batch_len
                        
                        progress = valid_count / total
                        self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                        self.root.after(0, lambda c=valid_count, t=total: 
                                      self.status_label.configure(text=f"Indexed {c}/{t} from {source_name}"))
            
            all_embeddings = all_embeddings[:valid_count]
            
            # Generate thumbnails once at end
            thumb_folder = os.path.join(self.settings['index_folder'], 'thumbnails')
            os.makedirs(thumb_folder, exist_ok=True)
            cache_size = self.settings.get('cached_thumbnail_size', 300)
            user_batch_size = self.settings.get('processing_batch_size', 10)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._generate_thumbnails_batch_async(valid_paths, thumb_folder, cache_size, user_batch_size)
                )
            finally:
                loop.close()
            
            # Save to index
            output = self.settings['index_folder']
            os.makedirs(output, exist_ok=True)
            
            if self.embeddings is not None and len(self.embeddings) > 0:
                combined_embeddings = np.concatenate([self.embeddings, all_embeddings], axis=0)
            else:
                combined_embeddings = all_embeddings
            
            combined_paths = self.image_paths + valid_paths
            
            with open(os.path.join(output, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(combined_embeddings, f)
            
            save_json(os.path.join(output, 'image_paths.json'), combined_paths)
            
            # Calculate folder counts for cache
            folder_counts = {}
            for path in combined_paths:
                folder = os.path.dirname(path)
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            save_json(os.path.join(output, 'metadata.json'), {
                'total_images': len(combined_paths),
                'model': self.settings['clip_model'],
                'indexed_folder': source_name,
                'indexed_at': time.time(),
                'folder_counts': folder_counts
            })
            
            self.load_index()
            self.search_cache.clear()
            self.image_loader.clear()
            
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(
                text=f"Indexed {len(valid_paths)} images from {source_name}"))
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: messagebox.showerror("Error", f"Quick index failed: {e}"))
    
    def open_indexer(self) -> None:
        indexer_window = ctk.CTkToplevel(self.root)
        if os.path.exists("Scout.ico"):
            indexer_window.iconbitmap("Scout.ico")
        indexer_window.title("Index Images")
        indexer_window.geometry("600x450")
        indexer_window.grab_set()
        
        ctk.CTkLabel(indexer_window, text="Index Images", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        folder_frame = ctk.CTkFrame(indexer_window, fg_color='transparent')
        folder_frame.pack(fill='x', padx=20, pady=10)
        
        ctk.CTkLabel(folder_frame, text="Images Folder:", font=ctk.CTkFont(size=14)).pack(anchor='w')
        
        folder_entry_frame = ctk.CTkFrame(folder_frame, fg_color='transparent')
        folder_entry_frame.pack(fill='x', pady=5)
        
        folder_var = tk.StringVar(value=self.settings.get('images_folder', ''))
        ctk.CTkEntry(folder_entry_frame, textvariable=folder_var, width=400).pack(side='left', expand=True, fill='x')
        
        ctk.CTkButton(folder_entry_frame, text="Browse", command=lambda: folder_var.set(filedialog.askdirectory() or folder_var.get()), width=80).pack(side='left', padx=(10, 0))
        
        progress_frame = ctk.CTkFrame(indexer_window)
        progress_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        progress_label = ctk.CTkLabel(progress_frame, text="Ready", font=ctk.CTkFont(size=12))
        progress_label.pack(pady=20)
        
        progress_bar = ctk.CTkProgressBar(progress_frame, width=500)
        progress_bar.pack(pady=10)
        progress_bar.set(0)
        
        details_label = ctk.CTkLabel(progress_frame, text="", font=ctk.CTkFont(size=11))
        details_label.pack()
        
        def start_indexing():
            folder = folder_var.get()
            if not folder or not os.path.exists(folder):
                messagebox.showerror("Error", "Please select a valid folder")
                return
            
            self.settings['images_folder'] = folder
            self.save_settings()
            
            threading.Thread(target=self._index_thread, args=(folder, progress_label, progress_bar, details_label, indexer_window.destroy), daemon=True).start()
        
        ctk.CTkButton(indexer_window, text="Start Indexing", command=start_indexing, width=200, height=40).pack(pady=20)
    
    def _index_thread(self, folder:str, progress_label:ctk.CTkLabel, progress_bar:ctk.CTkProgressBar, details_label:ctk.CTkLabel, close_callback:Any) -> None:
        """Background thread for indexing images from a folder"""

        try:
            self.root.after(0, lambda: progress_label.configure(text="Finding images..."))
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            image_files = [str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in image_extensions]
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showerror("Error", "No images found"))
                return
            
            new_image_files = [f for f in image_files if f not in self.image_paths_set]
            
            if not new_image_files:
                self.root.after(0, lambda: messagebox.showinfo("Info", "All images already indexed!"))
                self.root.after(0, close_callback)
                return
            
            total = len(new_image_files)
            self.root.after(0, lambda: details_label.configure(text=f"Found {total} new images to index"))
            
            if not self.model:
                self.root.after(0, lambda: progress_label.configure(text="Loading model..."))
                SentenceTransformer = lazy_import_sentence_transformers()
                self.model = SentenceTransformer(self.settings['clip_model'])
            
            batch_size = 32
            
            # Pre-allocate numpy array for embeddings (avoid list + conversion)
            embedding_dim = 512
            all_embeddings = np.zeros((total, embedding_dim), dtype=np.float32)
            valid_paths = []
            valid_count = 0
            
            self.root.after(0, lambda: progress_label.configure(text="Processing..."))
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                chunk_size = 256
                
                for chunk_start in range(0, total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total)
                    chunk_files = new_image_files[chunk_start:chunk_end]
                    
                    futures = {executor.submit(self._load_for_index, path): path for path in chunk_files}
                    
                    batch_images = []
                    batch_paths = []
                    
                    for future, path in futures.items():
                        try:
                            img = future.result()
                            if img is not None:
                                batch_images.append(img)
                                batch_paths.append(path)
                        except:
                            pass
                        
                        # Encode when batch is full
                        if len(batch_images) >= batch_size:
                            embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                            
                            batch_len = len(embeddings)
                            all_embeddings[valid_count:valid_count + batch_len] = embeddings
                            valid_paths.extend(batch_paths)
                            valid_count += batch_len
                            
                            batch_images = []
                            batch_paths = []
                            
                            progress = valid_count / total
                            self.root.after(0, lambda p=progress: progress_bar.set(p))
                            self.root.after(0, lambda c=valid_count, t=total: 
                                           details_label.configure(text=f"Processed {c}/{t}"))
                    
                    # Process remaining images in chunk
                    if batch_images:
                        embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                        
                        batch_len = len(embeddings)
                        all_embeddings[valid_count:valid_count + batch_len] = embeddings
                        valid_paths.extend(batch_paths)
                        valid_count += batch_len
                        
                        progress = valid_count / total
                        self.root.after(0, lambda p=progress: progress_bar.set(p))
                        self.root.after(0, lambda c=valid_count, t=total: 
                                       details_label.configure(text=f"Processed {c}/{t}"))
            
            all_embeddings = all_embeddings[:valid_count]
            
            # Generate thumbnails
            self.root.after(0, lambda: progress_label.configure(text="Generating thumbnails..."))
            thumb_folder = os.path.join(self.settings['index_folder'], 'thumbnails')
            os.makedirs(thumb_folder, exist_ok=True)
            cache_size = self.settings.get('cached_thumbnail_size', 300)
            user_batch_size = self.settings.get('processing_batch_size', 10)
            
            # Create event loop once for all thumbnail generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._generate_thumbnails_batch_async(valid_paths, thumb_folder, cache_size, user_batch_size)
                )
            finally:
                loop.close()
            
            self.root.after(0, lambda: progress_label.configure(text="Saving..."))
            
            output = self.settings['index_folder']
            os.makedirs(output, exist_ok=True)
            
            if self.embeddings is not None and len(self.embeddings) > 0:
                combined_embeddings = np.concatenate([self.embeddings, all_embeddings], axis=0)
            else:
                combined_embeddings = all_embeddings
            
            combined_paths = self.image_paths + valid_paths
            
            with open(os.path.join(output, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(combined_embeddings, f)
            
            save_json(os.path.join(output, 'image_paths.json'), combined_paths)
            
            save_json(os.path.join(output, 'metadata.json'), {
                'total_images': len(combined_paths),
                'model': self.settings['clip_model'],
                'indexed_folder': folder,
                'indexed_at': time.time()
            })
            
            self.load_index()
            self.search_cache.clear()
            self.image_loader.clear()
            
            msg = f'Indexing complete! Added {len(valid_paths)} new images. Total: {len(combined_paths)}'
            
            try:
                notification.notify(
                    title='Scout',
                    message=msg,
                    app_icon='Scout.ico' if os.path.exists('Scout.ico') else None,
                    timeout=10
                )
            except:
                pass
            
            self.root.after(0, lambda: progress_label.configure(text=msg))
            self.root.after(0, lambda: messagebox.showinfo("Success", msg))
            self.root.after(0, close_callback)
            
        except Exception as ex:
            self.root.after(0, lambda ex=ex: messagebox.showerror("Error", str(ex)))
    
    def _load_for_index(self, path:str) -> None:
        """Load and preprocess image for indexing"""

        try:
            img = Image.open(path).convert('RGB')
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            return img
        except:
            return None
    
    def refresh_all_folders(self) -> None:
        """Refresh all indexed folders - check for new/deleted images"""

        if not self.image_paths:
            messagebox.showinfo("No Index", "No folders indexed yet!")
            return
        
        # Group by common root folders
        all_folders = set()
        for path in self.image_paths:
            folder = os.path.dirname(path)
            all_folders.add(folder)
        
        # Find root folders 
        root_folders = set()
        for folder in all_folders:
            is_subfolder = False
            for other in all_folders:
                if folder != other and folder.startswith(other + os.sep):
                    is_subfolder = True
                    break
            if not is_subfolder:
                root_folders.add(folder)
        
        # Group images by root folder
        indexed_folders = {}
        for root_folder in root_folders:
            indexed_folders[root_folder] = [
                p for p in self.image_paths 
                if p.startswith(root_folder + os.sep) or os.path.dirname(p) == root_folder
            ]
        
        msg = f"Scan {len(indexed_folders)} top-level folder(s) for changes?"
        if not messagebox.askyesno("Refresh All Folders", msg):
            return
        
        self.status_label.configure(text="Scanning folders for changes...")
        threading.Thread(target=self._refresh_all_thread, args=(indexed_folders,), daemon=True).start()
    
    def _refresh_all_thread(self, indexed_folders) -> None:
        """Background thread to scan all folders"""

        try:
            self.root.after(0, lambda: self.progress_bar.pack(side='left', padx=15))
            self.root.after(0, lambda: self.progress_bar.set(0))
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            
            # Scan for new and deleted files
            new_files = []
            deleted_files = []
            
            total_folders = len(indexed_folders)
            processed = 0
            
            for folder, existing_files in indexed_folders.items():
                # Update progress
                progress = processed / total_folders
                self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                self.root.after(0, lambda f=os.path.basename(folder): 
                               self.status_label.configure(text=f"Scanning {f}..."))
                
                if not os.path.exists(folder):
                    # Entire folder deleted
                    deleted_files.extend(existing_files)
                    processed += 1
                    continue
                
                # Get current files in folder (this is slow for large folders)
                current_files = {str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in image_extensions}
                existing_set = set(existing_files)
                
                # Find new and deleted
                new_in_folder = current_files - existing_set
                deleted_in_folder = existing_set - current_files
                
                new_files.extend(list(new_in_folder))
                deleted_files.extend(list(deleted_in_folder))
                
                processed += 1
            
            # Hide progress bar
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            
            # Show results
            if not new_files and not deleted_files:
                self.root.after(0, lambda: self.status_label.configure(text="No changes found"))
                self.root.after(0, lambda: messagebox.showinfo("Up to Date", "All folders are up to date!"))
                return
            
            msg = f"Found changes:\n\n"
            if new_files:
                msg += f"➕ {len(new_files)} new image(s)\n"
            if deleted_files:
                msg += f"➖ {len(deleted_files)} deleted image(s)\n"
            msg += f"\nUpdate index?"
            
            self.root.after(0, lambda: self._show_refresh_dialog(msg, new_files, deleted_files))
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
    
    def _show_refresh_dialog(self, msg:str, new_files:list, deleted_files:list) -> None:
        """Show dialog with detailed changes"""

        # Create detailed window
        changes_window = ctk.CTkToplevel(self.root)
        if os.path.exists("Scout.ico"):
            changes_window.iconbitmap("Scout.ico")
        changes_window.title("Index Changes Found")
        changes_window.geometry("700x600")
        changes_window.grab_set()
        
        ctk.CTkLabel(changes_window, text="Index Changes", 
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        summary_text = ""
        if new_files:
            summary_text += f"➕ {len(new_files)} new image(s)\n"
        if deleted_files:
            summary_text += f"➖ {len(deleted_files)} deleted image(s)"
        
        ctk.CTkLabel(changes_window, text=summary_text, 
                     font=ctk.CTkFont(size=14)).pack(pady=10)
        
        # Tabview for new/deleted
        tabview = ctk.CTkTabview(changes_window)
        tabview.pack(fill='both', expand=True, padx=20, pady=10)
        
        if new_files:
            tab_new = tabview.add("New Images")
            scroll_new = ctk.CTkScrollableFrame(tab_new)
            scroll_new.pack(fill='both', expand=True, padx=10, pady=10)
            
            for filepath in new_files[:200]:
                filename = os.path.basename(filepath)
                
                frame = ctk.CTkFrame(scroll_new, fg_color='transparent')
                frame.pack(fill='x', pady=3)
                
                # Try to load thumbnail
                try:
                    img = Image.open(filepath)
                    img.thumbnail((60, 60), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    thumb_label = tk.Label(frame, image=photo, bg='#2b2b2b')
                    thumb_label.image = photo
                    thumb_label.pack(side='left', padx=5)
                except:
                    # Fallback: Show placeholder
                    placeholder = ctk.CTkFrame(frame, width=60, height=60, fg_color='gray30')
                    placeholder.pack(side='left', padx=5)
                    placeholder.pack_propagate(False)
                
                ctk.CTkLabel(frame, text=filename, font=ctk.CTkFont(size=11), 
                            anchor='w').pack(side='left', fill='x', expand=True, padx=10)
            
            if len(new_files) > 200:
                ctk.CTkLabel(scroll_new, text=f"... and {len(new_files) - 200} more", 
                            font=ctk.CTkFont(size=10), text_color='gray').pack(pady=10)
        
        if deleted_files:
            tab_del = tabview.add("Deleted Images")
            scroll_del = ctk.CTkScrollableFrame(tab_del)
            scroll_del.pack(fill='both', expand=True, padx=10, pady=10)
            
            for filepath in deleted_files[:200]:
                filename = os.path.basename(filepath)
                
                frame = ctk.CTkFrame(scroll_del, fg_color='transparent')
                frame.pack(fill='x', pady=3)
                
                # Deleted files don't exist
                placeholder = ctk.CTkFrame(frame, width=60, height=60, fg_color='gray20')
                placeholder.pack(side='left', padx=5)
                placeholder.pack_propagate(False)
                ctk.CTkLabel(placeholder, text="❌", font=ctk.CTkFont(size=24)).pack(expand=True)
                
                ctk.CTkLabel(frame, text=filename, font=ctk.CTkFont(size=11), 
                            anchor='w').pack(side='left', fill='x', expand=True, padx=10)
            
            if len(deleted_files) > 200:
                ctk.CTkLabel(scroll_del, text=f"... and {len(deleted_files) - 200} more", 
                            font=ctk.CTkFont(size=10), text_color='gray').pack(pady=10)
        
        # Buttons
        btn_frame = ctk.CTkFrame(changes_window, fg_color='transparent')
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="Update Index", width=120, height=35,
                     command=lambda: [changes_window.destroy(), 
                                     threading.Thread(target=self._apply_refresh_changes, 
                                                     args=(new_files, deleted_files), daemon=True).start()]).pack(side='left', padx=10)
        
        ctk.CTkButton(btn_frame, text="Cancel", width=100, height=35,
                     command=lambda: [changes_window.destroy(), 
                                     self.status_label.configure(text="Refresh cancelled")]).pack(side='left', padx=10)
    
    def _apply_refresh_changes(self, new_files:list, deleted_files:list) -> None:
        """Apply the refresh changes"""

        try:
            # Load manually deleted images
            deleted_by_user_file = os.path.join(self.settings['index_folder'], 'deleted_by_user.json')
            deleted_by_user = set()
            if os.path.exists(deleted_by_user_file):
                deleted_by_user = set(load_json(deleted_by_user_file))
            
            # Filter out images user manually deleted
            new_files = [f for f in new_files if f not in deleted_by_user]
            
            if deleted_files:
                indices_to_remove = []
                for img_path in deleted_files:
                    try:
                        idx = self.image_paths.index(img_path)
                        indices_to_remove.append(idx)
                    except ValueError:
                        pass
                
                if indices_to_remove:
                    # Use boolean mask
                    mask = np.ones(len(self.image_paths), dtype=bool)
                    mask[indices_to_remove] = False
                    self.embeddings = self.embeddings[mask]
                    
                    # Remove paths in reverse order
                    for idx in sorted(indices_to_remove, reverse=True):
                        self.image_paths.pop(idx)
            
            # Add new files
            added_count = 0
            if new_files and self.model:
                SentenceTransformer = lazy_import_sentence_transformers()
                if not self.model:
                    self.model = SentenceTransformer(self.settings['clip_model'])
                
                batch_images = []
                valid_paths = []
                
                for path in new_files:
                    try:
                        img = self._load_for_index(path)
                        if img:
                            batch_images.append(img)
                            valid_paths.append(path)
                    except:
                        pass
                
                if batch_images:
                    new_embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=32)
                    
                    # Cache thumbnails for new images
                    thumb_folder = os.path.join(self.settings['index_folder'], 'thumbnails')
                    os.makedirs(thumb_folder, exist_ok=True)
                    
                    for path in valid_paths:
                        try:
                            original_img = Image.open(path)
                            if original_img.mode == 'RGBA':
                                background = Image.new('RGB', original_img.size, '#3a3a3a')
                                background.paste(original_img, mask=original_img.split()[3])
                                original_img = background
                            elif original_img.mode != 'RGB':
                                original_img = original_img.convert('RGB')
                            
                            thumb = original_img.copy()
                            cache_size = self.settings.get("cached_thumbnail_size", 300); thumb.thumbnail((cache_size, cache_size), Image.Resampling.LANCZOS)
                            thumb_name = str(abs(hash(path)) % 10**10) + '.jpg'
                            thumb_path = os.path.join(thumb_folder, thumb_name)
                            thumb.save(thumb_path, 'JPEG', quality=85, optimize=True)
                        except:
                            pass
                    
                    if self.embeddings is not None and len(self.embeddings) > 0:
                        self.embeddings = np.vstack([self.embeddings, np.array(new_embeddings)])
                    else:
                        self.embeddings = np.array(new_embeddings)
                    
                    self.image_paths.extend(valid_paths)
                    added_count = len(valid_paths)
            
            # Save updated index
            output = self.settings['index_folder']
            os.makedirs(output, exist_ok=True)
            
            with open(os.path.join(output, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            save_json(os.path.join(output, 'image_paths.json'), self.image_paths)
            
            save_json(os.path.join(output, 'metadata.json'), {
                'total_images': len(self.image_paths),
                'model': self.settings['clip_model'],
                'indexed_folder': 'Multiple folders',
                'indexed_at': time.time()
            })
            
            self.search_cache.clear()
            
            msg = f"Index Updated!\n"
            if added_count:
                msg += f"Added: {added_count}\n"
            if deleted_files:
                msg += f"Removed: {len(deleted_files)}\n"
            if len(new_files) > added_count:
                skipped = len(new_files) - added_count
                msg += f"Skipped: {skipped} (previously deleted by user)"
            
            notification.notify(
                title='Scout',
                message=f'Refresh complete! Total: {len(self.image_paths)} images',
                app_icon='Scout.ico' if os.path.exists('Scout.ico') else None,
                timeout=5
            )
            
            self.root.after(0, lambda: self.status_label.configure(text=f"Index updated: {len(self.image_paths)} total images"))
            self.root.after(0, lambda: messagebox.showinfo("Complete", msg))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Update failed: {e}"))
    
    def open_folder_manager(self) -> None:
        """Open folder management window with incremental loading"""

        if not self.image_paths:
            messagebox.showinfo("No Index", "No folders indexed yet!")
            return
        
        # Get folder counts from cache
        folder_counts = self.metadata.get('folder_counts', {})
        if not folder_counts:
            for path in self.image_paths:
                folder = os.path.dirname(path)
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
        
        sorted_folders = sorted(folder_counts.items())
        
        # Create window
        manager_window = ctk.CTkToplevel(self.root)
        if os.path.exists("Scout.ico"):
            manager_window.iconbitmap("Scout.ico")
        manager_window.title("Manage Indexed Folders")
        manager_window.geometry("800x600")
        manager_window.grab_set()
        
        ctk.CTkLabel(manager_window, text="Indexed Folders", 
                     font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        ctk.CTkLabel(manager_window, text=f"Total: {len(folder_counts)} folders, {len(self.image_paths)} images", 
                     font=ctk.CTkFont(size=12)).pack(pady=5)
        
        # Scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(manager_window)
        scroll_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Incremental loading state
        loaded_count = [0] 
        is_loading = [False]
        LOAD_BATCH = 10 
        
        def load_more_folders() -> None:
            """Load next batch of folders"""

            if is_loading[0]:
                return
            
            is_loading[0] = True
            start = loaded_count[0]
            end = min(start + LOAD_BATCH, len(sorted_folders))
            
            for folder, count in sorted_folders[start:end]:
                folder_frame = ctk.CTkFrame(scroll_frame)
                folder_frame.pack(fill='x', pady=5, padx=10)
                
                # Folder info
                info_frame = ctk.CTkFrame(folder_frame, fg_color='transparent')
                info_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
                
                folder_name = folder if len(folder) < 50 else "..." + folder[-47:]
                ctk.CTkLabel(info_frame, text=folder_name, font=ctk.CTkFont(size=11), 
                            anchor='w').pack(anchor='w')
                ctk.CTkLabel(info_frame, text=f"{count} images", font=ctk.CTkFont(size=10), 
                            text_color='gray', anchor='w').pack(anchor='w')
                
                # Action buttons
                btn_frame = ctk.CTkFrame(folder_frame, fg_color='transparent')
                btn_frame.pack(side='right', padx=10)
                
                ctk.CTkButton(btn_frame, text="🔄 Refresh", width=80, height=30,
                             command=lambda f=folder: self.refresh_single_folder(f, manager_window)).pack(side='left', padx=5)
                ctk.CTkButton(btn_frame, text="🗑 Delete", width=80, height=30,
                             command=lambda f=folder: self.delete_folder_images(f, manager_window)).pack(side='left', padx=5)
            
            loaded_count[0] = end
            is_loading[0] = False
            
            # Show loading indicator if more to load
            if loaded_count[0] < len(sorted_folders):
                remaining = len(sorted_folders) - loaded_count[0]
                status = ctk.CTkLabel(scroll_frame, text=f"↓ Scroll down to load {remaining} more folders ↓", 
                                     font=ctk.CTkFont(size=11), text_color='gray')
                status.pack(pady=20)
        
        def on_scroll_check() -> None:
            """Check if user scrolled near bottom"""

            try:
                # Get scroll position
                canvas = scroll_frame._parent_canvas
                scroll_pos = canvas.yview()[1]
                
                # Load more when scrolled to 80%
                if scroll_pos > 0.8 and loaded_count[0] < len(sorted_folders):
                    # Remove "scroll down" label if exists
                    for widget in scroll_frame.winfo_children():
                        if isinstance(widget, ctk.CTkLabel) and "Scroll down" in str(widget.cget("text")):
                            widget.destroy()
                    
                    load_more_folders()
            except:
                pass
            
            # Check again in 200ms
            if manager_window.winfo_exists():
                manager_window.after(200, on_scroll_check)
        
        # Load initial batch
        load_more_folders()
        
        # Start scroll checking
        on_scroll_check()
        
        ctk.CTkButton(manager_window, text="Close", command=manager_window.destroy, 
                     width=100, height=35).pack(pady=20)
    
    def refresh_single_folder(self, folder:str, parent_window:ctk.CTkToplevel) -> None:
        """Refresh a single folder"""

        parent_window.destroy()
        self.status_label.configure(text=f"Scanning {os.path.basename(folder)}...")
        
        # Get existing files in this folder
        existing_files = [p for p in self.image_paths if os.path.dirname(p) == folder]
        indexed_folders = {folder: existing_files}
        
        threading.Thread(target=self._refresh_all_thread, args=(indexed_folders,), daemon=True).start()
    
    def delete_folder_images(self, folder:str, parent_window:ctk.CTkToplevel) -> None:
        """Delete all images from a specific folder"""

        folder_images = [p for p in self.image_paths if os.path.dirname(p) == folder]
        count = len(folder_images)
        
        if not messagebox.askyesno("Confirm Delete", 
                                   f"Remove all {count} images from this folder?\n\n{folder}\n\n(Files will NOT be deleted from disk)"):
            return
        
        parent_window.destroy()
        
        for img_path in folder_images:
            try:
                idx = self.image_paths.index(img_path)
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
                self.image_paths.pop(idx)
            except ValueError:
                pass
        
        self._save_index()
        self.search_cache.clear()
        
        self.status_label.configure(text=f"Removed {count} images from {os.path.basename(folder)}")
        messagebox.showinfo("Complete", f"Removed {count} images!\nRemaining: {len(self.image_paths)} images")
    
    def open_settings(self) -> None:
        """Open settings window"""

        settings_window = ctk.CTkToplevel(self.root)
        if os.path.exists("Scout.ico"):
            settings_window.iconbitmap("Scout.ico")
        settings_window.title("Settings")
        settings_window.geometry("500x600")
        settings_window.grab_set()
        
        ctk.CTkLabel(settings_window, text="Settings", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        settings_frame = ctk.CTkScrollableFrame(settings_window)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(settings_frame, text="CLIP Model:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor='w', pady=(10, 5))
        
        model_var = tk.StringVar(value=self.settings['clip_model'])
        
        model_info = {
            'clip-ViT-B-32': 'Fast - Good quality, fastest indexing (~5-6 hrs for 100K images)',
            'clip-ViT-B-16': 'Balanced - Better quality, moderate speed (~7-8 hrs for 100K)',
            'clip-ViT-L-14': 'Best - Highest quality, slowest indexing (~12-15 hrs for 100K)'
        }
        
        for model in ['clip-ViT-B-32', 'clip-ViT-B-16', 'clip-ViT-L-14']:
            ctk.CTkRadioButton(settings_frame, text=model, variable=model_var, value=model).pack(anchor='w', padx=20, pady=2)
            ctk.CTkLabel(settings_frame, text=model_info[model], 
                        font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=40)
        
        # Warning frame
        warning_frame = ctk.CTkFrame(settings_frame, fg_color=('#FFE6E6', '#4A2020'), corner_radius=8)
        warning_frame.pack(anchor='w', padx=20, pady=(10, 5), fill='x')
        ctk.CTkLabel(warning_frame, text="⚠️ Changing Model Requires Reindexing", 
                    font=ctk.CTkFont(size=11, weight="bold"), text_color=('#CC0000', '#FF6666')).pack(anchor='w', padx=10, pady=(8, 2))
        ctk.CTkLabel(warning_frame, text="• All images must be reindexed with the new model", 
                    font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=10)
        ctk.CTkLabel(warning_frame, text="• Thumbnails will be reused (no need to regenerate)", 
                    font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=10)
        ctk.CTkLabel(warning_frame, text="• Indexing time depends on model choice (see above)", 
                    font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=10, pady=(0, 8))
        
        ctk.CTkLabel(settings_frame, text="Search Settings:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor='w', pady=(20, 5))
        
        dynamic_var = tk.BooleanVar(value=self.settings['dynamic_search'])
        ctk.CTkCheckBox(settings_frame, text="Dynamic search", variable=dynamic_var).pack(anchor='w', padx=20)
        
        ctk.CTkLabel(settings_frame, text="Delay (ms):", font=ctk.CTkFont(size=12)).pack(anchor='w', padx=20, pady=(10, 5))
        delay_var = tk.StringVar(value=str(self.settings['dynamic_delay']))
        ctk.CTkEntry(settings_frame, textvariable=delay_var, width=100).pack(anchor='w', padx=20)
        
        ctk.CTkLabel(settings_frame, text="Display:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor='w', pady=(20, 5))
        
        ctk.CTkLabel(settings_frame, text="Thumbnail size (storage):", font=ctk.CTkFont(size=12)).pack(anchor='w', padx=20, pady=(10, 5))
        cached_size_var = tk.StringVar(value=str(self.settings.get('cached_thumbnail_size', 300)))
        ctk.CTkEntry(settings_frame, textvariable=cached_size_var, width=100).pack(anchor='w', padx=20)
        ctk.CTkLabel(settings_frame, text="Smaller = less storage, faster load (e.g. 200, 250, 300)", 
                     font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=20)
        
        ctk.CTkLabel(settings_frame, text="Thumbnail format:", font=ctk.CTkFont(size=12)).pack(anchor='w', padx=20, pady=(10, 5))
        format_var = tk.StringVar(value=self.settings.get('thumbnail_format', 'webp'))
        format_frame = ctk.CTkFrame(settings_frame, fg_color='transparent')
        format_frame.pack(anchor='w', padx=20)
        ctk.CTkRadioButton(format_frame, text="WebP (uses less storage)", variable=format_var, value='webp').pack(side='left', padx=(0, 10))
        ctk.CTkRadioButton(format_frame, text="JPEG (faster)", variable=format_var, value='jpeg').pack(side='left')
        ctk.CTkLabel(settings_frame, text="WebP: ~30% less storage space, JPEG: slightly faster loading", 
                     font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=20)
        
        def regenerate_with_settings() -> None:
            """Regenerate all thumbnails with new settings"""

            # Save current settings including the size
            self.settings['cached_thumbnail_size'] = int(cached_size_var.get())
            self.settings['thumbnail_format'] = format_var.get()
            self.save_settings()
            settings_window.destroy()
            self.regenerate_thumbnails(int(cached_size_var.get()))
        
        ctk.CTkButton(settings_frame, text="🔄 Regenerate All Thumbnails", 
                     command=regenerate_with_settings, 
                     width=200).pack(anchor='w', padx=20, pady=(10, 5))
        ctk.CTkLabel(settings_frame, text="Applies new size and format to all thumbnails", 
                     font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=20)
        
        ctk.CTkLabel(settings_frame, text="Hover preview size:", font=ctk.CTkFont(size=12)).pack(anchor='w', padx=20, pady=(10, 5))
        hover_size_var = tk.StringVar(value=str(self.settings.get('hover_preview_size', 800)))
        ctk.CTkEntry(settings_frame, textvariable=hover_size_var, width=100).pack(anchor='w', padx=20)
        
        ctk.CTkLabel(settings_frame, text="Performance:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor='w', pady=(20, 5))
        
        ctk.CTkLabel(settings_frame, text="Parallel processing batch size:", font=ctk.CTkFont(size=12)).pack(anchor='w', padx=20, pady=(10, 5))
        batch_size_var = tk.StringVar(value=str(self.settings.get('processing_batch_size', 10)))
        ctk.CTkEntry(settings_frame, textvariable=batch_size_var, width=100).pack(anchor='w', padx=20)
        ctk.CTkLabel(settings_frame, text="Higher = faster indexing and thumbnail generation (Try 32 for SSD's with 16GB RAM)", 
                     font=ctk.CTkFont(size=10), text_color='gray').pack(anchor='w', padx=20)
        
        ctk.CTkButton(settings_frame, text="Clear Cache", command=lambda: (self.search_cache.clear(), self.image_loader.clear(), messagebox.showinfo("Done", "Cache cleared")), width=150).pack(anchor='w', padx=20, pady=20)
        
        def save() -> None:
            """Save settings and handle model change"""

            old_model = self.settings.get('clip_model')
            new_model = model_var.get()
            model_changed = old_model != new_model
            
            if model_changed and len(self.image_paths) > 0:
                reindex_dialog = ctk.CTkToplevel(settings_window)
                reindex_dialog.title("Model Changed - Reindex Required")
                reindex_dialog.geometry("500x350")
                reindex_dialog.grab_set()
                reindex_dialog.transient(settings_window)
                
                # Center the dialog
                reindex_dialog.update_idletasks()
                x = settings_window.winfo_x() + (settings_window.winfo_width() - 500) // 2
                y = settings_window.winfo_y() + (settings_window.winfo_height() - 350) // 2
                reindex_dialog.geometry(f"+{x}+{y}")
                
                user_choice = tk.StringVar(value="cancel")
                
                # Warning icon and title
                title_frame = ctk.CTkFrame(reindex_dialog, fg_color="transparent")
                title_frame.pack(pady=20)
                ctk.CTkLabel(title_frame, text="⚠️", font=ctk.CTkFont(size=40)).pack()
                ctk.CTkLabel(title_frame, text="Model Changed - Reindex Required", 
                            font=ctk.CTkFont(size=16, weight="bold")).pack()
                
                # Message
                msg_frame = ctk.CTkFrame(reindex_dialog, fg_color="transparent")
                msg_frame.pack(pady=10, padx=30, fill='both', expand=True)
                
                ctk.CTkLabel(msg_frame, text=f"Changed: {old_model} → {new_model}", 
                            font=ctk.CTkFont(size=12)).pack(pady=5)
                ctk.CTkLabel(msg_frame, text=f"Images to reindex: {len(self.image_paths):,}", 
                            font=ctk.CTkFont(size=12)).pack(pady=5)
                ctk.CTkLabel(msg_frame, text="✓ Thumbnails will be reused (faster!)", 
                            font=ctk.CTkFont(size=11), text_color='gray').pack(pady=5)
                ctk.CTkLabel(msg_frame, text="All images will be automatically reindexed.", 
                            font=ctk.CTkFont(size=11), text_color='gray').pack(pady=5)
                
                # Buttons
                btn_frame = ctk.CTkFrame(reindex_dialog, fg_color="transparent")
                btn_frame.pack(pady=20)
                
                def on_cancel():
                    user_choice.set("cancel")
                    reindex_dialog.destroy()
                
                def on_confirm():
                    user_choice.set("confirm")
                    reindex_dialog.destroy()
                
                ctk.CTkButton(btn_frame, text="Cancel", command=on_cancel, 
                             width=150, fg_color="gray", height=40).pack(side='left', padx=10)
                ctk.CTkButton(btn_frame, text="Save & Reindex", command=on_confirm, 
                             width=150, fg_color="green", height=40).pack(side='left', padx=10)
                
                # Wait for user choice
                reindex_dialog.wait_window()
                
                if user_choice.get() == "cancel":
                    return  # User cancelled
            
            # Save settings
            old_cached_size = self.settings.get('cached_thumbnail_size', 300)
            new_cached_size = int(cached_size_var.get())
            
            self.settings['clip_model'] = new_model
            self.settings['dynamic_search'] = dynamic_var.get()
            self.settings['dynamic_delay'] = int(delay_var.get())
            self.settings['cached_thumbnail_size'] = new_cached_size
            self.settings['hover_preview_size'] = int(hover_size_var.get())
            self.settings['processing_batch_size'] = int(batch_size_var.get())
            self.settings['thumbnail_format'] = format_var.get()
            self.save_settings()
            
            # Update image loader with new cached size
            self.image_loader = ImageLoader(self.settings.get('cached_thumbnail_size', 300), 
                                           os.path.join(self.settings['index_folder'], 'thumbnails'))
            
            settings_window.destroy()
            
            # Start reindexing if model changed
            if model_changed and len(self.image_paths) > 0:
                self.status_label.configure(text=f"Model changed to {new_model} - Starting reindex...")
                # Reindex all existing images with new model
                threading.Thread(target=self._reindex_all_with_new_model, daemon=True).start()
            elif old_cached_size != new_cached_size:
                self.status_label.configure(text=f"Thumbnail size changed to {new_cached_size}x{new_cached_size} - click Regenerate to apply")
            else:
                self.status_label.configure(text="Settings saved")
        
        ctk.CTkButton(settings_window, text="Save", command=save, width=200, height=40).pack(pady=20)
    
    def _reindex_all_with_new_model(self) -> None:
        """Reindex all existing images with the new model"""

        try:
            # Show progress bar
            self.root.after(0, lambda: self.progress_bar.pack(side='left', padx=15))
            self.root.after(0, lambda: self.progress_bar.set(0))
            
            all_image_paths = self.image_paths.copy()
            total = len(all_image_paths)
            
            SentenceTransformer = lazy_import_sentence_transformers()
            
            # Load the new model
            self.root.after(0, lambda: self.status_label.configure(text=f"Loading {self.settings['clip_model']}..."))
            self.model = SentenceTransformer(self.settings['clip_model'])
            
            batch_size = 32
            embedding_dim = 512
            all_embeddings = np.zeros((total, embedding_dim), dtype=np.float32)
            valid_paths = []
            valid_count = 0
            
            self.root.after(0, lambda: self.status_label.configure(text="Reindexing with new model..."))
            
            # Process in chunks
            with ThreadPoolExecutor(max_workers=8) as executor:
                chunk_size = 256
                
                for chunk_start in range(0, total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total)
                    chunk_files = all_image_paths[chunk_start:chunk_end]
                    
                    futures = {executor.submit(self._load_for_index, path): path for path in chunk_files}
                    
                    batch_images = []
                    batch_paths = []
                    
                    for future, path in futures.items():
                        try:
                            img = future.result()
                            if img is not None:
                                batch_images.append(img)
                                batch_paths.append(path)
                        except:
                            pass
                        
                        if len(batch_images) >= batch_size:
                            embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                            
                            batch_len = len(embeddings)
                            all_embeddings[valid_count:valid_count + batch_len] = embeddings
                            valid_paths.extend(batch_paths)
                            valid_count += batch_len
                            
                            batch_images = []
                            batch_paths = []
                            
                            progress = valid_count / total
                            self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                            self.root.after(0, lambda c=valid_count, t=total: 
                                          self.status_label.configure(text=f"Reindexed {c}/{t} images"))
                    
                    if batch_images:
                        embeddings = self.model.encode(batch_images, show_progress_bar=False, batch_size=batch_size)
                        
                        batch_len = len(embeddings)
                        all_embeddings[valid_count:valid_count + batch_len] = embeddings
                        valid_paths.extend(batch_paths)
                        valid_count += batch_len
                        
                        progress = valid_count / total
                        self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                        self.root.after(0, lambda c=valid_count, t=total: 
                                      self.status_label.configure(text=f"Reindexed {c}/{t} images"))
            
            all_embeddings = all_embeddings[:valid_count]
            
            output = self.settings['index_folder']
            
            with open(os.path.join(output, 'embeddings.pkl'), 'wb') as f:
                pickle.dump(all_embeddings, f)
            
            save_json(os.path.join(output, 'image_paths.json'), valid_paths)
            
            folder_counts = {}
            for path in valid_paths:
                folder = os.path.dirname(path)
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            save_json(os.path.join(output, 'metadata.json'), {
                'total_images': len(valid_paths),
                'model': self.settings['clip_model'],
                'indexed_folder': 'Reindexed',
                'indexed_at': time.time(),
                'folder_counts': folder_counts
            })
            
            # Reload index
            self.load_index()
            self.search_cache.clear()
            
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(
                text=f"✓ Reindexing complete! {valid_count} images with {self.settings['clip_model']}"))
            
            notification.notify(
                title='Scout',
                message=f'Reindexing complete! {valid_count} images with new model.',
                app_icon='Scout.ico' if os.path.exists('Scout.ico') else None,
                timeout=10
            )
            
        except Exception as e:
            self.root.after(0, lambda: self.progress_bar.pack_forget())
            self.root.after(0, lambda: self.status_label.configure(text=f"Reindex failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Reindexing failed: {e}"))
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ScoutExplorer()
    app.run()