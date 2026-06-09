<script>
  import { invoke } from '@tauri-apps/api/core';
  import { convertFileSrc } from '@tauri-apps/api/core';
  import { open, confirm } from '@tauri-apps/plugin-dialog';
  import { onMount, onDestroy } from 'svelte';
  import { listen } from '@tauri-apps/api/event';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { isPermissionGranted, requestPermission, sendNotification } from '@tauri-apps/plugin-notification';
  import { tick } from 'svelte';

  import Header from '$lib/components/Header.svelte';
  import Gallery from '$lib/components/Gallery.svelte'; 
  import ContextMenu from '$lib/components/ContextMenu.svelte';
  import Lightbox from '$lib/components/Lightbox.svelte';
  import HamburgerMenu from '$lib/components/HamburgerMenu.svelte';
  import SettingsModal from '$lib/components/SettingsModal.svelte';
  import IndexModal from '$lib/components/IndexModal.svelte';
  import DeleteConfirmModal from '$lib/components/DeleteConfirmModal.svelte';
  import DuplicatesModal from '$lib/components/DuplicatesModal.svelte';
  import ManageFoldersModal from '$lib/components/ManageFoldersModal.svelte';
  import CollectionsModal from '$lib/components/CollectionsModal.svelte';

  let query = '';
  let results = [];
  let count = 0;
  let status = 'Ready';
  let indexing = false;
  let searching = false;
  let selectedImage = null;
  let debounceTimer = null;
  let appDragOver = false;
  let showHamburger = false;
  let selectedPaths = new Set();
  let lastClickedIndex = null;
  let feedbackFlash = new Map();
  let pendingDeletePaths = null;
  let deletingPaths = new Set();
  let thumbnailSizeChanged = false;
  let selectAllHandler;
  let isReindexing = false;

  let duplicateGroups = [];
  let showDuplicatesModal = false;

  // Index modal
  let showIndexModal = false;
  let showSettingsModal = false;
  let indexMode = 'folder';
  let indexPath = '';
  let indexRecursive = true;
  let autoIndex = false;
  let showStopIndexingConfirm = false;

  // Lightbox zoom
  let zoomLevel = 1;
  let zoomX = 50;
  let zoomY = 50;

  // Context menu
  let contextMenu = null;
  let showDeleteConfirm = null;
  let pendingDeletePath = null;

  // Progress
  let indexDone = 0;
  let indexTotal = 0;

  let visibleResults = [];
  let renderHandle = null;
  let previousPaths = new Set();

  // Tracks how many results have had thumbnails fetched already
  let thumbLoadedCount = 0;
  const THUMB_PAGE_SIZE = 40;
  let thumbLoading = false;

  async function loadNextThumbnailPage() {
    if (thumbLoading || thumbLoadedCount >= results.length) return;
    thumbLoading = true;

    const batch = results.slice(thumbLoadedCount, thumbLoadedCount + THUMB_PAGE_SIZE);
    const paths = batch.map(([path]) => path);

    try {
      const fetched = await invoke('get_thumbnails', { paths });
      // Merge fetched thumbnails into results in-place
      const thumbMap = new Map(fetched.map(([p, thumb, raw]) => [p, { thumb, raw }]));
      results = results.map(([path, score, ratio, existingThumb, existingRaw]) => {
        const hit = thumbMap.get(path);
        if (hit) return [path, score, ratio, hit.thumb, hit.raw];
        return [path, score, ratio, existingThumb, existingRaw];
      });
      visibleResults = results.slice(0, thumbLoadedCount + THUMB_PAGE_SIZE);
      thumbLoadedCount += batch.length;
    } catch (e) {
      console.error('Failed to load thumbnails:', e);
    }
    thumbLoading = false;
  }

  function handleGalleryScroll(e) {
    const el = e.target;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 400;
    if (nearBottom) loadNextThumbnailPage();
  }

  //manage folders
  let showManageFolders = false;
  let managedFolders = [];
  let foldersLoading = false;

  //collections
  let showCollectionsModal = false;
  let collections = [];
  let imageCollectionsMap = new Map();
  let collectionFlash = new Map();

  function handleSelectAll(e) {
      if ((e.ctrlKey || e.metakey) && e.key === 'a') {
          e.preventDefault();
          console.log(selectedImage);
          if (visibleResults.length > 0) {
              selectedPaths = new Set(visibleResults.map(([path]) => path));
          }
      }
  }

  async function loadImageCollections(paths) {
    try {
      for (const path of paths) {
        const imageCollections_ = await invoke('get_image_collections', { imagePath: path });
        imageCollectionsMap.set(path, new Set(imageCollections_.map(c => c.id)));
      }
      imageCollectionsMap = imageCollectionsMap;
    } catch (e) {
      console.error('Failed to load image collections:', e);
    }
  }

  //estimations
  let indexEstimatedMs = null;
  let indexStartTime = null;
  let indexElapsedMs = 0;
  let updateInterval = null;

  //settings
  let settings = {
    thumbnail_size: '224',
    excluded_file_types: '',
    image_layout: 'grid',
    notify_on_complete: '1',
    prompt_delete_db: '1',
    prompt_delete_system: '1',
    auto_deduplicate: '0',
    deduplicate_mode: 'db',
};

let thumbCache = new Map();

$: if (settings.image_layout) {
    if (settings.image_layout === 'masonry') {
        tick().then(() => applyMasonry());
    } else {
        resetGrid();
    }
}

async function handleModelChange(selectedModel) {
    try {
        await saveSetting('model_family', selectedModel.id);
        await saveSetting('image_model', selectedModel.image_model);
        await saveSetting('text_model', selectedModel.text_model);

        results = [];
        visibleResults = [];
        previousPaths = new Set();
        selectedPaths = new Set();
        query = '';

        try {
            for(const url of thumbCache.values()) URL.revokeObjectURL(url);
        } catch(e) {}
        thumbCache = new Map();

        status = `Model changed to ${selectedModel.name}`;

        if (shouldNotify()) {
          await notify('Scout', { 
            body:  `Model changed to ${selectedModel.name}`,
          });
        }
    } catch (e) {
        console.error("Failed to change model:", e);
        status = "Failed to change model";
    }
}

function formatTime(ms) {
    if (ms === null) return 'Calculating...';
    if (ms < 1000) return `${Math.round(ms / 100) * 100}ms`;
    if (ms < 60000) return `${Math.round(ms / 1000)}s`;
    const minutes = Math.round(ms / 60000);
    return `${minutes}m`;
  }

async function loadFolderStats() {
    foldersLoading = true;
    try {
        managedFolders = await invoke('get_folder_stats');
    } catch (e) {
        console.error("Failed to load folder stats:", e);
    }
    foldersLoading = false;
}

async function openManageFolders() {
    showManageFolders = true;
    await loadFolderStats();
}

async function toggleFolderAutoIndex(path, enabled) {
    try {
        await invoke("set_folder_auto_index", { path, autoIndex: enabled });
        managedFolders = managedFolders.map(f => 
            f.path === path ? { ...f, auto_index: enabled } : f
        );
    } catch (e) { console.error(e); }
}

async function deleteFolderFromSystem(path) {
  try {
    const removed = await invoke('delete_folder_from_system', { path });
    managedFolders = managedFolders.filter(f => f.path !== path);
    
    await getCount();
    await refreshResults();

    status = `Deleted ${removed} file${removed !== 1 ? 's' : ''} from disk`;
  } catch (e) { console.error(e); }
}

async function unindexFolder(path) {
  try {
    const removed = await invoke('unindex_folder', { path });
    managedFolders = managedFolders.filter(f => f.path !== path);
    
    await getCount();
    await refreshResults();

    status = `Removed ${removed} image${removed !== 1 ? 's' : ''} from index`;
  } catch (e) { 
      console.error(e); 
      status = "Failed to unindex folder";
  }
}

async function loadCollections() {
  try {
    collections = await invoke('get_collections');
    // Pre-load collections for all visible images
    for (const [path] of visibleResults) {
      if (!imageCollectionsMap.has(path)) {
        const imageCollections = await invoke('get_image_collections', { imagePath: path });
        imageCollectionsMap.set(path, new Set(imageCollections.map(c => c.id)));
      }
    }
  } catch (e) {
    console.error('Failed to load collections:', e);
  }
}

async function addToCollection(collectionId) {
  if (!contextMenu?.path && selectedPaths.size === 0) return;

  try {
    const paths = selectedPaths.size > 0 ? [...selectedPaths] : [contextMenu.path];
    for (const path of paths) {
      await invoke('add_to_collection', {
        collectionId,
        imagePath: path
      });
      const collections_set = imageCollectionsMap.get(path) || new Set();
      collections_set.add(collectionId);
      imageCollectionsMap.set(path, collections_set);
      imageCollectionsMap = imageCollectionsMap;
      
      // Add flash feedback for this image
      const newMap = new Map(collectionFlash);
      newMap.set(path, 'collection');
      collectionFlash = newMap;
      setTimeout(() => {
        const m = new Map(collectionFlash);
        m.delete(path);
        collectionFlash = m;
      }, 1000);
    }
    status = `Added ${paths.length} image${paths.length !== 1 ? 's' : ''} to collection`;
  } catch (e) {
    console.error('Failed to add to collection:', e);
    status = 'Failed to add to collection';
  }
  contextMenu = null;
}

async function removeFromCollection(collectionId) {
  if (!contextMenu?.path) return;

  try {
    await invoke('remove_from_collection', {
      collectionId,
      imagePath: contextMenu.path
    });
    const collections_set = imageCollectionsMap.get(contextMenu.path) || new Set();
    collections_set.delete(collectionId);
    imageCollectionsMap.set(contextMenu.path, collections_set);
    imageCollectionsMap = imageCollectionsMap;
    status = 'Removed from collection';
  } catch (e) {
    console.error('Failed to remove from collection:', e);
    status = 'Failed to remove from collection';
  }
  contextMenu = null;
}

function renderProgressively(allResults) {
    if (renderHandle) cancelAnimationFrame(renderHandle);

    const BATCH = 30;
    let index = 0;

    if (settings.image_layout === 'masonry') {
        tick().then(() => applyMasonry());
    }

    function nextBatch() {
        visibleResults = allResults.slice(0, index + BATCH);
        index += BATCH;

        if (index < allResults.length) {
            renderHandle = requestAnimationFrame(nextBatch);
        } else {
            renderHandle = null;
        }
    }

    nextBatch();
}

function getThumbUrl(path, bytes) {
    if (thumbCache.has(path)) return thumbCache.get(path);
    if (!bytes || bytes.length === 0) return null;
    const url = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: "image/jpeg" }));
    thumbCache.set(path, url);
    return url;
}

function clearThumbCache() {
    for (const url of thumbCache.values()) URL.revokeObjectURL(url);
    thumbCache = new Map();
}

async function runDuplicateFind() {
    showSettingsModal = false;
    status = 'Finding duplicates...';
    indexing = true;
    try {
        duplicateGroups = await invoke('find_duplicates');
        if (duplicateGroups.length === 0) {
            status = 'No duplicates found';
        } else {
            showDuplicatesModal = true;
            status = `Found ${duplicateGroups.length} duplicate groups`;
        }
    } catch(e) {
        status = 'Failed to find duplicates';
        console.error(e);
    }
    indexing = false;
}


async function runRemoveDuplicates() {
    const mode = settings.deduplicate_mode;
    try {
        const count = await invoke('remove_duplicates', { mode });
        showDuplicatesModal = false;
        await getCount();
        status = `Removed ${count} duplicate${count !== 1 ? 's' : ''}`;
    } catch(e) {
        console.error(e);
    }
}

function resetGrid() {
    const grid = document.querySelector('.grid');
    if (!grid) return;
    grid.style.position = '';
    grid.style.height = '';
    const cards = [...grid.querySelectorAll('.card')];
    cards.forEach(card => {
        card.style.position = '';
        card.style.width = '';
        card.style.left = '';
        card.style.top = '';
        card.style.height = '';
        card.style.aspectRatio = '';
    });
}


async function regenThumbnails() {
    thumbnailSizeChanged = false;
    showSettingsModal = false;
    status = 'Regenerating thumbnails...';
    indexing = true;

    results = [];
    visibleResults = [];
    query = '';
    clearThumbCache();

    try {
        await invoke('regenerate_thumbnails', { size: parseInt(settings.thumbnail_size) });
        status = 'Thumbnails regenerated';
    } catch(e) {
        status = 'Failed';
        console.error(e);
    }
    indexing = false;
}

async function clearDatabase() {
  showSettingsModal = false;
  const confirmed = await confirm(
    'This will delete ALL indexed images, embeddings, thumbnails, and feedback from the database. Files on disk are not affected. This cannot be undone.',
    { title: 'Clear Database', kind: 'warning' }
  );
  if (!confirmed) return;
  try {
    await invoke('clear_database');
    results = [];
    query = '';
    clearThumbCache();
    await getCount();
    status = 'Database cleared';
  } catch(e) {
    status = 'Failed to clear database';
    console.error(e);
  }
}

async function applyMasonry() {
    if (settings.image_layout !== 'masonry') return;
    await tick();
    const grid = document.querySelector('.grid');
    if (!grid) return;
    const cards = [...grid.querySelectorAll('.card')];
    if (!cards.length) return;

    const gap = 12;
    const cols = Math.max(2, Math.floor(grid.offsetWidth / 212));
    const colWidth = (grid.offsetWidth - (cols - 1) * gap) / cols;
    const colHeights = new Array(cols).fill(0);

    grid.style.position = 'relative';

    cards.forEach((card, idx) => {
        const ratio = results[idx]?.[2] || 1;
        const col = colHeights.indexOf(Math.min(...colHeights));
        const x = col * (colWidth + gap);
        const y = colHeights[col];
        const cardHeight = colWidth / ratio;

        card.style.position = 'absolute';
        card.style.width = `${colWidth}px`;
        card.style.height = `${cardHeight}px`;
        card.style.left = `${x}px`;
        card.style.top = `${y}px`;
        card.style.aspectRatio = 'unset';

        colHeights[col] += cardHeight + gap;
    });

    grid.style.height = `${Math.max(...colHeights)}px`;
}

  async function search() {
    if (!query.trim()) 
    { 
        visibleResults = []; 
        results = []; 
        previousPaths = new Set();
        thumbLoadedCount = 0;
        status = 'Ready'; 
        return; 
    }

    searching = true;
    status = 'Searching...';
    try {
        clearThumbCache();
        previousPaths = new Set(visibleResults.map(([p]) => p));
        // search now returns [path, score, ratio] � no thumbnails yet
        const ranked = await invoke('search', { query });
        console.log('[search] ranked count:', ranked?.length, 'sample:', ranked?.[0]);
        // Initialise results with empty thumb/raw slots so Gallery can render placeholders
        results = ranked.map(([path, score, ratio]) => [path, score, ratio, null, null]);
        thumbLoadedCount = 0;
        status = `${results.length} results`;
        // Show first page of cards immediately (no thumbnails yet � convertFileSrc fallback kicks in)
        visibleResults = results.slice(0, THUMB_PAGE_SIZE);
        // Fetch first page of thumbnails right away
        await loadNextThumbnailPage();
        await loadCollections();
    } catch (e) {
        status = 'Search failed';
        console.error(e);
    }
    searching = false;
  }

  async function loadSettings() {
    try {
        settings = await invoke('get_settings');
    } catch(e) { console.error('Failed to load settings:', e); }
}
 
async function saveSetting(key, value) {
    try {
        await invoke('set_setting', { key, value: String(value) });
        settings = { ...settings, [key]: String(value) };
    } catch(e) { console.error('Failed to save setting:', e); }
}

  async function findSimilar(path) {
      contextMenu = null;
      searching = true;
      status = 'Finding similar...';

      try {
          clearThumbCache();
          previousPaths = new Set(visibleResults.map(([p]) => p));
          const ranked = await invoke('find_similar', { path });
          results = ranked.map(([p, score, ratio]) => [p, score, ratio, null, null]);
          thumbLoadedCount = 0;
          status = `${results.length} similar images`;
          visibleResults = results.slice(0, THUMB_PAGE_SIZE);
          await loadNextThumbnailPage();
          query = '';
      } catch (e) {
          status = 'Failed';
          console.error(e);
      }
      searching = false;
  }

  function onQueryInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(search, 400);
  }

  function toDataUrl(bytes) {
    if (!bytes || bytes.length === 0) return '';
    const b64 = btoa(String.fromCharCode(...new Uint8Array(bytes)));
    return `data:image/jpeg;base64,${b64}`;
  }

  function getCardIndex(path) {
      return results.findIndex(([p]) => p === path);
  }

  function handleCardClick(e, path, index, rawPreview) {
    if (e.ctrlKey || e.metaKey) {
        const newSet = new Set(selectedPaths);
        if (newSet.has(path)) newSet.delete(path);
        else newSet.add(path);
        selectedPaths = newSet;
        lastClickedIndex = index;
    } else if (e.shiftKey && lastClickedIndex !== null) {
        const start = Math.min(lastClickedIndex, index);
        const end = Math.max(lastClickedIndex, index);
        const total = end - start;
        const totalDuration = Math.min(600, Math.max(200, total * 40));
        const delayPerCard = total > 0 ? totalDuration / total : 0;
        for (let i = start; i <= end; i++) {
            setTimeout(() => {
                selectedPaths = new Set([...selectedPaths, results[i][0]]);
            }, (i - start) * delayPerCard);
        }
    } else {
        selectedPaths = new Set([path]);
        lastClickedIndex = index;
        openLightbox(path, results[index][1], rawPreview);
    }
  }

  function onCardRightClick(e, path, score, index) {
    e.preventDefault();
    e.stopPropagation();
    if (!selectedPaths.has(path)) {
        selectedPaths = new Set([path]);
        lastClickedIndex = index;
    }
    const menuW = 210;
    const menuH = selectedPaths.size > 1 ? 180 : 320;
    const x = e.clientX + menuW > window.innerWidth ? e.clientX - menuW : e.clientX;
    const y = e.clientY + menuH > window.innerHeight ? e.clientY - menuH : e.clientY;
    contextMenu = { x, y, path, score };
  }

  function flashFeedback(path, type) {
      const newMap = new Map(feedbackFlash);
      newMap.set(path, type);
      feedbackFlash = newMap;
      setTimeout(() => {
          const m = new Map(feedbackFlash);
          m.delete(path);
          feedbackFlash = m;
      }, 2500);
  }

  async function markFeedback(signal) {
    const paths = selectedPaths.size > 0 ? [...selectedPaths] : [contextMenu.path];
    contextMenu = null;
    try {
        for (const path of paths) {
            await invoke('mark_feedback', { path, query, signal });
            flashFeedback(path, signal === 1 ? 'correct' : 'incorrect');
        }
        if (signal === -1) {
            deletingPaths = new Set(paths);
            await new Promise(r => setTimeout(r, 500));

            results = results.filter(([p]) => !paths.includes(p));
            visibleResults = visibleResults.filter(([p]) => !paths.includes(p));

            previousPaths = new Set([...previousPaths].filter(p => !paths.includes(p)));

            selectedPaths = new Set();
            deletingPaths = new Set();
            status = `${results.length} results`;
        }
    } catch(e) { console.error(e); }
  }

  function deleteSelected(type) {
    const paths = selectedPaths.size > 0 ? [...selectedPaths] : [contextMenu?.path].filter(Boolean);
    pendingDeletePaths = paths;
    pendingDeletePath = paths[0];
    contextMenu = null;

    const shouldPrompt = type === 'system'
        ? settings.prompt_delete_system === '1'
        : settings.prompt_delete_db === '1';

    if (shouldPrompt) {
        showDeleteConfirm = type;
    } else {
        if (type === 'system') deleteFromSystem();
        else deleteFromDb();
    }
}

  async function pickPath() {
    try {
      const selected = await open({
        directory: indexMode === 'folder',
        multiple: false,
        title: indexMode === 'folder' ? 'Select Folder' : 'Select Image'
      });
      if (selected) indexPath = selected;
    } catch (e) {
      console.error('Dialog failed:', e);
    }
  }

async function startIndexing() {
    if (!indexPath || !indexPath.trim()) return;
    indexing = true;
    indexDone = 0;
    indexTotal = 0;
    showIndexModal = false;
    status = 'Indexing...';

    try {
        // Save folder to watched list with auto_index preference
        if (indexMode === 'folder') {
            await invoke('save_folder', { path: indexPath, autoIndex });
        }
        await invoke('index_directory', { path: indexPath, notifyOnComplete: settings.notify_on_complete === '1' });
        await getCount();
        status = 'Indexing complete';
    } catch (e) {
        status = 'Indexing failed';
        console.error(e);
    }

    indexing = false;
    indexPath = '';
  }

  async function refreshResults() {
      if (query.trim()) {
          await search();
      } else {
          results = [];
          visibleResults = [];
      }
  }

function shouldNotify() {
    return settings.notify_on_complete === '1';
}

  async function getCount() {
    try {
      count = await invoke('get_indexed_count');
    } catch (e) {}
  }

  // Context menu actions
  async function openImage(path) {
    try { await invoke('open_path', { path }); } catch(e) { console.error(e); }
    contextMenu = null;
  }

  async function showInFolder(path) {
    try { await invoke('show_in_folder', { path }); } catch(e) { console.error(e); }
    contextMenu = null;
  }

  function confirmDelete(type) {
    pendingDeletePath = contextMenu.path;
    showDeleteConfirm = type;
    contextMenu = null;
  }

  async function deleteFromDb() {
    const paths = pendingDeletePaths || [pendingDeletePath];
    showDeleteConfirm = null;
    pendingDeletePath = null;
    pendingDeletePaths = null;
    
    deletingPaths = new Set(paths);
    await new Promise(r => setTimeout(r, 500));
    try {
        for (const p of paths) {
            await invoke('delete_from_db', { path: p });
        }
        results = results.filter(([p]) => !paths.includes(p));
        visibleResults = visibleResults.filter(([p]) => !paths.includes(p));
        selectedPaths = new Set();
        deletingPaths = new Set();
        await getCount();
        status = 'Removed from index';
    } catch(e) { console.error(e); }
}

async function deleteFromSystem() {
    const paths = pendingDeletePaths || [pendingDeletePath];
    showDeleteConfirm = null;
    pendingDeletePath = null;
    pendingDeletePaths = null;

    deletingPaths = new Set(paths);
    await new Promise(r => setTimeout(r, 500));
    try {
        for (const p of paths) {
            await invoke('delete_from_system', { path: p });
        }
        results = results.filter(([p]) => !paths.includes(p));
        visibleResults = visibleResults.filter(([p]) => !paths.includes(p));
        selectedPaths = new Set();
        deletingPaths = new Set();
        await getCount();
        status = 'File deleted';
    } catch(e) { console.error(e); }
}


  function closeContextMenu(e) {
    contextMenu = null;
    showHamburger = false;
    if (e && !e.target.closest('.card') && !e.target.closest('.ctx-menu')) {
        selectedPaths = new Set();
        lastClickedIndex = null;
    }
}

  // Lightbox
  function openLightbox(path, score, rawPreview = null) {
    console.log('rawPreview bytes:', rawPreview ? rawPreview.length : 'NULL');
    selectedImage = { path, score, rawPreview };
    zoomLevel = 1;
    zoomX = 50;
    zoomY = 50;
  }

  function onLightboxWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.2 : 0.2;
    zoomLevel = Math.max(1, Math.min(5, zoomLevel + delta));
  }

  function onLightboxMouseMove(e) {
    if (zoomLevel <= 1) return;
    const rect = e.currentTarget.getBoundingClientRect();
    zoomX = ((e.clientX - rect.left) / rect.width) * 100;
    zoomY = ((e.clientY - rect.top) / rect.height) * 100;
  }

  function handleKeydown(e) {
    if (e.key === 'Escape') {
      if (showDeleteConfirm) { showDeleteConfirm = null; pendingDeletePath = null; return; }
      selectedImage = null;
      showIndexModal = false;
      showSettingsModal = false;
      contextMenu = null;
    }
  }

  async function copyPath(path) {
      await navigator.clipboard.writeText(path);
      contextMenu = null;
  }

  async function notify(title, options) {
    let granted = await isPermissionGranted();
    if (!granted) {
        const permission = await requestPermission();
        granted = permission === 'granted';
    }
    if (granted) sendNotification({ title, ...options });
}

  function onWindowMouseDown(e) {
    if (e.target.closest('.ctx-menu')) return;
    if (e.target.closest('.hamburger-menu')) return;
    closeContextMenu(e);
}

  onMount(async () => {
    // Set up event listeners FIRST (non-blocking)
    const unlisten = await listen('tauri://drag-drop', (event) => {
      appDragOver = false;
      const paths = event.payload.paths;
      if (paths && paths.length > 0) {
        indexPath = paths[0];
        const p = paths[0];
        const hasExtension = p.includes('.') && !p.endsWith('\\') && !p.endsWith('/');
        indexMode = hasExtension ? 'file' : 'folder';
        showIndexModal = true;
      }
    });

    selectAllHandler = (e) => handleSelectAll(e);
    window.addEventListener('keydown', selectAllHandler);

    await listen('tauri://drag-enter', () => { appDragOver = true; });
    await listen('tauri://drag-leave', () => { appDragOver = false; });

    await listen('reindex-started', (event) => {
        indexing = true;
        isReindexing = true;
        indexProgress = { done: 0, total: event.payload, estimated_remaining_ms: null };
    });

    await listen('index-complete', () => {
        indexing = false;
        isReindexing = false;
    });

    await listen('index-complete', async () => {
        await getCount();
        indexing = false;
        status = 'Indexing complete';
        if (shouldNotify()) {
            await notify('Scout', { body: 'Indexing complete' });
        }
    });

    await listen('duplicates-removed', async (event) => {
        await getCount();
        status = `Auto-removed ${event.payload} duplicate${event.payload !== 1 ? 's' : ''}`;
    });

    await listen("close-requested-while-indexing", async () => {
        const confirmed = await confirm(
            "Indexing is currently in progress. Are you sure you want to close?",
            { title: 'Scout', kind: 'warning' }
        );
        if (confirmed) {
            await getCurrentWindow().close();
        }
    })

    const unlistenProgress = await listen('index-progress', (event) => {
      indexDone = event.payload.done;
      indexTotal = event.payload.total;
      indexEstimatedMs = event.payload.estimated_remaining_ms;
      status = `${isReindexing ? 'Re-indexing' : 'Indexing'} ${indexDone} / ${indexTotal}`;
  
      // Track elapsed time
      if (!indexStartTime) {
        indexStartTime = Date.now();
        updateInterval = setInterval(() => {
          indexElapsedMs = Date.now() - indexStartTime;
        }, 100);
      }
    });

    // Load data in background WITHOUT awaiting (non-blocking)
    loadSettings().catch(e => console.error('Failed to load settings:', e));
    getCount().catch(e => console.error('Failed to get count:', e));
    loadCollections().catch(e => console.error('Failed to load collections:', e));
    loadCollections().catch(e => console.error('Failed to load collections:', e));

    return () => { unlisten(); unlistenProgress(); };
});

onMount(() => {
    // Listen to index progress
    return listen('index-progress', (event) => {
      const { done, total, estimated_remaining_ms } = event.payload;
      indexDone = done;
      indexTotal = total;
      indexEstimatedMs = estimated_remaining_ms;
      
      // Track elapsed time
      if (!indexStartTime) {
        indexStartTime = Date.now();
        updateInterval = setInterval(() => {
          indexElapsedMs = Date.now() - indexStartTime;
        }, 100);
      }
    });
  });
  function onIndexComplete() {
    if (updateInterval) clearInterval(updateInterval);
    indexStartTime = null;
    indexElapsedMs = 0;
  }

  onDestroy(() => {
      if (selectAllHandler) {
          window.removeEventListener('keydown', selectAllHandler);
      }
  })
</script>

<svelte:window on:keydown={handleKeydown} on:click={closeContextMenu} on:mousedown={onWindowMouseDown} />

<div class="app">
  {#if appDragOver}
    <div class="app-drop-overlay" role="status" aria-live="polite">
      <div class="app-drop-inner">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span>Drop to index</span>
      </div>
    </div>
  {/if}

  <Header 
    {status}
    {count}
    {indexing}
    {indexDone}
    {indexTotal}
    {indexEstimatedMs}
    {searching}
    {query}
    {showHamburger}
    onIndexClick={() => { indexPath = ''; showIndexModal = true; }}
    onSettingsClick={() => showSettingsModal = true}
    onQueryInput={(e) => { query = e.target.value; onQueryInput(); }}
    onHamburgerToggle={() => showHamburger = !showHamburger}
    onManageFoldersClick={openManageFolders}
    onCollectionsClick={() => { loadCollections(); showCollectionsModal = true; }}
  />

  {#if showHamburger}
    <HamburgerMenu
      {indexing}
      onIndexClick={() => { showHamburger = false; indexPath = ''; showIndexModal = true; }}
      onSettingsClick={() => { showHamburger = false; showSettingsModal = true; }}
      onManageFoldersClick={() => { showHamburger = false; openManageFolders(); }}
      onCollectionsClick={() => { showHamburger = false; loadCollections(); showCollectionsModal = true; }}
    />
  {/if}

  <Gallery 
    results={visibleResults}
    {count}
    {selectedPaths}
    feedbackFlash={new Map([...feedbackFlash.entries(), ...collectionFlash.entries()].map(([k, v]) => [k, typeof v === 'object' ? 'correct' : v]))}
    {deletingPaths}
    {previousPaths}
    contextMenuPath={contextMenu?.path}
    {settings}
    getThumbUrl={getThumbUrl}
    onCardClick={handleCardClick}
    onCardRightClick={onCardRightClick}
    onGridClick={() => { selectedPaths = new Set(); lastClickedIndex = null; }}
    onScroll={handleGalleryScroll}
  />

  {#if contextMenu}
    <ContextMenu
      x={contextMenu.x}
      y={contextMenu.y}
      {selectedPaths}
      contextMenuPath={contextMenu.path}
      {collections}
      imageCollectionsMap={imageCollectionsMap}
      onOpen={() => openImage(contextMenu.path)}
      onShowInFolder={() => showInFolder(contextMenu.path)}
      onFindSimilar={() => findSimilar(contextMenu.path)}
      onCopyPath={() => copyPath(contextMenu.path)}
      onAddToCollection={(collectionId) => addToCollection(collectionId)}
      onRemoveFromCollection={(collectionId) => removeFromCollection(collectionId)}
      onOpenCollectionsModal={() => { loadCollections(); showCollectionsModal = true; contextMenu = null; }}
      onMarkCorrect={() => markFeedback(1)}
      onMarkIncorrect={() => markFeedback(-1)}
      onDeleteFromDb={() => deleteSelected('db')}
      onDeleteFromSystem={() => deleteSelected('system')}
    />
  {/if}

  <!-- Delete confirm -->
  {#if showDeleteConfirm && pendingDeletePath}
    <DeleteConfirmModal
      deleteType={showDeleteConfirm}
      {pendingDeletePath}
      {pendingDeletePaths}
      onConfirm={() => showDeleteConfirm === 'system' ? deleteFromSystem() : deleteFromDb()}
      onCancel={() => { showDeleteConfirm = null; pendingDeletePath = null; }}
    />
  {/if}

  {#if showIndexModal}
    <IndexModal
      {indexMode}
      {indexPath}
      {indexRecursive}
      {autoIndex}
      onIndexModeChange={(mode) => indexMode = mode}
      onPickPath={pickPath}
      onRecursiveChange={(val) => indexRecursive = val}
      onAutoIndexChange={(val) => autoIndex = val}
      onStartIndexing={startIndexing}
      onClose={() => showIndexModal = false}
    />
  {/if}

  {#if showSettingsModal}
    <SettingsModal
      {settings}
      onImageLayoutChange={(layout) => saveSetting('image_layout', layout)}
      onThumbnailSizeChange={(size) => saveSetting('thumbnail_size', size)}
      onNotifyChange={(val) => saveSetting('notify_on_complete', val)}
      onPromptDbChange={(val) => saveSetting('prompt_delete_db', val)}
      onPromptSystemChange={(val) => saveSetting('prompt_delete_system', val)}
      onExcludedFileTypesChange={(val) => saveSetting('excluded_file_types', val)}
      onModelChange={handleModelChange}
      onRegenThumbnails={regenThumbnails}
      onDuplicateFind={runDuplicateFind}
      onClearDatabase={clearDatabase}
      onClose={() => showSettingsModal = false}
    />
  {/if}

  {#if showDuplicatesModal}
    <DuplicatesModal
      {duplicateGroups}
      {settings}
      onRemove={runRemoveDuplicates}
      onClose={() => showDuplicatesModal = false}
    />
  {/if}

  {#if selectedImage}
    <Lightbox
      imagePath={selectedImage.path}
      score={selectedImage.score}
      {zoomLevel}
      {zoomX}
      {zoomY}
      rawPreview={selectedImage.rawPreview ?? null}
      onWheel={onLightboxWheel}
      onMouseMove={onLightboxMouseMove}
      onResetZoom={() => { zoomLevel = 1; zoomX = 50; zoomY = 50; }}
      onAddToCollection={() => {
        selectedPaths = new Set([selectedImage.path]);
        contextMenu = {
          x: window.innerWidth / 2,
          y: window.innerHeight / 2,
          path: selectedImage.path
        };
      }}
      onClose={() => { selectedImage = null; }}
    />
  {/if}
  {#if showManageFolders}
    <ManageFoldersModal
    folders={managedFolders}
    loading={foldersLoading}
    onToggleAutoIndex={toggleFolderAutoIndex}
    onUnindexFolder={unindexFolder}
    onDeleteFromSystem={deleteFolderFromSystem}
    onAddFolder={() => { showManageFolders = false; indexPath = ''; showIndexModal = true; }}
    onClose={() => showManageFolders = false}
    />
  {/if}

  {#if showCollectionsModal}
    <CollectionsModal
      bind:showModal={showCollectionsModal}
      {selectedImage}
      {collections}
    />
  {/if}
</div>

<style>
.app {
	display: flex;
	flex-direction: column;
	height: 100vh;
	background: var(--background);
	position: relative;
}

.app-drop-overlay {
	position: fixed;
	inset: 0;
	background: var(--background-transparent);
	border: var(--border-medium) dashed var(--primary);
	z-index: 200;
	display: flex;
	align-items: center;
	justify-content: center;
	pointer-events: none;
}

.app-drop-inner {
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 12px;
	color: var(--primary);
	font-family: var(--secondary-font);
	font-size: var(--font-large);
	letter-spacing: 2px;
}
</style>