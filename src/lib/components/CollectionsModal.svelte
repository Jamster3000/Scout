<script>
  import { invoke } from '@tauri-apps/api/core';
  import { convertFileSrc } from '@tauri-apps/api/core';
  import Gallery from './Gallery.svelte';
  import Lightbox from './Lightbox.svelte';

  export let showModal = false;
  export let collections = [];

  let selectedCollectionId = null;
  let collectionImages = [];
  let selectedCollectionName = '';
  let selectedCollectionDescription = '';
  let selectedImagePaths = new Set();
  let contextMenu = null;
  let newCollectionName = '';
  let newCollectionDescription = '';
  let showNewForm = false;
  let loading = false;
  let collectionFlash = new Map();
  let deletingPaths = new Set();
  let errorMessage = '';
  let showError = false;
  let confirmAction = null;
  let selectedImage = null;
  let editingCollectionId = null;
  let editingName = '';
  let editingDescription = '';

  async function selectCollection(collectionId, name, description = '') {
    selectedCollectionId = collectionId;
    selectedCollectionName = name;
    selectedCollectionDescription = description;
    selectedImagePaths = new Set();
    contextMenu = null;
    try {
      const items = await invoke('get_collection_items', { collectionId });
      collectionImages = items.map(item => [
        item.path,
        0,
        item.aspect_ratio,
        item.thumbnail,
        null
      ]);
    } catch (e) {
      console.error('Failed to load collection items:', e);
      collectionImages = [];
    }
  }

  async function createCollection() {
    if (!newCollectionName.trim()) return;
    
    if (collections.some(c => c.name.toLowerCase() === newCollectionName.toLowerCase())) {
      errorMessage = 'A collection with this name already exists';
      showError = true;
      setTimeout(() => { showError = false; }, 3000);
      return;
    }
    
    loading = true;
    try {
      const id = await invoke('create_collection', {
        name: newCollectionName,
        description: newCollectionDescription.trim() || null
      });
      collections = [...collections, { 
        id, 
        name: newCollectionName, 
        description: newCollectionDescription.trim() || null, 
        created_at: Date.now(), 
        updated_at: Date.now() 
      }];
      
      newCollectionName = '';
      newCollectionDescription = '';
      showNewForm = false;
      errorMessage = '';

    } catch (e) {
      console.error('Failed to create collection:', e);
      const errorStr = String(e);

      if (errorStr.includes('UNIQUE constraint')) {
        errorMessage = 'A collection with this name already exists';
      } else if (errorStr.includes('database is locked')) {
        errorMessage = 'Database is busy, please try again';
      } else {
        errorMessage = `Error: ${e}`;
      }

      showError = true;
      setTimeout(() => { 
          showError = false;
      }, 4000);

    } finally {
      loading = false;
    }
  }

  function deleteCollection() {
    if (!selectedCollectionId) return;
    confirmAction = { type: 'deleteCollection', collectionId: selectedCollectionId };
  }

  async function executeDeleteCollection() {
    if (!confirmAction?.collectionId) return;
    try {
      await invoke('delete_collection', { collectionId: confirmAction.collectionId });
      collections = collections.filter(c => c.id !== confirmAction.collectionId);
      selectedCollectionId = null;
      selectedCollectionName = '';
      selectedCollectionDescription = '';
      collectionImages = [];
      confirmAction = null;
    } catch (e) {
      console.error('Failed to delete collection:', e);
      confirmAction = null;
    }
  }

  function cancelConfirm() {
    confirmAction = null;
  }

  function openLightbox(path) {
    const imageData = collectionImages.find(([p]) => p === path);
    if (imageData) {
      selectedImage = { path: imageData[0], score: 0, thumb: imageData[2], ratio: imageData[3] };
    }
    contextMenu = null;
  }

  async function removeFromCollection(imagePath) {
    if (!selectedCollectionId) return;
    confirmAction = { type: 'removeFromCollection', imagePath };
    contextMenu = null;
  }

  async function executeRemoveFromCollection() {
    if (!confirmAction?.imagePath) return;
    const imagePath = confirmAction.imagePath;
    confirmAction = null;
    try {
      // Trigger delete animation first
      deletingPaths = new Set([imagePath]);
      await new Promise(r => setTimeout(r, 500));
      
      await invoke('remove_from_collection', {
        collectionId: selectedCollectionId,
        imagePath
      });
      collectionImages = collectionImages.filter(([p]) => p !== imagePath);
      selectedImagePaths.delete(imagePath);
      selectedImagePaths = selectedImagePaths;
      deletingPaths = new Set();

    } catch (e) {
      console.error('Failed to remove from collection:', e);
      deletingPaths = new Set();
    }
  }

  async function deleteImageFromDb(imagePath) {
    if (!selectedCollectionId) return;
    confirmAction = { type: 'deleteFromDb', imagePath };
    contextMenu = null;
  }

  async function executeDeleteImageFromDb() {
    if (!confirmAction?.imagePath) return;
    const imagePath = confirmAction.imagePath;
    confirmAction = null;
    try {
      deletingPaths = new Set([imagePath]);
      await new Promise(r => setTimeout(r, 500));
      
      await invoke('delete_from_db', { path: imagePath });
      collectionImages = collectionImages.filter(([p]) => p !== imagePath);
      selectedImagePaths.delete(imagePath);
      selectedImagePaths = selectedImagePaths;
      deletingPaths = new Set();
    } catch (e) {
      console.error('Failed to delete from database:', e);
      deletingPaths = new Set();
    }
  }

  async function deleteImageFromSystem(imagePath) {
    if (!selectedCollectionId) return;
    confirmAction = { type: 'deleteFromSystem', imagePath };
    contextMenu = null;
  }

  async function executeDeleteImageFromSystem() {
    if (!confirmAction?.imagePath) return;
    const imagePath = confirmAction.imagePath;
    confirmAction = null;
    try {
      deletingPaths = new Set([imagePath]);
      await new Promise(r => setTimeout(r, 500));
      
      await invoke('delete_from_system', { path: imagePath });
      collectionImages = collectionImages.filter(([p]) => p !== imagePath);
      selectedImagePaths.delete(imagePath);
      selectedImagePaths = selectedImagePaths;
      deletingPaths = new Set();
    } catch (e) {
      console.error('Failed to delete from system:', e);
      deletingPaths = new Set();
    }
  }

  function clearCollectionImages() {
    if (!selectedCollectionId) return;
    confirmAction = { type: 'clearCollection', collectionId: selectedCollectionId };
  }

  async function executeClearCollection() {
    if (!confirmAction?.collectionId) return;
    const collectionId = confirmAction.collectionId;
    confirmAction = null;
    try {
      // Animate all cards out
      deletingPaths = new Set(collectionImages.map(([p]) => p));
      await new Promise(r => setTimeout(r, 500));

      for (const [path] of collectionImages) {
        await invoke('remove_from_collection', {
          collectionId,
          imagePath: path
        });
      }
      collectionImages = [];
      selectedImagePaths = new Set();
      deletingPaths = new Set();
    } catch (e) {
      console.error('Failed to clear collection:', e);
      deletingPaths = new Set();
    }
  }

  function editCollection() {
    if (!selectedCollectionId) return;
    editingCollectionId = selectedCollectionId;
    editingName = selectedCollectionName;
    editingDescription = selectedCollectionDescription || '';
  }

  async function saveEditCollection() {
    if (!editingCollectionId) return;
    try {
      await invoke('rename_collection', {
        collectionId: editingCollectionId,
        newName: editingName
      });
      collections = collections.map(c => 
        c.id === editingCollectionId 
          ? { ...c, name: editingName, description: editingDescription }
          : c
      );
      selectedCollectionName = editingName;
      selectedCollectionDescription = editingDescription;
      editingCollectionId = null;
    } catch (e) {
      console.error('Failed to edit collection:', e);
    }
  }

  function cancelEditCollection() {
    editingCollectionId = null;
  }
</script>

{#if showModal}
  <div 
    class="modal-backdrop" 
    role="button"
    tabindex="0"
    aria-label="Close Collections Modal"
    on:click|self={() => showModal = false}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && (showModal = false)}
  >
    <div class="collections-modal-content">
      <div class="collections-modal-header">
        <h2>Legal/Saved Collections</h2>
        <button class="modal-close" type="button" on:click={() => showModal = false} aria-label="Close modal">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </div>

      <div class="collections-modal-body">
        <div class="collections-sidebar">
          <div class="sidebar-header">
            <h3>Collections</h3>
            <button class="btn btn-secondary" type="button" on:click={() => showNewForm = !showNewForm} title="New collection">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
          </div>

          {#if showNewForm}
            <div class="new-form">
              {#if showError}
                <div class="error-message" role="alert">
                  {errorMessage}
                </div>
              {/if}
              <input
                type="text"
                placeholder="Collection name"
                bind:value={newCollectionName}
                on:keydown={(e) => e.key === 'Enter' && createCollection()}
              />
              <textarea
                placeholder="Description (optional)"
                bind:value={newCollectionDescription}
                rows="2"
              ></textarea>
              <button class="btn-create" type="button" on:click={createCollection} disabled={loading || !newCollectionName.trim()}>
                Create
              </button>
            </div>
          {/if}

          <div class="collections-list">
            {#each collections as collection (collection.id)}
              <button
                class="collection-item"
                type="button"
                class:active={selectedCollectionId === collection.id}
                on:click={() => selectCollection(collection.id, collection.name, collection.description)}
              >
                <span class="collection-item-name">{collection.name}</span>
              </button>
            {/each}
          </div>
        </div>

        <div class="collections-content">
          {#if selectedCollectionId}
            <div class="content-header">
              <div class="header-title">
                <h3>{selectedCollectionName}</h3>
                {#if selectedCollectionDescription}
                  <p class="header-description">{selectedCollectionDescription}</p>
                {/if}
              </div>
              <div class="header-buttons">
                <button class="btn-action" type="button" on:click={editCollection} title="Edit collection name and description">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                </button>
                <button class="btn-action" type="button" on:click={clearCollectionImages} title="Remove all images from collection">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
                </button>
                <button class="btn-delete" type="button" on:click={deleteCollection} title="Delete collection">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
                </button>
              </div>
            </div>

            <div class="gallery-wrapper">
              {#if collectionImages.length > 0}
                <Gallery
                  results={collectionImages}
                  count={collectionImages.length}
                  selectedPaths={selectedImagePaths}
                  feedbackFlash={collectionFlash}
                  {deletingPaths}
                  previousPaths={new Set()}
                  settings={{ image_layout: 'grid' }}
                  onCardClick={(e, path, index, rawPreview) => {
                    if (e.ctrlKey || e.metaKey) {
                      const newSet = new Set(selectedImagePaths);
                      if (newSet.has(path)) newSet.delete(path);
                      else newSet.add(path);
                      selectedImagePaths = newSet;
                    } else {
                      selectedImagePaths = new Set([path]);
                      openLightbox(path);
                    }
                  }}
                  onCardRightClick={(e, path, score, index, rawPreview) => {
                    e.preventDefault();
                    if (!selectedImagePaths.has(path)) {
                      selectedImagePaths = new Set([path]);
                    }
                    const menuW = 200;
                    const menuH = selectedImagePaths.size > 1 ? 140 : 160;
                    const x = e.clientX + menuW > window.innerWidth ? e.clientX - menuW : e.clientX;
                    const y = e.clientY + menuH > window.innerHeight ? e.clientY - menuH : e.clientY;
                    contextMenu = { x, y, path };
                  }}
                  onGridClick={() => { selectedImagePaths = new Set(); contextMenu = null; }}
                />
              {:else}
                <div class="empty-state">
                  <p>No images in this collection</p>
                </div>
              {/if}
            </div>

            {#if contextMenu}
              <div class="collection-context-menu" style="left:{contextMenu.x}px; top:{contextMenu.y}px" role="menu">
                {#if selectedImagePaths.size > 1}
                  <div class="ctx-header">{selectedImagePaths.size} images selected</div>
                {/if}
                <button type="button" on:click={() => { openLightbox(contextMenu.path); }} class="menu-item">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                  Open
                </button>
                <button type="button" on:click={() => removeFromCollection(contextMenu.path)} class="menu-item">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
                  Remove from collection
                </button>
                <button type="button" on:click={() => deleteImageFromDb(contextMenu.path)} class="menu-item menu-warn">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
                  Delete from database
                </button>
                <button type="button" on:click={() => deleteImageFromSystem(contextMenu.path)} class="menu-item menu-danger">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
                  Delete from system
                </button>
              </div>
            {/if}
          {:else}
            <div class="empty-state">
              <p>Select a collection to view images</p>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

{#if selectedImage}
  <Lightbox
    imagePath={selectedImage.path}
    score={selectedImage.score}
    zoomLevel={1}
    zoomX={50}
    zoomY={50}
    onWheel={() => {}}
    onMouseMove={() => {}}
    onAddToCollection={() => {}}
    onClose={() => { selectedImage = null; }}
    onResetZoom={() => {}}
  />
{/if}

{#if editingCollectionId}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel editing"
    on:click|self={cancelEditCollection}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelEditCollection()}
  >
    <div class="edit-modal">
      <div class="edit-header">
        <h3>Edit Collection</h3>
        <button class="close-btn" type="button" on:click={cancelEditCollection}>✕</button>
      </div>
      <div class="edit-body">
        <div class="edit-field">
          <label for="edit-name-input">Name</label>
          <input id="edit-name-input" type="text" bind:value={editingName} placeholder="Collection name" />
        </div>
        <div class="edit-field">
          <label for="edit-desc-input">Description</label>
          <textarea id="edit-desc-input" bind:value={editingDescription} placeholder="Description (optional)" rows="3"></textarea>
        </div>
        <div class="edit-actions">
          <button class="btn btn-primary" type="button" on:click={cancelEditCollection}>Cancel</button>
          <button class="btn btn-primary" type="button" on:click={saveEditCollection}>Save</button>
        </div>
      </div>
    </div>
  </div>
{/if}

{#if confirmAction && confirmAction.type === 'deleteCollection'}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel confirmation"
    on:click|self={cancelConfirm}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelConfirm()}
  >
    <div class="confirm-box">
      <div class="confirm-icon danger">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
      </div>
      <span class="confirm-title">Delete this collection?</span>
      <p class="confirm-desc">This will permanently delete the collection "<strong>{selectedCollectionName}</strong>". Images will remain on disk and in the database.</p>
      <div class="confirm-actions">
        <button class="btn btn-primary" type="button" on:click={cancelConfirm}>Cancel</button>
        <button class="btn btn-primary danger" type="button" on:click={executeDeleteCollection}>Delete Collection</button>
      </div>
    </div>
  </div>
{/if}

{#if confirmAction && confirmAction.type === 'clearCollection'}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel confirmation"
    on:click|self={cancelConfirm}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelConfirm()}
  >
    <div class="confirm-box">
      <div class="confirm-icon warn">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
      </div>
      <span class="confirm-title">Remove all images?</span>
      <p class="confirm-desc">This will remove all <strong>{collectionImages.length}</strong> image{collectionImages.length !== 1 ? 's' : ''} from this collection. Images will not be deleted from the database or system.</p>
      <div class="confirm-actions">
        <button class="btn btn-primary" type="button" on:click={cancelConfirm}>Cancel</button>
        <button class="btn btn-primary warn" type="button" on:click={executeClearCollection}>Remove All</button>
      </div>
    </div>
  </div>
{/if}

{#if confirmAction && confirmAction.type === 'removeFromCollection'}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel confirmation"
    on:click|self={cancelConfirm}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelConfirm()}
  >
    <div class="confirm-box">
      <div class="confirm-icon warn">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
      </div>
      <span class="confirm-title">Remove from collection?</span>
      <p class="confirm-desc">This will remove this image from the collection. The image will remain in the database and on your system.</p>
      <div class="confirm-actions">
        <button class="btn btn-primary" type="button" on:click={cancelConfirm}>Cancel</button>
        <button class="btn btn-primary warn" type="button" on:click={executeRemoveFromCollection}>Remove</button>
      </div>
    </div>
  </div>
{/if}

{#if confirmAction && confirmAction.type === 'deleteFromDb'}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel confirmation"
    on:click|self={cancelConfirm}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelConfirm()}
  >
    <div class="confirm-box">
      <div class="confirm-icon warn">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
      </div>
      <span class="confirm-title">Delete from database?</span>
      <p class="confirm-desc">This will remove this image from Scout's index. The image file will remain on your system.</p>
      <div class="confirm-actions">
        <button class="btn btn-primary" type="button" on:click={cancelConfirm}>Cancel</button>
        <button class="btn btn-primary warn" type="button" on:click={executeDeleteImageFromDb}>Delete from index</button>
      </div>
    </div>
  </div>
{/if}

{#if confirmAction && confirmAction.type === 'deleteFromSystem'}
  <div 
    class="modal-overlay" 
    role="button"
    tabindex="0"
    aria-label="Cancel confirmation"
    on:click|self={cancelConfirm}
    on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && cancelConfirm()}
  >
    <div class="confirm-box">
      <div class="confirm-icon danger">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
      </div>
      <span class="confirm-title">Delete from system?</span>
      <p class="confirm-desc"><strong>This will permanently delete the image file from your computer.</strong> This action cannot be undone.</p>
      <div class="confirm-actions">
        <button class="btn btn-primary" type="button" on:click={cancelConfirm}>Cancel</button>
        <button class="btn btn-primary danger" type="button" on:click={executeDeleteImageFromSystem}>Delete file</button>
      </div>
    </div>
  </div>
{/if}

<style>
.collections-modal-content {
  background: var(--background);
  border: var(--border-small) solid var(--primary-dark);
  border-radius: 12px;
  width: 95%;
  height: 90vh;
  max-width: 1400px;
  display: flex;
  flex-direction: column;         
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6);
  overflow: hidden;      
}

.collections-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: var(--border-small) solid var(--primary-dark);
  flex-shrink: 0;
}

.collections-modal-header h2 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--primary-light);
  font-family: var(--secondary-font);
  letter-spacing: 3px;
  text-transform: uppercase;
}

.collections-modal-body {
  display: flex;
  flex: 1; 
  overflow: hidden;
  min-height: 0;
}

.collections-sidebar {
  width: 280px;
  border-right: var(--border-small) solid var(--primary-dark);
  display: flex;
  flex-direction: column;
  background: var(--background);
  flex-shrink: 0;
  overflow: hidden;
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: var(--border-small) solid var(--primary-dark);
  flex-shrink: 0;
}

.sidebar-header h3 {
  margin: 0;
  color: var(--primary-very-light);
  font-size: 14px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.new-form {
  padding: 12px;
  border-bottom: var(--border-small) solid var(--primary-dark);
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
}

.error-message {
  font-size: 12px;
  color: var(--primary-danger);
  background: rgba(248, 113, 113, 0.1);
  border: 1px solid rgba(248, 113, 113, 0.3);
  padding: 8px 10px;
  border-radius: 4px;
}

.new-form input {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(124, 58, 237, 0.3);
  border-radius: 4px;
  color: var(--primary-very-light);
  padding: 6px 10px;
  font-size: 12px;
}

.new-form textarea {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(124, 58, 237, 0.3);
  border-radius: 4px;
  color: var(--primary-very-light);
  padding: 6px 10px;
  font-size: 12px;
  font-family: inherit;
  resize: vertical;
  min-height: 50px;
}

.new-form input:focus,
.new-form textarea:focus {
  outline: none;
  border-color: var(--primary);
}

.btn-create {
  background: var(--primary-dark);
  border: none;
  color: var(--primary-light);
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-create:hover:not(:disabled) {
  background: var(--primary);
}

.btn-create:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.collections-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px;
}

.collection-item {
  background: transparent;
  border: 1px solid transparent;
  color: var(--primary-very-light);
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
  font-size: var(--font-small);
  transition: all 0.15s;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}

.collection-item:hover {
  background: rgba(124, 58, 237, 0.1);
  color: #c4b5fd;
}

.collection-item.active {
  background: rgba(124, 58, 237, 0.2);
  color: var(--primary-very-light);
  border-color: var(--primary);
}

.collection-item-name {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
}

.collections-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden; 
  position: relative;
  min-width: 0;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: var(--border-small) solid var(--primary-dark);
  flex-shrink: 0;
  gap: 12px;
}

.header-title {
  flex: 1;
  min-width: 0;
}

.content-header h3 {
  margin: 0 0 4px 0;
  color: var(--primary-very-light);
  font-size: 16px;
  font-weight: 600;
}

.header-description {
  margin: 0;
  font-size: 12px;
  color: var(--primary-very-light);
  line-height: 1.3;
}

.header-buttons {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.btn-action {
  background: none;
  border: var(--border-small) solid var(--primary-dark);
  color: var(--primary-very-light);
  cursor: pointer;
  padding: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s;
}

.btn-action:hover {
  background: rgba(124, 58, 237, 0.1);
  color: var(--primary-very-light);
  border-color: var(--primary);
}

.btn-delete {
  background: none;
  border: var(--border-small) solid transparent;
  color: var(--primary-very-light);
  cursor: pointer;
  padding: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s;
}

.btn-delete:hover {
  background: rgba(239, 68, 68, 0.1);
  color: var(--primary-danger);
  border-color: var(--primary-danger);
}

.gallery-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
}

.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--primary-very-light);
  font-size: 14px;
}

.collection-context-menu {
  position: fixed;
  background: var(--background);
  border: var(--border-small) solid var(--primary-dark);
  border-radius: 8px;
  padding: 6px;
  min-width: 200px;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(124, 58, 237, 0.1);
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.ctx-header {
  padding: 6px 10px 8px;
  font-size: 11px;
  color: var(--primary-very-light);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: var(--border-small) solid var(--primary-dark);
  margin-bottom: 4px;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 4px;
  background: transparent;
  border: none;
  color: var(--primary-very-light);
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  text-align: left;
  transition: all 0.15s;
}

.menu-item:hover {
  background: rgba(124, 58, 237, 0.1);
  color: var(--primary-very-light);
}

.menu-warn {
  color: var(--primary-warn);
}

.menu-warn:hover {
  background: rgba(251, 146, 60, 0.1);
}

.menu-danger {
  color: var(--primary-danger);
}

.menu-danger:hover {
  background: rgba(239, 68, 68, 0.1);
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(4, 4, 10, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  backdrop-filter: blur(4px);
}

.edit-modal {
  background: var(--background);
  border: var(--border-small) solid var(--primary-dark);
  border-radius: 12px;
  width: 440px;
  box-shadow: 0 32px 80px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(124, 58, 237, 0.1);
  overflow: hidden;
}

.edit-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: var(--border-small) solid var(--primary-dark);
}

.edit-header h3 {
  margin: 0;
  font-size: var(--font-medium);
  font-weight: 600;
  color: var(--primary-very-light);
}

.close-btn {
  background: none;
  border: none;
  color: var(--primary-very-light);
  cursor: pointer;
  font-size: var(--font-small);
  transition: color 0.2s;
}

.close-btn:hover {
  color: var(--primary-light);
}

.edit-body {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.edit-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.edit-field label {
  font-size: 12px;
  color: var(--primary-light);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.edit-field input,
.edit-field textarea {
  background: var(--background);
  border: var(--border-small) solid var(--primary-dark);
  border-radius: 7px;
  color: var(--primary-very-light);
  padding: 8px 12px;
  font-size: var(--font-small);
  outline: none;
  transition: border-color 0.2s;
  font-family: inherit;
}

.edit-field input:focus,
.edit-field textarea:focus {
  border-color: var(--primary);
}

.edit-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  padding-top: 8px;
  border-top: var(--border-small) solid var(--primary-dark);
}

.confirm-box {
  background: var(--background);
  border: var(--border-small) solid var(--primary-dark);
  border-radius: 12px;
  width: 440px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-shadow: 0 32px 80px rgba(0, 0, 0, 0.8);
}

.confirm-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.confirm-icon.danger {
  background: rgba(239, 68, 68, 0.15);
  color: var(--primary-danger);
  border: var(--border-small) solid rgba(239, 68, 68, 0.3);
}

.confirm-icon.warn {
  background: rgba(251, 146, 60, 0.15);
  color: var(--primary-warn);
  border: var(--border-small) solid rgba(251, 146, 60, 0.3);
}

.confirm-title {
  font-size: var(--font-medium);
  font-weight: 600;
  color: var(--primary-very-light);
}

.confirm-desc {
  font-size: var(--font-small);
  color: var(--primary-very-light);
  line-height: 1.6;
  margin: 0;
}

.confirm-desc strong {
  color: var(--primary-very-light);
}

.confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding-top: 12px;
  border-top: var(--border-small) solid var(--primary-dark);
}

.collections-list::-webkit-scrollbar {
  width: 6px;
}

.collections-list::-webkit-scrollbar-track {
  background: transparent;
}

.collections-list::-webkit-scrollbar-thumb {
  background: rgba(124, 58, 237, 0.3);
  border-radius: 3px;
}

.collections-list::-webkit-scrollbar-thumb:hover {
  background: rgba(124, 58, 237, 0.5);
}
</style>