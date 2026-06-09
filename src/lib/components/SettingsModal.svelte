<script>
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';
  
  export let settings = {};

  export let onImageLayoutChange = () => {};
  export let onThumbnailSizeChange = () => {};
  export let onNotifyChange = () => {};
  export let onExcludedFileTypesChange = () => {};
  export let onPromptDbChange = () => {};
  export let onPromptSystemChange = () => {};
  export let onRegenThumbnails = () => {};
  export let onDuplicateFind = () => {};
  export let onClearDatabase = () => {};
  export let onModelChange = () => {};
  export let onClose = () => {};

  let thumbnailSizeChanged = false;
  let availableModels = [];
  let currentModel = settings.model_family || 'CLIP ViT-B-16';
  let showModelChangeWarning = false;
  let pendingModel = null;
  let showDropdown = false;
  let selectContainer;

  async function loadModels() {
    try {
      availableModels = await invoke('get_available_models');
    } catch (e) {
      console.error('Failed to load available models:', e);
    }
  }

  function handleModelChange(e) {
    const newModel = e.target.value;
    if (newModel !== currentModel) {
      pendingModel = newModel;
      showModelChangeWarning = true;
    }
  }

  async function confirmModelChange() {
    if (!pendingModel) return;
    
    showModelChangeWarning = false;
    const selectedModel = availableModels.find(m => m.id === pendingModel);
    if (selectedModel) {
      // Save the model settings
      await onModelChange(selectedModel);
      
      // Reload the model immediately
      try {
        await invoke('reload_model');
      } catch (e) {
        console.error('Failed to reload model:', e);
      }
      
      currentModel = pendingModel;
    }
    pendingModel = null;
  }

  function cancelModelChange() {
    showModelChangeWarning = false;
    pendingModel = null;
  }

  function handleExcludedFileTypesInput(e) {
    onExcludedFileTypesChange(e.target.value);
  }

  function handleClickOutside(e) {
    if (selectContainer && !selectContainer.contains(e.target)) {
      showDropdown = false;
    }
  }

  onMount(() => {
    loadModels();
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  });
</script>

<svelte:window on:keydown={(e) => {
  if (e.key === 'Escape') {
    if (showModelChangeWarning) cancelModelChange(); else onClose();
  }
}} />

<div class="modal-backdrop" style="position: fixed; inset: 0;">
  <button 
    type="button" 
    class="backdrop-trigger" 
    on:click={onClose} 
    aria-label="Close settings"
    style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
  ></button>

  <div class="modal" role="dialog" aria-modal="true" aria-label="Settings" style="position: relative; z-index: 1;">
    <div class="modal-header">
      <span class="modal-title">Settings</span>
      <button class="modal-close" on:click={onClose}>✕</button>
    </div>
    <div class="modal-body">

      <div class="settings-section">
        <span class="settings-section-title">Appearance</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Image layout</span>
            <span class="settings-hint">How search results are displayed</span>
          </div>
          <div class="toggle-group small">
            <button
              class="toggle-btn {settings.image_layout === 'grid' ? 'active' : ''}"
              on:click={() => onImageLayoutChange('grid')}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
              Grid
            </button>
            <button
              class="toggle-btn {settings.image_layout === 'masonry' ? 'active' : ''}"
              on:click={() => onImageLayoutChange('masonry')}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="10"/><rect x="14" y="3" width="7" height="6"/><rect x="3" y="16" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/></svg>
              Masonry
            </button>
          </div>
        </div>

        <div class="settings-row">
          <div class="settings-label">
            <span>Thumbnail size</span>
            <span class="settings-hint">Stored thumbnail resolution in pixels — changing requires re-indexing</span>
          </div>
          <div class="settings-input-wrap">
            <input
              type="number"
              class="settings-number-input"
              min="64"
              max="512"
              step="32"
              value={settings.thumbnail_size}
              on:input={(e) => {
                onThumbnailSizeChange(e.target.value);
                thumbnailSizeChanged = true;
              }}
            />
            <span class="settings-unit">px</span>
            {#if thumbnailSizeChanged}
              <button class="btn btn-primary" on:click={onRegenThumbnails}>
                Regen
              </button>
            {/if}
          </div>
        </div>

        <div class="settings-row">
          <div class="settings-label">
            <span>Excluded file types</span>
            <span class="settings-hint">Comma-separated list (e.g., gif,tiff,svg)</span>
          </div>
          <div class="settings-input-wrap">
            <input
              type="text"
              class="input setting-input"
              value={settings.excluded_file_types || ''}
              placeholder="gif,tiff,svg"
              on:input={handleExcludedFileTypesInput}
            />
          </div>
        </div>
      </div>

      <div class="settings-divider"></div>

      <div class="settings-section">
        <span class="settings-section-title">Models</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Embedding model</span>
            <span class="settings-hint">Changes take effect after app restart. Requires re-indexing.</span>
          </div>
          <div class="settings-input-wrap">
            <div class="custom-select" bind:this={selectContainer}>
              <button 
                class="custom-select-button"
                on:click={() => showDropdown = !showDropdown}
              >
                <span class="selected-text">
                  {availableModels.find(m => m.id === currentModel)?.name || 'Select model'}
                </span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              </button>

              {#if showDropdown}
                <div class="custom-select-dropdown">
                  {#each availableModels as model}
                    <button
                      class="custom-select-option {currentModel === model.id ? 'selected' : ''}"
                      on:click={() => {
                        pendingModel = model.id;
                        if (model.id !== currentModel) {
                          showModelChangeWarning = true;
                        }
                        showDropdown = false;
                      }}
                    >
                      <div class="option-name">{model.name}</div>
                      <div class="option-description">{model.description}</div>
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>

      <div class="settings-divider"></div>

      <div class="settings-section">
        <span class="settings-section-title">Notifications</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Notify on indexing complete</span>
            <span class="settings-hint">Show a system notification when indexing finishes</span>
          </div>
          <label class="custom-checkbox">
            <input
              type="checkbox"
              checked={settings.notify_on_complete === '1'}
              on:change={(e) => onNotifyChange(e.target.checked ? '1' : '0')}
            />
            <div class="checkbox-track">
              <div class="checkbox-thumb"></div>
              <svg class="checkbox-check" viewBox="0 0 10 8" fill="none"><polyline points="1 4 3.5 6.5 9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
          </label>
        </div>
      </div>

      <div class="settings-divider"></div>

      <div class="settings-section">
        <span class="settings-section-title">Deletion</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Prompt before removing from database</span>
            <span class="settings-hint">Show confirmation dialog before removing an image from the index</span>
          </div>
          <label class="custom-checkbox">
            <input
              type="checkbox"
              checked={settings.prompt_delete_db === '1'}
              on:change={(e) => onPromptDbChange(e.target.checked ? '1' : '0')}
            />
            <div class="checkbox-track">
              <div class="checkbox-thumb"></div>
              <svg class="checkbox-check" viewBox="0 0 10 8" fill="none"><polyline points="1 4 3.5 6.5 9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
          </label>
        </div>

        <div class="settings-row">
          <div class="settings-label">
            <span>Prompt before deleting from system</span>
            <span class="settings-hint">Show confirmation dialog before permanently deleting a file</span>
          </div>
          <label class="custom-checkbox">
            <input
              type="checkbox"
              checked={settings.prompt_delete_system === '1'}
              on:change={(e) => onPromptSystemChange(e.target.checked ? '1' : '0')}
            />
            <div class="checkbox-track">
              <div class="checkbox-thumb"></div>
              <svg class="checkbox-check" viewBox="0 0 10 8" fill="none"><polyline points="1 4 3.5 6.5 9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
          </label>
        </div>
      </div>

      <div class="settings-divider"></div>

      <div class="settings-section">
        <span class="settings-section-title">Duplicates</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Find &amp; remove duplicates now</span>
            <span class="settings-hint">Scan the current index for duplicate images and remove them</span>
          </div>
          <button class="btn btn-primary" on:click={onDuplicateFind}>
            Run now
          </button>
        </div>
      </div>

      <div class="settings-divider"></div>

      <div class="settings-section">
        <span class="settings-section-title">Danger Zone</span>

        <div class="settings-row">
          <div class="settings-label">
            <span>Clear entire database</span>
            <span class="settings-hint">Remove all indexed images, embeddings, thumbnails, and feedback from the database. Files on disk are not deleted.</span>
          </div>
          <button class="btn btn-primary danger" on:click={onClearDatabase}>
            Clear Database
          </button>
        </div>
      </div>

    </div>

    {#if showModelChangeWarning}
      <div class="modal-backdrop-overlay" style="position: fixed; inset: 0; z-index: 10;">
        <button 
          type="button" 
          class="backdrop-trigger" 
          on:click={cancelModelChange} 
          aria-label="Cancel model configuration change"
          style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
        ></button>

        <div class="warning-modal" role="dialog" aria-modal="true" aria-label="Switch Model Warning" style="position: relative; z-index: 11;">
          <div class="warning-header">
            <span>⚠️ Switch Model</span>
          </div>
          <div class="warning-body">
            <p>Switching models may require Scout to re-analyse your images — but if you've used this model before, it'll switch instantly.</p>
            <p>Your folders and collections won't be affected.</p>
          </div>
          <div class="warning-actions">
            <button class="btn btn-primary" on:click={cancelModelChange}>
              Cancel
            </button>
            <button class="btn btn-primary danger" on:click={confirmModelChange}>
              Switch Model
            </button>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
.modal-body {
	overflow-y: auto;
	padding: 0;
}

.settings-section {
	padding: 20px 24px;
	display: flex;
	flex-direction: column;
	gap: 18px;
}

.settings-section-title {
	font-family: var(--secondary-font);
	font-size: var(--font-medium);
	font-weight: 600;
	letter-spacing: 2px;
	text-transform: uppercase;
	color: var(--primary-very-light);
}

.settings-row {
	display: flex;
	align-items: center;
	justify-content: space-between;
	gap: 24px;
}

.settings-label {
	display: flex;
	flex-direction: column;
	gap: 3px;
	flex: 1;
	min-width: 0;
}

.setting-input {
    padding-left: 15px;
    padding-right: 15px;
}

.settings-label span:first-child {
	font-size: var(--font-small);
	color: var(--primary-light);
}

.settings-hint {
	font-size: var(--font-very-small);
	color: var(--text-colour);
	line-height: 1.4;
}

.settings-divider {
	display: flex;
	height: 0.5px;
	border: var(--border-small) solid var(--primary-dark);
	margin-left: 20px;
	margin-right: 20px;
	background: var(--primary);
}

.toggle-group {
	display: flex;
	gap: 8px;
}

.toggle-group.small .toggle-btn {
	font-size: var(--font-small);
	padding: 7px 12px;
}

.toggle-btn {
	flex: 1;
	display: flex;
	align-items: center;
	justify-content: center;
	gap: 8px;
	padding: 10px;
	background: transparent;
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 8px;
	color: var(--text-colour);
	cursor: pointer;
	transition: all 0.2s;
}

.toggle-btn.active {
	border-color: var(--primary);
	color: var(--text-colour);
	background: var(--header-background);
}

.toggle-btn:hover:not(.active) {
	border-color: var(--primary-dark);
	color: var(--text-colour);
}

.settings-input-wrap {
	display: flex;
	align-items: center;
	gap: 6px;
	flex-shrink: 0;
}

.settings-number-input {
	width: 72px;
	padding: 6px 10px;
	background: var(--header-background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 7px;
	color: var(--text-colour);
	font-size: var(--font-small);
	outline: none;
	text-align: center;
	transition: border-color 0.2s;
}

.settings-number-input:focus {
	border-color: var(--primary);
}

.settings-number-input::-webkit-inner-spin-button {
	opacity: 0.4;
}

.settings-unit {
	font-size: 12px;
	color: var(--text-colour);
}

.custom-checkbox {
	position: relative;
	flex-shrink: 0;
	margin-top: 1px;
}

.custom-checkbox input {
	position: absolute;
	opacity: 0;
	width: 100%;
	height: 100%;
	cursor: pointer;
	margin: 0;
	z-index: 1;
}

.checkbox-track {
	width: 36px;
	height: 20px;
	border-radius: 10px;
	background: var(--header-background);
	border: 1px solid var(--primary-dark);
	position: relative;
	transition: all 0.25s;
	display: flex;
	align-items: center;
}

.custom-checkbox input:checked~.checkbox-track {
	background: var(--primary);
	border-color: var(--primary);
}

.checkbox-thumb {
	width: 14px;
	height: 14px;
	border-radius: 50%;
	background: var(--primary-very-light);
	position: absolute;
	left: 2px;
	transition: all 0.25s;
	box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.custom-checkbox input:checked~.checkbox-track .checkbox-thumb {
	transform: translateX(16px);
	background: white;
}

.checkbox-check {
	position: absolute;
	width: 10px;
	height: 8px;
	left: 19px;
	opacity: 0;
	transition: opacity 0.2s;
}

.custom-checkbox input:checked~.checkbox-track .checkbox-check {
	opacity: 1;
}

.custom-select {
	position: relative;
	width: 100%;
	min-width: 250px;
}

.custom-select-button {
	display: flex;
	align-items: center;
	justify-content: space-between;
	width: 100%;
	padding: 8px 12px;
	background: var(--header-background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 7px;
	color: var(--text-colour);
	font-size: var(--font-small);
	cursor: pointer;
	transition: border-color 0.2s;
}

.custom-select-button:hover {
	border-color: var(--primary);
}

.custom-select-button:focus {
	outline: none;
	border-color: var(--primary);
}

.selected-text {
	flex: 1;
	text-align: left;
}

.custom-select-dropdown {
	position: absolute;
	top: 100%;
	left: 0;
	right: 0;
	background: var(--header-background);
	border: var(--border-small) solid var(--primary-dark);
	border-top: none;
	border-radius: 0 0 7px 7px;
	margin-top: -1px;
	z-index: 100;
	max-height: 300px;
	overflow-y: auto;
	box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.custom-select-option {
	display: block;
	width: 100%;
	padding: 12px;
	background: transparent;
	border: none;
	border-bottom: var(--border-small) solid var(--primary-dark);
	color: var(--text-colour);
	text-align: left;
	cursor: pointer;
	transition: background-color 0.2s;
}

.custom-select-option:last-child {
	border-bottom: none;
}

.custom-select-option:hover {
	background: rgba(124, 58, 237, 0.1);
}

.custom-select-option.selected {
	background: rgba(124, 58, 237, 0.2);
	border-left: 2px solid var(--primary);
	padding-left: 10px;
}

.option-name {
	font-weight: 700;
	color: var(--primary-light);
	margin-bottom: 4px;
}

.option-description {
	font-size: 11px;
	color: var(--text-colour);
	line-height: 1.4;
	word-wrap: break-word;
	white-space: normal;
}

.modal-backdrop-overlay {
	position: fixed;
	inset: 0;
	background: rgba(0, 0, 0, 0.5);
	display: flex;
	align-items: center;
	justify-content: center;
	z-index: 300;
}

.warning-modal {
	background: var(--header-background);
	border: 1px solid var(--primary-dark);
	border-radius: 12px;
	max-width: 400px;
	box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.warning-header {
	padding: 16px 20px;
	border-bottom: 1px solid var(--primary-dark);
	font-weight: 600;
	color: var(--primary-light);
}

.warning-body {
	padding: 16px 20px;
	color: var(--text-colour);
	font-size: var(--font-small);
	line-height: 1.5;
}

.warning-body p {
	margin: 8px 0;
}

.warning-actions {
	padding: 12px 20px 16px;
	display: flex;
	gap: 8px;
	justify-content: flex-end;
}

@media (max-width: 680px) {
	.settings-row {
		flex-direction: column;
		align-items: flex-start;
		gap: 10px;
	}
}
</style>