<script>
  export let indexMode = 'folder';
  export let indexPath = '';
  export let indexRecursive = true;
  export let autoIndex = false;
 
  export let onIndexModeChange = () => {};
  export let onPickPath = () => {};
  export let onRecursiveChange = () => {};
  export let onAutoIndexChange = () => {};
  export let onStartIndexing = () => {};
  export let onClose = () => {};
</script>
 
<svelte:window on:keydown={(e) => { if (e.key === 'Escape') onClose(); }} />
 
<div class="modal-backdrop" style="position: fixed; inset: 0;">
  <button 
    type="button" 
    class="backdrop-trigger" 
    on:click={onClose} 
    aria-label="Close indexing dialog"
    style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
  ></button>
 
  <div class="modal index-modal" role="dialog" aria-modal="true" aria-label="Index Images" style="position: relative; z-index: 1;">
    <div class="modal-header">
      <span class="modal-title">Index Images</span>
      <button class="modal-close" on:click={onClose}>✕</button>
    </div>
    <div class="modal-body">
      <div class="toggle-group" role="group" aria-label="Index mode">
        <button class="btn btn-primary toggle-btn {indexMode === 'folder' ? 'active' : ''}" on:click={() => onIndexModeChange('folder')}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
          Folder
        </button>
        <button class="btn btn-primary toggle-btn {indexMode === 'file' ? 'active' : ''}" on:click={() => onIndexModeChange('file')}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
          Single File
        </button>
      </div>
 
      <div class="pick-row">
        <div class="picked-path" title={indexPath}>
          {#if indexPath}
            <span class="path-text">{indexPath}</span>
          {:else}
            <span class="path-placeholder">No {indexMode === 'folder' ? 'folder' : 'file'} selected</span>
          {/if}
        </div>
        <button class="btn btn-primary" on:click={onPickPath}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
          Browse...
        </button>
      </div>
 
      <div class="drop-zone" role="region" aria-label="Drop zone hint">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span>Or drag and drop anywhere on the app</span>
      </div>
 
      {#if indexMode === 'folder'}
        <div class="options">
          <label class="option-label" for="recursive-check">
            <div class="custom-checkbox">
              <input id="recursive-check" type="checkbox" checked={indexRecursive} on:change={(e) => onRecursiveChange(e.target.checked)} />
              <div class="checkbox-track">
                <div class="checkbox-thumb"></div>
                <svg class="checkbox-check" viewBox="0 0 10 8" fill="none"><polyline points="1 4 3.5 6.5 9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </div>
            </div>
            <div>
              <span>Include subfolders</span>
              <span class="option-hint">Recursively index all images inside nested folders</span>
            </div>
          </label>
          <label class="option-label" for="auto-index-check">
            <div class="custom-checkbox">
              <input id="auto-index-check" type="checkbox" checked={autoIndex} on:change={(e) => onAutoIndexChange(e.target.checked)} />
              <div class="checkbox-track">
                <div class="checkbox-thumb"></div>
                <svg class="checkbox-check" viewBox="0 0 10 8" fill="none"><polyline points="1 4 3.5 6.5 9 1" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </div>
            </div>
            <div>
              <span>Auto-index new images</span>
              <span class="option-hint">Automatically index new images found on app start</span>
            </div>
          </label>
        </div>
      {/if}
 
      <div class="modal-footer">
        <button class="btn btn-primary" on:click={onClose}>Cancel</button>
        <button class="btn btn-primary" on:click={onStartIndexing} disabled={!indexPath}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Start Indexing
        </button>
      </div>
    </div>
  </div>
</div>
 
<style>
.toggle-group {
	display: flex;
	gap: 8px;
}
 
.toggle-btn {
	flex: 1;
	justify-content: center;
}
 
.pick-row {
	display: flex;
	gap: 10px;
	align-items: center;
	padding-bottom: 20px;
}
 
.picked-path {
	flex: 1;
	display: flex;
	align-items: center;
	padding: 10px 14px;
	background: var(--background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 8px;
	min-width: 0;
	overflow: hidden;
	cursor: default;
}
 
.path-text {
	font-size: var(--font-small);
	color: var(--primary-very-light);
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
	width: 100%;
	direction: rtl;
	text-align: left;
}
 
.path-placeholder {
	font-size: var(--font-small);
	color: var(--primary-very-light);
}
 
.drop-zone {
	border: var(--border-small) dashed var(--primary-dark);
	border-radius: 8px;
	padding: 20px;
	display: flex;
	align-items: center;
	justify-content: center;
	gap: 10px;
	color: var(--primary-light);
	font-size: var(--font-very-small);
}
 
.options {
	display: flex;
	flex-direction: column;
	gap: 14px;
	padding-top: 20px;
}
 
.option-label {
	display: flex;
	align-items: flex-start;
	gap: 12px;
	font-size: var(--font-small);
	color: var(--primary-light);
	cursor: pointer;
	user-select: none;
}
 
.option-label div {
	display: flex;
	flex-direction: column;
	gap: 3px;
}
 
.option-hint {
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
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
	background: var(--primary-dark);
	border: 1px solid var(--primary);
	position: relative;
	transition: all 0.55s;
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
	background: var(--header-background);
	position: absolute;
	left: 2px;
	transition: all 0.55s;
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
	opacity: 0 !important;
	transition: opacity 0.2s;
}
 
.custom-checkbox input:checked~.checkbox-track .checkbox-check {
	opacity: 1;
}
 
.modal-footer {
	display: flex;
	justify-content: flex-end;
	gap: 8px;
	padding-top: 16px;
	border-top: var(--border-small) solid var(--primary-dark);
	margin-top: 2px;
}
 
@media (max-width: 680px) {
	.index-modal {
		width: calc(100vw - 32px);
	}
}
</style>