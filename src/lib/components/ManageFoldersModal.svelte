<script>
  export let folders = [];
  export let loading = false;

  export let onToggleAutoIndex = () => {};
  export let onUnindexFolder = () => {};
  export let onDeleteFromSystem = () => {};
  export let onAddFolder = () => {};
  export let onClose = () => {};

  let confirmAction = null;
  let expandedPath = null;

  function formatDate(unix) {
	  if (!unix) return 'Never';
	  const d = new Date(unix * 1000);
	  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function basename(path) {
	  return path.replace(/\\/g, '/').split('/').pop() || path;
  }

  function requestConfirm(type, folder) {
	  confirmAction = { type, folder };
  }

  function cancelConfirm() {
	  confirmAction = null;
  }

  function executeConfirm() {
    if (!confirmAction) return;
    if (confirmAction.type === 'unindex') onUnindexFolder(confirmAction.folder.path);
    else if (confirmAction.type === 'delete') onDeleteFromSystem(confirmAction.folder.path);
    confirmAction = null;
  }
</script>

<svelte:window on:keydown={(e) => { 
  if (e.key === 'Escape') { 
    if (confirmAction) cancelConfirm(); else onClose(); 
  } 
}} />

<div class="modal-backdrop" style="position: fixed; inset: 0;">
  <button 
    type="button" 
    class="backdrop-trigger" 
    on:click={onClose} 
    aria-label="Close dialog"
    style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
  ></button>

  <div class="modal manage-modal" role="dialog" aria-modal="true" aria-label="Manage Folders" style="position: relative; z-index: 1;">
 
    <div class="modal-header">
      <div class="header-left">
        <span class="modal-title">Manage Folders</span>
        <span class="folder-count">{folders.length} folder{folders.length !== 1 ? 's' : ''}</span>
      </div>
      <div class="header-actions">
        <button class="btn btn-primary" on:click={onAddFolder}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Add Folder
        </button>
        <button class="modal-close" on:click={onClose}>✕</button>
      </div>
    </div>
 
    <div class="modal-body">
      {#if loading}
        <div class="empty-state">
          <div class="mini-radar">
            <div class="mini-ring mr1"></div>
            <div class="mini-ring mr2"></div>
            <div class="mini-sweep"></div>
            <div class="mini-dot"></div>
          </div>
          <span>Loading folders...</span>
        </div>
      {:else if folders.length === 0}
        <div class="empty-state">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
          <span class="empty-text">No folders indexed yet</span>
          <span class="empty-hint">Add a folder using the Index Images button or the Add Folder button above</span>
        </div>
      {:else}
        <div class="folder-list">
          {#each folders as folder}
            {@const isExpanded = expandedPath === folder.path}
            <div class="folder-card" class:expanded={isExpanded}>
 
              <div class="folder-row">
                <button
                  class="folder-expand-btn"
                  aria-label={isExpanded ? 'Collapse' : 'Expand'}
                  on:click={() => expandedPath = isExpanded ? null : folder.path}
                >
                  <svg
                    class="chevron"
                    class:open={isExpanded}
                    width="18" height="18" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" stroke-width="2.5"
                  ><polyline points="9 18 15 12 9 6"/></svg>
                </button>
 
                <div class="folder-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                </div>
 
                <div class="folder-info">
                  <span class="folder-name" title={folder.path}>{basename(folder.path)}</span>
                  <span class="folder-path" title={folder.path}>{folder.path}</span>
                </div>
 
                <div class="folder-stats">
                  <span class="stat-pill">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                    {folder.image_count ?? 0}
                  </span>
                </div>
 
                <div class="folder-controls">
                  <label class="auto-index-toggle" title="Auto-index new images">
                    <input
                      type="checkbox"
                      checked={folder.auto_index}
                      on:change={(e) => onToggleAutoIndex(folder.path, e.target.checked)}
                    />
                    <div class="toggle-track">
                      <div class="toggle-thumb"></div>
                    </div>
                    <span class="toggle-label">Auto</span>
                  </label>
 
                  <button
                    class="icon-btn warn"
                    title="Remove all images from index (files stay on disk)"
                    on:click={() => requestConfirm('unindex', folder)}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
                  </button>
 
                  <button
                    class="icon-btn danger"
                    title="Delete all image files from disk permanently"
                    on:click={() => requestConfirm('delete', folder)}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
                  </button>
                </div>
              </div>
 
              {#if isExpanded}
                <div class="folder-details">
                  <div class="detail-row">
                    <span class="detail-label">Full path</span>
                    <span class="detail-value monospace">{folder.path}</span>
                  </div>
                  <div class="detail-row">
                    <span class="detail-label">Last indexed</span>
                    <span class="detail-value">{formatDate(folder.last_indexed)}</span>
                  </div>
                  <div class="detail-row">
                    <span class="detail-label">Images indexed</span>
                    <span class="detail-value accent">{folder.image_count ?? 0} image{folder.image_count !== 1 ? 's' : ''}</span>
                  </div>
                  {#if folder.last_image_path}
                    <div class="detail-row">
                      <span class="detail-label">Most recently indexed</span>
                      <span class="detail-value monospace dim" title={folder.last_image_path}>
                        {folder.last_image_path.replace(/\\/g, '/').split('/').pop()}
                      </span>
                    </div>
                  {/if}
                  <div class="detail-row">
                    <span class="detail-label">Auto-index</span>
                    <span class="detail-value {folder.auto_index ? 'green' : 'dim'}">
                      {folder.auto_index ? 'Enabled — watching for new images' : 'Disabled'}
                    </span>
                  </div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>
 
{#if confirmAction}
  <div class="confirm-overlay" style="position: fixed; inset: 0;">
    <button 
      type="button" 
      class="backdrop-trigger" 
      on:click={cancelConfirm} 
      aria-label="Cancel action"
      style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
    ></button>

    <div class="confirm-box" role="dialog" aria-modal="true" aria-label="Confirm action" style="position: relative; z-index: 1;">
      <div class="confirm-icon {confirmAction.type === 'delete' ? 'danger' : 'warn'}">
        {#if confirmAction.type === 'delete'}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
        {:else}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
        {/if}
      </div>
 
      <div class="confirm-content">
        <span class="confirm-title">
          {confirmAction.type === 'delete' ? 'Delete files from system?' : 'Remove from index?'}
        </span>
        <p class="confirm-desc">
          {#if confirmAction.type === 'delete'}
            This will <strong>permanently delete all image files</strong> inside this folder from your disk and remove them from the index. <strong>This cannot be undone.</strong>
          {:else}
            This will remove all images inside this folder from Scout's index. The files will remain on your disk.
          {/if}
        </p>
        <div class="confirm-path">{confirmAction.folder.path}</div>
        <div class="confirm-count">
          Affects <strong>{confirmAction.folder.image_count ?? 0}</strong> indexed image{confirmAction.folder.image_count !== 1 ? 's' : ''}
        </div>
      </div>
 
      <div class="confirm-actions">
        <button class="btn btn-primary" on:click={cancelConfirm}>Cancel</button>
        <button
          class="btn {confirmAction.type === 'delete' ? 'danger' : 'warn'}"
          on:click={executeConfirm}
        >
          {confirmAction.type === 'delete' ? 'Delete files' : 'Remove from index'}
        </button>
      </div>
    </div>
  </div>
{/if}
 
<style>
.confirm-overlay {
	position: fixed;
	inset: 0;
	background: rgba(4, 4, 10, 0.75);
	display: flex;
	align-items: center;
	justify-content: center;
	z-index: 200;
	backdrop-filter: blur(4px);
}

.manage-modal {
	width: 660px;
	max-height: 82vh;
	display: flex;
	flex-direction: column;
}

.folder-count {
	font-size: var(--font-very-small);
	color: var(--primary-light);
	background: var(--primary-dark);
	padding: 3px 8px;
	border-radius: 10px;
	border: var(--border-small) solid var(--primary-dark);
}

.header-actions {
	display: flex;
	align-items: center;
	gap: 8px;
}

.empty-state {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	gap: 12px;
	padding: 48px 24px;
	color: var(--primary-very-light);
}

.empty-text {
	font-size: var(--font-medium);
	color: var(--primary-very-light);
}

.empty-hint {
	font-size: var(--font-small);
	color: var(--primary-very-light);
	text-align: center;
	max-width: 340px;
	line-height: 1.6;
}

.folder-list {
	display: flex;
	flex-direction: column;
	gap: 6px;
}

.folder-card {
	background: var(--background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 10px;
	overflow: hidden;
	transition: border-color 0.2s;
}

.folder-row {
	display: flex;
	align-items: center;
	gap: 10px;
	padding: 11px 14px;
}

.folder-expand-btn {
	background: none;
	border: none;
	color: var(--text-colour);
	cursor: pointer;
	padding: 2px;
	flex-shrink: 0;
	display: flex;
	align-items: center;
	transition: color 0.2s;
}

.folder-expand-btn:hover {
	color: var(--primary-very-light);
}

.chevron {
	transition: transform 0.2s;
}

.chevron.open {
	transform: rotate(90deg);
}

.folder-icon {
	color: var(--primary-very-light);
	flex-shrink: 0;
	display: flex;
	align-items: center;
}

.folder-info {
	flex: 1;
	min-width: 0;
	display: flex;
	flex-direction: column;
	gap: 2px;
}

.folder-name {
	font-size: var(--font-small);
	color: var(--primary-light);
	font-weight: 500;
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
}

.folder-path {
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
	white-space: nowrap;
	overflow: hidden;
	text-overflow: ellipsis;
	font-family: var(--font-primary);
}

.folder-stats {
	flex-shrink: 0;
}

.stat-pill {
	display: flex;
	align-items: center;
	gap: 4px;
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
	background: var(--primary-dark);
	border: var(--border-small) solid var(--primary-dark);
	padding: 3px 8px;
	border-radius: 10px;
}

.folder-controls {
	display: flex;
	align-items: center;
	gap: 6px;
	flex-shrink: 0;
}

.auto-index-toggle {
	display: flex;
	align-items: center;
	gap: 6px;
	cursor: pointer;
	user-select: none;
}

.auto-index-toggle input {
	position: absolute;
	opacity: 0;
	width: 0;
	height: 0;
	pointer-events: none;
}

.toggle-track {
	width: 30px;
	height: 17px;
	border-radius: 9px;
	background: var(--primary-dark);
	border: var(--border-small) solid var(--primary-dark);
	position: relative;
	transition: all 0.25s;
	flex-shrink: 0;
}

.auto-index-toggle input:checked~.toggle-track {
	background: var(--primary);
	border-color: var(--primary);
}

.toggle-thumb {
	width: 11px;
	height: 11px;
	border-radius: 50%;
	background: #4a4870;
	position: absolute;
	left: 2px;
	top: 2px;
	transition: all 0.25s;
}

.auto-index-toggle input:checked~.toggle-track .toggle-thumb {
	transform: translateX(13px);
	background: white;
}

.toggle-label {
	font-size: 10px;
	color: var(--primary-very-light);
	letter-spacing: 0.5px;
}

.icon-btn {
	width: 28px;
	height: 28px;
	border-radius: 6px;
	background: transparent;
	border: var(--border-small) solid var(--primary-dark);
	display: flex;
	align-items: center;
	justify-content: center;
	cursor: pointer;
	transition: all 0.2s;
	flex-shrink: 0;
}

.icon-btn.warn {
	color: var(--primary-warn);
}

.icon-btn.warn:hover {
	border-color: var(--primary-warn);
	background: var(--background-transparent);
}

.icon-btn.danger {
	color: var(--primary-danger);
}

.icon-btn.danger:hover {
	border-color: var(--primary-danger);
	background: var(--background-transparent);
}

.folder-details {
	border-top: var(--border-small) solid var(--primary-dark);
	padding: 12px 14px 14px 38px;
	display: flex;
	flex-direction: column;
	gap: 8px;
	background: var(--background);
}

.detail-row {
	display: flex;
	align-items: flex-start;
	gap: 16px;
	font-size: var(--font-small);
}

.detail-label {
	color: var(--primary-very-light);
	min-width: 150px;
	flex-shrink: 0;
	padding-top: 1px;
	font-size: var(--font-very-small);
	letter-spacing: 0.3px;
}

.detail-value {
	color: var(--primary-very-light);
	flex: 1;
	word-break: break-all;
	line-height: 1.5;
}

.detail-value.monospace {
	font-family: var(--font-primary);
	font-size: var(--font-very-small);
}

.detail-value.accent {
	color: var(--primary-light);
}

.detail-value.green {
	color: var(--primary-success);
}

.detail-value.dim {
	color: var(--primary);
}

.confirm-box {
	background: var(--background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 12px;
	width: 440px;
	padding: 24px;
	display: flex;
	flex-direction: column;
	gap: 18px;
	box-shadow: 0 32px 80px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(239, 68, 68, 0.1);
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

.confirm-icon.warn {
	background: rgba(251, 146, 60, 0.15);
	color: var(--primary-warn);
	border: var(--border-small) solid rgba(251, 146, 60, 0.3);
}

.confirm-icon.danger {
	background: var(--header-background);
	color: var(--primary-danger);
	border: var(--border-small) solid rgba(239, 68, 68, 0.3);
}

.confirm-content {
	display: flex;
	flex-direction: column;
	gap: 10px;
}

.confirm-title {
	font-family: var(--primary-font);
	font-size: var(--font-medium);
	font-weight: 600;
	color: var(--text-colour);
	letter-spacing: 1px;
}

.confirm-desc {
	font-size: var(--font-small);
	color: var(--text-colour);
	line-height: 1.65;
}

.confirm-desc strong {
	color: var(--text-colour);
}

.confirm-path {
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
	background: var(--header-background);
	border: var(--border-small) solid var(--primary-dark);
	padding: 7px 10px;
	border-radius: 6px;
	word-break: break-all;
}

.confirm-count {
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
}

.confirm-count strong {
	color: var(--primary-light);
}

.confirm-actions {
	display: flex;
	justify-content: flex-end;
	gap: 8px;
}

.mini-radar {
	position: relative;
	width: 32px;
	height: 32px;
}

.mini-ring {
	position: absolute;
	border-radius: 50%;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	border: 1px solid rgba(124, 58, 237, 0.5);
}

.mr1 {
	width: 32px;
	height: 32px;
}

.mr2 {
	width: 18px;
	height: 18px;
	border-color: rgba(124, 58, 237, 0.8);
}

.mini-sweep {
	position: absolute;
	top: 50%;
	left: 50%;
	width: 16px;
	height: 1.5px;
	transform-origin: left center;
	background: linear-gradient(90deg, rgba(124, 58, 237, 1), transparent);
	animation: sweep 1.5s linear infinite;
}

.mini-dot {
	position: absolute;
	width: 4px;
	height: 4px;
	background: var(--primary);
	border-radius: 50%;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	box-shadow: 0 0 6px var(--primary);
}

@media (max-width: 700px) {
	.manage-modal {
		width: calc(100vw - 32px);
	}

	.confirm-box {
		width: calc(100vw - 48px);
	}

	.folder-path {
		display: none;
	}

	.folder-stats {
		display: none;
	}

	.detail-label {
		min-width: 110px;
	}
}
</style>