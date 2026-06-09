<script>
  import { convertFileSrc } from '@tauri-apps/api/core';
 
  export let duplicateGroups = [];
  export let settings = {};
 
  export let onRemove = () => {};
  export let onClose = () => {};
</script>
 
<svelte:window on:keydown={(e) => { 
  if (e.key === 'Escape' && duplicateGroups && duplicateGroups.length > 0) { 
    onClose(); 
  } 
}} />
 
{#if duplicateGroups && duplicateGroups.length > 0}
  <div class="modal-backdrop" style="position: fixed; inset: 0;">
    <button 
      type="button" 
      class="backdrop-trigger" 
      on:click={onClose} 
      aria-label="Close duplicates dialog"
      style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
    ></button>
 
    <div class="modal duplicates-modal" role="dialog" aria-modal="true" aria-label="Duplicates found" style="position: relative; z-index: 1;">
      <div class="modal-header">
        <span class="modal-title">Duplicates Found</span>
        <button class="modal-close" on:click={onClose}>✕</button>
      </div>
      <div class="modal-body">
        <div class="dup-summary">
          <span class="dup-summary-count">
            {duplicateGroups.length} group{duplicateGroups.length !== 1 ? 's' : ''} found
            — {duplicateGroups.reduce((a, g) => a + g.length - 1, 0)} images will be removed
          </span>
          <span class="dup-summary-mode {settings.deduplicate_mode === 'system' ? 'danger' : ''}">
            {#if settings.deduplicate_mode === 'system'}
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
              Will permanently delete files from your system
            {:else}
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>
              Will remove from index only — files stay on your system
            {/if}
          </span>
        </div>
 
        <div class="dup-list">
          {#each duplicateGroups.slice(0, 8) as group}
            <div class="dup-group">
              <div class="dup-group-images">
                {#each group as path, gi}
                  <div class="dup-img-wrap {gi === 0 ? 'dup-keep' : 'dup-remove'}">
                    <img
                      src={convertFileSrc(path)}
                      alt={path.split('\\').pop()}
                    />
                    <span class="dup-img-label">{gi === 0 ? 'Keep' : 'Remove'}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/each}
          {#if duplicateGroups.length > 8}
            <p class="settings-hint" style="padding: 8px 0; text-align:center">
              ...and {duplicateGroups.length - 8} more groups not shown
            </p>
          {/if}
        </div>
 
        <div class="modal-footer">
          <button class="btn btn-primary" on:click={onClose}>Cancel</button>
          <button class="btn btn-primary" on:click={onRemove}>Remove from Index</button>
        </div>
      </div>
    </div>
  </div>
{/if}
 
<style>
.dup-summary {
	display: flex;
	flex-direction: column;
	gap: 6px;
	padding: 12px 14px;
	background: var(--background);
	border-radius: 8px;
	border: var(--border-small) solid var(--primary-dark);
}
 
.dup-summary-count {
	font-size: var(--font-small);
	color: var(--primary-light);
}
 
.dup-summary-mode {
	display: flex;
	align-items: center;
	gap: 6px;
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
}
 
.dup-summary-mode.danger {
	color: var(--primary-warn);
}
 
.dup-list {
	display: flex;
	flex-direction: column;
	gap: 12px;
}
 
.dup-group {
	background: var(--background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 8px;
	padding: 10px;
}
 
.dup-group-images {
	display: flex;
	gap: 8px;
	flex-wrap: wrap;
}
 
.dup-img-wrap {
	position: relative;
	width: 80px;
	height: 80px;
	border-radius: 6px;
	overflow: hidden;
	flex-shrink: 0;
}
 
.dup-img-wrap img {
	width: 100%;
	height: 100%;
	object-fit: cover;
	display: block;
}
 
.dup-img-wrap.dup-keep {
	border: var(--border-medium) solid var(--primary-success);
}
 
.dup-img-wrap.dup-remove {
	border: var(--border-medium) solid var(--primary-danger);
	opacity: 0.7;
}
 
.dup-img-label {
	position: absolute;
	bottom: 0;
	left: 0;
	right: 0;
	font-size: 9px;
	text-align: center;
	padding: 2px;
	font-family: var(--secondary-font);
	font-weight: 600;
	letter-spacing: 1px;
	text-transform: uppercase;
}
 
.dup-keep .dup-img-label {
	background: rgba(74, 222, 128, 0.8);
	color: var(--primary-dark-success);
}
 
.dup-remove .dup-img-label {
	background: rgba(248, 113, 113, 0.8);
	color: var(--primary-dark-danger);
}
 
.modal-footer {
	display: flex;
	justify-content: flex-end;
	gap: 8px;
	padding-top: 16px;
	border-top: var(--border-small) solid var(--primary-dark);
	margin-top: 2px;
}
 
.settings-hint {
	font-size: 11px;
	color: var(--primary-light);
	line-height: 1.4;
}
 
@media (max-width: 680px) {
	.duplicates-modal {
		width: calc(100vw - 32px);
	}
}
</style>