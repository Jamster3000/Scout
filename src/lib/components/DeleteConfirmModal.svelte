<script>
  export let deleteType = null; //This can be `system`(delete from system) or `db`(delete from database)
  export let pendingDeletePath = '';
  export let pendingDeletePaths = [];
 
  export let onConfirm = () => {};
  export let onCancel = () => {};
</script>
 
<svelte:window on:keydown={(e) => { 
  if (e.key === 'Escape' && deleteType && pendingDeletePath) { 
    onCancel(); 
  } 
}} />
 
{#if deleteType && pendingDeletePath}
  <div class="modal-backdrop" style="position: fixed; inset: 0;">
    <button 
      type="button" 
      class="backdrop-trigger" 
      on:click={onClose} 
      aria-label="Close confirmation dialog"
      style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
    ></button>
 
    <div class="modal confirm-modal" role="dialog" aria-modal="true" aria-label="Confirm delete" style="position: relative; z-index: 1;">
      <div class="modal-header">
        <span class="modal-title" style="color: #f87171">
          {deleteType === 'system' ? 'Delete from system' : 'Delete from database'}
        </span>
        <button class="modal-close" on:click={onCancel}>✕</button>
      </div>
      <div class="modal-body">
        <p class="confirm-text">
          {#if deleteType === 'system'}
            This will permanently delete the file from your storage. This cannot be undone.
          {:else}
            This will remove the image from Scout's index. The file will remain on your system.
          {/if}
        </p>
        <p class="confirm-path">
          {#if pendingDeletePaths && pendingDeletePaths.length > 1}
            {pendingDeletePaths.length} files selected
          {:else}
            {pendingDeletePath}
          {/if}
        </p>
        <div class="modal-footer">
          <button class="btn btn-primary" on:click={onCancel}>Cancel</button>
          <button
            class="btn btn-primary danger"
            on:click={onConfirm}
          >
            {deleteType === 'system' ? 'Delete file' : 'Remove from index'}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}
 
<style>
.confirm-text {
	font-size: 13px;
	color: var(--primary-very-light);
	line-height: 1.6;
}
 
.confirm-path {
	font-size: var(--font-very-small);
	color: var(--primary-light);
	padding: 8px 12px;
	background: var(--background);
	border-radius: 6px;
	border: var(--border-small) solid var(--primary-dark);
	word-break: break-all;
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
 
	.modal,
	.confirm-modal {
		width: calc(100vw - 32px);
	}
}
</style>