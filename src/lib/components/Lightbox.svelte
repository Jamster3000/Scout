<script>
  import { convertFileSrc } from '@tauri-apps/api/core';
  import { onDestroy } from 'svelte';
 
  export let imagePath = '';
  export let score = 0;
  export let zoomLevel = 1;
  export let zoomX = 50;
  export let zoomY = 50;
  export let rawPreview = null;
 
  export let onWheel = () => {};
  export let onMouseMove = () => {};
  export let onClose = () => {};
  export let onResetZoom = () => {};
  export let onAddToCollection = () => {};
 
  let blobUrl = null;
 
  $: {
    if (blobUrl) {
      URL.revokeObjectURL(blobUrl);
      blobUrl = null;
    }
	console.log(rawPreview);
    if (rawPreview && (rawPreview.length > 0 || Object.keys(rawPreview).length > 0)) {
      const bytes = rawPreview instanceof Uint8Array
		  ? rawPreview
		  : new Uint8Array(Object.values(rawPreview));
		blobUrl = URL.createObjectURL(
		  new Blob([bytes], { type: 'image/jpeg' })
		);
    }
  }
 
  onDestroy(() => {
    if (blobUrl) URL.revokeObjectURL(blobUrl);
  });
 
  $: displaySrc = blobUrl ?? convertFileSrc(imagePath);
</script>
 
<svelte:window on:keydown={(e) => { if (e.key === 'Escape') onClose(); }} />
 
<div class="modal-backdrop" style="position: fixed; inset: 0;">
  <button 
    type="button" 
    class="backdrop-trigger" 
    on:click={onClose} 
    aria-label="Close image preview"
    style="position: absolute; inset: 0; background: transparent; border: none; padding: 0; margin: 0; width: 100%; height: 100%; cursor: default;"
  ></button>
 
  <div class="lightbox-inner" role="dialog" aria-modal="true" aria-label="Image preview" style="position: relative; z-index: 1;">
    <button
      type="button"
      class="lightbox-img-wrap"
      on:wheel|preventDefault={onWheel}
      on:mousemove={onMouseMove}
      style="cursor: {zoomLevel > 1 ? 'zoom-out' : 'zoom-in'}; background: transparent; border: none; padding: 0; display: block;"
      aria-label="Zoomable image, click to reset zoom"
      on:click={() => { if (zoomLevel > 1) onResetZoom(); }}
    >
      <img
        src={displaySrc}
        alt={imagePath.split('\\').pop()}
        style="transform: scale({zoomLevel}); transform-origin: {zoomX}% {zoomY}%; display: block;"
      />
    </button>
    <div class="lightbox-meta">
      <span class="lightbox-name">{imagePath.split('\\').pop()}</span>
      <span class="lightbox-path">{imagePath}</span>
      <span class="lightbox-score">Match: {(score * 100).toFixed(1)}%</span>
      <span class="lightbox-zoom">{Math.round(zoomLevel * 100)}%</span>
      <button class="btn btn-secondary" on:click={onAddToCollection} title="Add to collection">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/><path d="M7 10h10M7 14h10M7 7h10"/></svg>
      </button>
    </div>
    <button class="lightbox-close" on:click={onClose}>✕</button>
  </div>
</div>
 
<style>
.lightbox-inner {
	position: relative;
	max-width: 85vw;
	max-height: 90vh;
	display: flex;
	flex-direction: column;
	gap: 12px;
}
 
.lightbox-img-wrap {
	overflow: hidden;
	border-radius: 8px;
	border: var(--border-medium) solid var(--primary-dark);
	max-height: 75vh;
	display: flex;
	align-items: center;
	justify-content: center;
	background: var(--header-background);
	padding: 0;
	width: 100%;
}
 
.lightbox-img-wrap img {
	max-width: 100%;
	max-height: 75vh;
	object-fit: contain;
	display: block;
	transition: transform 0.1s ease;
	pointer-events: none;
	user-select: none;
}
 
.lightbox-meta {
	display: flex;
	gap: 16px;
	align-items: center;
	padding: 0 4px;
	flex-wrap: wrap;
}
 
.lightbox-name {
	font-size: 13px;
	color: var(--primary-very-light);
	font-weight: 500;
}
 
.lightbox-path {
	font-size: var(--font-very-small);
	color: var(--primary-very-light);
	flex: 1;
	overflow: hidden;
	text-overflow: ellipsis;
	white-space: nowrap;
	min-width: 0;
}
 
.lightbox-score {
	font-family: var(--secondary-font);
	font-size: 14px;
	font-weight: 600;
	color: var(--primary-light);
}
 
.lightbox-zoom {
	font-family: var(--secondary-font);
	font-size: var(--font-small);
	color: var(--primary-light);
	min-width: 36px;
	text-align: right;
}
 
.lightbox-close {
	position: absolute;
	top: -12px;
	right: -12px;
	width: 26px;
	height: 26px;
	border-radius: 50%;
	background: var(--header-background);
	border: var(--border-medium) solid var(--primary-dark);
	color: var(--primary-very-light);
	cursor: pointer;
	font-size: var(--font-very-small);
	display: flex;
	align-items: center;
	justify-content: center;
	transition: all 0.2s;
}
 
.lightbox-close:hover {
	border-color: var(--primary);
	color: var(--primary-light);
}
 
@media (max-width: 480px) {
	.lightbox-inner {
		max-width: 95vw;
	}
 
	.lightbox-meta {
		gap: 8px;
	}
 
	.lightbox-path {
		display: none;
	}
}
</style>