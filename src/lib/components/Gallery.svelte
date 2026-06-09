<script>
  import { convertFileSrc } from '@tauri-apps/api/core';
  import { onDestroy } from 'svelte';

  export let results = [];
  export let count = 0;
  export let selectedPaths = new Set();
  export let feedbackFlash = new Map();
  export let deletingPaths = new Set();
  export let contextMenuPath = null;
  export let settings = {};

  export let getThumbUrl = (path, bytes) => null;

  export let onCardClick = () => {};
  export let onCardRightClick = () => {};
  export let onScroll = () => {};
  export let onGridClick = () => {};

  let gridEl;
  let resizeObserver;

  function layoutMasonry() {
    if (!gridEl || settings.image_layout !== 'masonry') return;

    const containerWidth = gridEl.clientWidth;
    const gap = 12;
    const minColWidth = 200;
    const colCount = Math.max(1, Math.floor((containerWidth + gap) / (minColWidth + gap)));
    const colWidth = (containerWidth - gap * (colCount - 1)) / colCount;

    const colHeights = new Array(colCount).fill(0);
    const cards = gridEl.querySelectorAll('.card');

    cards.forEach((card, i) => {
      const ratio = results[i] ? (results[i][2] ?? 1) : 1;
      const cardHeight = colWidth / ratio;

      // Find shortest column
      const shortestCol = colHeights.indexOf(Math.min(...colHeights));
      const x = shortestCol * (colWidth + gap);
      const y = colHeights[shortestCol];

      card.style.position = 'absolute';
      card.style.left = `${x}px`;
      card.style.top = `${y}px`;
      card.style.width = `${colWidth}px`;
      card.style.height = `${cardHeight}px`;
      card.style.aspectRatio = 'unset';

      colHeights[shortestCol] += cardHeight + gap;
    });

    // Set grid container height to tallest column
    gridEl.style.height = `${Math.max(...colHeights)}px`;
  }

  function setupObserver() {
    if (!gridEl) return;
    resizeObserver = new ResizeObserver(() => {
      if (settings.image_layout === 'masonry') layoutMasonry();
    });
    resizeObserver.observe(gridEl);
  }

  function teardownObserver() {
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
  }

  // Re-layout when results or layout setting changes
  $: if (gridEl && settings.image_layout === 'masonry' && results.length > 0) {
    // Wait a tick for the DOM to update with new cards
    setTimeout(layoutMasonry, 0);
  }

  // Clean up grid styles when switching back to grid layout
  $: if (settings.image_layout !== 'masonry' && gridEl) {
    const cards = gridEl.querySelectorAll('.card');
    cards.forEach(card => {
      card.style.position = '';
      card.style.left = '';
      card.style.top = '';
      card.style.width = '';
      card.style.height = '';
      card.style.aspectRatio = '';
    });
    gridEl.style.height = '';
  }

  onDestroy(teardownObserver);
</script>

<main on:scroll={onScroll}>
  <div class="count-bar">
    <span>{count} images indexed</span>
    {#if results.length > 0}
      <span class="result-count">{results.length} results</span>
    {/if}
  </div>

  {#if results.length === 0}
    <div class="empty">
      <div class="empty-radar">
        <div class="radar-ring r1"></div>
        <div class="radar-ring r2"></div>
        <div class="radar-ring r3"></div>
        <div class="radar-sweep"></div>
        <div class="radar-dot"></div>
      </div>
      <p class="empty-text">Search for images or drag and drop files here</p>
    </div>
  {:else}
    <div
      class="grid"
      class:masonry={settings.image_layout === 'masonry'}
      on:click={onGridClick}
      role="presentation"
      bind:this={gridEl}
      use:setupObserver
    >
      {#each results as [path, score, ratio, thumb, rawWebp], i (path)}
        {@const isSelected = selectedPaths.has(path)}
        {@const flash = feedbackFlash.get(path)}
        {@const isRightClicked = contextMenuPath === path}
        <button
          class="card"
          class:selected={isSelected || isRightClicked}
          class:flash-correct={flash === 'correct'}
          class:flash-incorrect={flash === 'incorrect'}
          class:flash-collection={flash === 'collection'}
          class:deleting={deletingPaths.has(path)}
          style="--i: {Math.min(i, 40)}"
          on:click|stopPropagation={(e) => onCardClick(e, path, i, rawWebp)}
          on:contextmenu={(e) => onCardRightClick(e, path, score, i, rawWebp)}
        >
          <img
            src={getThumbUrl(path, thumb) ?? convertFileSrc(path)}
            alt={path.split('\\').pop()}
            loading="lazy"
            decoding="async"
            fetchpriority={i < 12 ? 'high' : 'auto'}
          />
          <div class="card-overlay">
            <span class="card-name">{path.split('\\').pop()}</span>
            <span class="card-score">{(score * 100).toFixed(1)}%</span>
          </div>
        </button>
      {/each}
    </div>
  {/if}
</main>

<style>
  main {
    flex: 1;
    overflow-y: auto;
    padding: 0 24px 24px;
    display: flex;
    flex-direction: column;
  }

  main::-webkit-scrollbar {
    width: 5px;
  }

  main::-webkit-scrollbar-track {
    background: transparent;
  }

  main::-webkit-scrollbar-thumb {
    background: var(--primary-dark);
    border-radius: 3px;
  }

  .count-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    font-size: 11px;
    color: var(--primary-light);
    letter-spacing: 1px;
    border-bottom: var(--border-small) solid var(--primary-dark);
    margin-bottom: 16px;
    flex-shrink: 0;
  }

  .result-count {
    color: var(--primary);
  }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    gap: 28px;
  }

  .empty-radar {
    position: relative;
    width: 140px;
    height: 140px;
  }

  .radar-ring {
    position: absolute;
    border-radius: 50%;
    border: var(--border-small) solid rgba(124, 58, 237, 0.25);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .r1 { width: 140px; height: 140px; }
  .r2 { width: 95px; height: 95px; border-color: rgba(124, 58, 237, 0.45); }
  .r3 { width: 50px; height: 50px; border-color: rgba(124, 58, 237, 0.65); }

  .radar-sweep {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 70px;
    height: 2px;
    transform-origin: left center;
    background: linear-gradient(90deg, rgba(124, 58, 237, 0.9), transparent);
    animation: sweep 3s linear infinite;
  }

  .radar-dot {
    position: absolute;
    width: 6px;
    height: 6px;
    background: var(--primary);
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 8px var(--primary);
  }

  .empty-text {
    color: var(--primary-very-light);
    font-size: 13px;
    letter-spacing: 0.5px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
  }

  .grid.masonry {
    display: block;
    position: relative;
  }

  .card {
    position: relative;
    aspect-ratio: 1;
    border-radius: 10px;
    border: var(--border-small) solid var(--primary-dark);
    overflow: hidden;
    cursor: pointer;
    background: var(--primary-dark);
    padding: 0;
    animation: card-appear 0.25s ease-out both;
    animation-delay: calc(var(--i, 0) * 15ms);
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
  }

  @keyframes card-appear {
    from { opacity: 0; transform: translateY(10px) scale(0.97); }
    to   { opacity: 1; transform: none; }
  }

  .card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.18);
  }

  .card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .card-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 24px 10px 10px;
    background: linear-gradient(transparent, rgba(8, 8, 16, 0.94));
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .card:hover .card-overlay { opacity: 1; }

  .card-name {
    font-size: 11px;
    color: var(--primary-light);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 70%;
  }

  .card-score {
    font-size: 11px;
    color: var(--primary-light);
    font-family: var(--font-secondary);
    font-weight: 600;
  }

  .card.selected {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.4), 0 6px 20px rgba(124, 58, 237, 0.25);
  }

  .card.selected img {
    filter: brightness(0.85) saturate(0.9);
  }

  .card.selected::after {
    content: '';
    position: absolute;
    inset: 0;
    background: rgba(124, 58, 237, 0.12);
    pointer-events: none;
    border-radius: 10px;
  }

  @keyframes flash-correct {
    0%   { box-shadow: 0 0 0 3px rgba(74, 222, 128, 0.9), 0 0 20px rgba(74, 222, 128, 0.5); border-color: #4ade80; }
    100% { box-shadow: none; border-color: #1e1c38; }
  }

  @keyframes flash-incorrect {
    0%   { box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.9), 0 0 20px rgba(248, 113, 113, 0.5); border-color: #f87171; }
    100% { box-shadow: none; border-color: #1e1c38; }
  }

  @keyframes flash-collection {
    0%   { box-shadow: 0 0 0 5px rgba(139, 92, 246, 1), 0 0 30px rgba(139, 92, 246, 0.7); border-color: #c4b5fd; transform: scale(1.02); }
    50%  { box-shadow: 0 0 0 5px rgba(167, 139, 250, 0.5), 0 0 20px rgba(139, 92, 246, 0.4); }
    100% { box-shadow: none; border-color: #1e1c38; transform: scale(1); }
  }

  .card.flash-correct  { animation: flash-correct  1.5s ease-out forwards; }
  .card.flash-incorrect { animation: flash-incorrect 1.5s ease-out forwards; }
  .card.flash-collection { animation: flash-collection 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }

  @keyframes card-delete {
    0%   { box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.9), 0 0 20px rgba(248, 113, 113, 0.5); border-color: #f87171; opacity: 1; transform: scale(1); }
    60%  { box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.5); border-color: #f87171; opacity: 0.6; transform: scale(0.96); }
    100% { opacity: 0; transform: scale(0.92); }
  }

  .card.deleting {
    animation: card-delete 0.5s ease-out forwards;
    pointer-events: none;
  }

  @media (max-width: 680px) {
    .grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
    main  { padding: 0 12px 12px; }
  }

  @media (max-width: 480px) {
    .grid { grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 6px; }
  }
</style>