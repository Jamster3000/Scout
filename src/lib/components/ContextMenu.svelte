<script>
  export let x = 0;
  export let y = 0;
  export let selectedPaths = new Set();
  export let contextMenuPath = null;
  export let collections = [];
  export let imageCollectionsMap = new Map();

  export let onOpen = () => {};
  export let onShowInFolder = () => {};
  export let onFindSimilar = () => {};
  export let onCopyPath = () => {};
  export let onAddToCollection = () => {};
  export let onRemoveFromCollection = () => {};
  export let onOpenCollectionsModal = () => {};
  export let onMarkCorrect = () => {};
  export let onMarkIncorrect = () => {};
  export let onDeleteFromDb = () => {};
  export let onDeleteFromSystem = () => {};

  $: imageCollections = imageCollectionsMap.get(contextMenuPath) ?? new Set();
  $: multiSelect = selectedPaths.size > 1;

  let showCollectionSubmenu = false;
</script>

<div
  class="ctx-menu"
  style="left:{x}px; top:{y}px"
  role="menu"
  tabindex="-1"
  on:click|stopPropagation
  on:keydown|stopPropagation
>
  {#if multiSelect}
    <div class="ctx-header">{selectedPaths.size} images selected</div>
  {/if}

  {#if !multiSelect}
    <button class="ctx-item" role="menuitem" on:click={onOpen}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      Open
    </button>
    <button class="ctx-item" role="menuitem" on:click={onShowInFolder}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
      Show in folder
    </button>
    <button class="ctx-item" role="menuitem" on:click={onFindSimilar}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      Find similar
    </button>
    <button class="ctx-item" role="menuitem" on:click={onCopyPath}>
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
      Copy path
    </button>
    <div class="ctx-divider"></div>
  {/if}

  <div
    class="ctx-item ctx-submenu-trigger"
    role="menuitem"
    tabindex="-1"
    on:mouseenter={() => showCollectionSubmenu = true}
    on:mouseleave={() => showCollectionSubmenu = false}
    on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') showCollectionSubmenu = !showCollectionSubmenu; }}
  >
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/><path d="M7 10h10M7 14h10M7 7h10"/></svg>
    Add to collection
    <svg class="submenu-arrow" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>

    {#if showCollectionSubmenu}
      <div class="ctx-submenu" role="menu" tabindex="-1">
        <div class="submenu-scroll">
          {#if collections.length === 0}
            <div class="ctx-empty">No collections yet</div>
          {:else}
            {#each collections as collection}
              {@const inCollection = imageCollections.has(collection.id)}
              <button
                class="ctx-item ctx-collection-item"
                class:in-collection={inCollection}
                role="menuitem"
                on:click|stopPropagation={() => inCollection ? onRemoveFromCollection(collection.id) : onAddToCollection(collection.id)}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  {#if inCollection}
                    <polyline points="20 6 9 17 4 12"/>
                  {:else}
                    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
                  {/if}
                </svg>
                {collection.name}
              </button>
            {/each}
          {/if}
        </div>
        <div class="ctx-divider"></div>
        <button class="ctx-item ctx-dim" role="menuitem" on:click|stopPropagation={onOpenCollectionsModal}>
          Manage collections...
        </button>
      </div>
    {/if}
  </div>

  <div class="ctx-divider"></div>

  <button class="ctx-item ctx-correct" role="menuitem" on:click={onMarkCorrect}>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
    Relevant
  </button>
  <button class="ctx-item ctx-incorrect" role="menuitem" on:click={onMarkIncorrect}>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    Not relevant
  </button>

  <div class="ctx-divider"></div>

  <button class="ctx-item ctx-warn" role="menuitem" on:click={onDeleteFromDb}>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
    Remove from index
  </button>
  <button class="ctx-item ctx-danger" role="menuitem" on:click={onDeleteFromSystem}>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M9 6V4h6v2"/></svg>
    Delete from system
  </button>
</div>

<style>
  .ctx-menu {
    position: fixed;
    background: var(--background);
    border: var(--border-small) solid var(--primary-dark);
    border-radius: 8px;
    padding: 6px;
    min-width: 210px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(124,58,237,0.1);
    z-index: 150;
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

  .ctx-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: 4px;
    background: transparent;
    border: none;
    color: var(--primary-very-light);
    font-family: inherit;
    font-size: 12px;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
    width: 100%;
  }

  .ctx-item:hover {
    background: rgba(124, 58, 237, 0.1);
    color: var(--text-colour);
  }

  .ctx-divider {
    height: 1px;
    background: var(--primary-dark);
    margin: 3px 0;
  }

  .ctx-submenu-trigger {
    position: relative;
  }

  .submenu-arrow {
    margin-left: auto;
  }

  .ctx-submenu {
    position: absolute;
    left: 100%;
    top: 0;
    background: var(--background);
    border: var(--border-small) solid var(--primary-dark);
    border-radius: 8px;
    padding: 6px;
    min-width: 180px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(124,58,237,0.1);
    z-index: 151;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .submenu-scroll {
    max-height: 200px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .submenu-scroll::-webkit-scrollbar {
    width: 4px;
  }

  .submenu-scroll::-webkit-scrollbar-thumb {
    background: var(--primary-dark);
    border-radius: 2px;
  }

  .ctx-collection-item.in-collection {
    color: var(--primary-light);
  }

  .ctx-empty {
    padding: 8px 10px;
    font-size: 11px;
    color: var(--primary-very-light);
  }

  .ctx-dim {
    opacity: 0.6;
    font-size: 11px;
  }

  .ctx-correct {
    color: #4ade80;
    border: var(--border-small) solid rgba(74, 222, 128, 0.3);
    margin-bottom: 2px;
  }

  .ctx-correct:hover {
    background: rgba(74, 222, 128, 0.1);
    border-color: #4ade80;
  }

  .ctx-incorrect {
    color: #f87171;
    border: var(--border-small) solid rgba(248, 113, 113, 0.3);
  }

  .ctx-incorrect:hover {
    background: rgba(248, 113, 113, 0.1);
    border-color: #f87171;
  }

  .ctx-warn {
    color: var(--primary-warn);
  }

  .ctx-warn:hover {
    background: rgba(251, 146, 60, 0.1);
  }

  .ctx-danger {
    color: var(--primary-danger);
  }

  .ctx-danger:hover {
    background: rgba(239, 68, 68, 0.1);
  }
</style>