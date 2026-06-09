<script>
    export let status = "Ready";
    export let count = 0;
    export let indexing = false;
    export let indexDone = 0;
    export let indexTotal = 0;
    export let indexEstimatedMs = null;
    export let searching = false;
    export let query = "";
    export let showHamburger = false;

    export let onIndexClick = () => {};
    export let onSettingsClick = () => {};
    export let onQueryInput = () => {};
    export let onHamburgerToggle = () => {};
    export let onManageFoldersClick = () => {};
    export let onCollectionsClick = () => {};

    function formatTime(ms) {
        if (ms === null || ms === undefined) return '';
        if (ms < 1000) return `${Math.round(ms / 100) * 100}ms`;
        if (ms < 60000) return `${Math.round(ms / 1000)}s`;
        if (ms < 3600000) {
            const m = Math.floor(ms / 60000);
            const s = Math.round((ms % 60000) / 1000);
            return s > 0 ? `${m}m ${s}s` : `${m}m`;
        }
        const h = Math.floor(ms / 3600000);
        const m = Math.round((ms % 3600000) / 60000);
        return m > 0 ? `${h}h ${m}m` : `${h}h`;
    }
</script>

<header>
  <div class="header-left">
    <div class="brand">
      <span class="brand-text">SCOUT</span>
      <div class="brand-line"></div>
    </div>
    <div class="header-meta">
      <span class="status-text">{status}</span>
      <span class="count-text">{count} images indexed</span>
    </div>
  </div>

  <nav>
    <div class="status-pill" class:active={indexing || status !== 'Ready'}>
      {#if indexing}
        <div class="mini-radar">
          <div class="mini-ring mr1"></div>
          <div class="mini-ring mr2"></div>
          <div class="mini-sweep"></div>
          <div class="mini-dot"></div>
        </div>
        {#if indexTotal > 0}
          <span>{indexDone} / {indexTotal}</span>
          {#if indexEstimatedMs}
            <span class="estimate-time">{formatTime(indexEstimatedMs)}</span>
          {/if}
        {:else}
          <span>Indexing...</span>
        {/if}
      {:else if status !== 'Ready'}
        <span>{status}</span>
      {/if}
    </div>

    <button class="btn btn-primary hide-sm" on:click={onIndexClick} disabled={indexing}>
      {#if indexing}
        <div class="mini-radar">
          <div class="mini-ring mr1"></div>
          <div class="mini-ring mr2"></div>
          <div class="mini-sweep"></div>
          <div class="mini-dot"></div>
        </div>
        {#if indexTotal > 0}
          <span>{indexDone} / {indexTotal}</span>
        {:else}
          <span>Indexing...</span>
        {/if}
      {:else}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Index Images
      {/if}
    </button>

    {#if indexing && indexEstimatedMs !== null}
      <div class="estimate-pill hide-sm">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        {formatTime(indexEstimatedMs)} left
      </div>
    {/if}

    <button class="btn btn-primary hide-sm" on:click={onManageFoldersClick}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
      Manage Folders
    </button>

    <button class="btn btn-primary hide-sm" on:click={onCollectionsClick}>
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/><path d="M7 10h10M7 14h10M7 7h10"/></svg>
      Collections
    </button>

    <div class="search-wrap">
      <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input
        class="search-input input"
        value={query}
        placeholder="Search images..."
        on:input={onQueryInput}
        disabled={indexing}
        aria-disabled={indexing}
      />
      {#if searching}
        <div class="search-radar" role="status" aria-label="Searching">
          <div class="mini-ring mr1"></div>
          <div class="mini-ring mr2"></div>
          <div class="mini-sweep"></div>
          <div class="mini-dot"></div>
        </div>
      {/if}
    </div>

    <button class="setting-btn hide-sm" title="Settings" on:click={onSettingsClick}>
      <span class="bar bar1"></span>
      <span class="bar bar2"></span>
      <span class="bar bar1"></span>
    </button>

    <button
      class="btn btn-secondary show-sm"
      aria-label="Menu"
      aria-expanded={showHamburger}
      on:click|stopPropagation={onHamburgerToggle}
    >
      {#if showHamburger}
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      {:else}
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      {/if}
    </button>
  </nav>
</header>

<style>
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    min-height: 72px;
    background: var(--header-background);
    border-bottom: var(--border-small) solid var(--primary-dark);
    flex-shrink: 0;
    gap: 16px;
    overflow: hidden;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }

  .brand {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex-shrink: 0;
  }

  .brand-text {
    font-family: var(--secondary-font);
    font-size: var(--font-large);
    font-weight: 700;
    letter-spacing: 8px;
    color: var(--primary-light);
    line-height: 1;
  }

  .brand-line {
    height: 1px;
    background: linear-gradient(90deg, var(--primary), transparent);
    width: 100%;
  }

  .header-meta {
    display: flex;
    flex-direction: column;
    gap: 3px;
    border-left: var(--border-small) solid var(--primary-dark);
    padding-left: 16px;
  }

  .status-text {
    font-size: var(--font-very-small);
    letter-spacing: 2px;
    color: var(--primary-very-light);
    text-transform: uppercase;
    font-weight: 500;
  }

  .count-text {
    font-size: var(--font-very-small);
    color: var(--primary-very-light);
    letter-spacing: 0.5px;
  }

  .estimate-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid var(--primary-dark);
    border-radius: 20px;
    font-size: var(--font-small);
    color: var(--primary-light);
    white-space: nowrap;
    flex-shrink: 0;
  }

  nav {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    justify-content: flex-end;
    min-width: 0;
    overflow: hidden;
  }

  .search-wrap {
    position: relative;
    display: flex;
    align-items: center;
    flex: 1;
    min-width: 120px;
    max-width: 320px;
  }

  .search-icon {
    position: absolute;
    left: 10px;
    color: var(--primary-very-light);
    pointer-events: none;
    z-index: 1;
  }

  .search-input {
    width: 100%;
  }

  .search-input:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .mini-radar,
  .search-radar {
    position: relative;
    width: 16px;
    height: 16px;
    flex-shrink: 0;
  }

  .search-radar {
    position: absolute;
    right: 10px;
  }

  .mini-ring {
    position: absolute;
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border: var(--border-small) solid rgba(124, 58, 237, 0.5);
  }

  .mr1 {
    width: 16px;
    height: 16px;
  }

  .mr2 {
    width: 9px;
    height: 9px;
    border-color: rgba(124, 58, 237, 0.8);
  }

  .mini-sweep {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 8px;
    height: 1.5px;
    transform-origin: left center;
    background: linear-gradient(90deg, rgba(124, 58, 237, 1), transparent);
    animation: sweep 1.5s linear infinite;
  }

  .mini-dot {
    position: absolute;
    width: 3px;
    height: 3px;
    background: var(--primary);
    border-radius: 50%;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 4px var(--primary);
  }

  .status-pill {
    display: none;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: rgba(124, 58, 237, 0.12);
    border: var(--border-small) solid var(--primary-dark);
    border-radius: 20px;
    font-size: var(--font-small);
    color: var(--primary-light);
    white-space: nowrap;
    flex-shrink: 0;
    transition: all 0.2s;
  }

  .setting-btn {
    width: 34px;
    height: 34px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 5px;
    background-color: var(--primary-dark);
    border-radius: 8px;
    cursor: pointer;
    border: var(--border-small) solid var(--primary-dark);
    transition: border-color 0.2s, background-color 0.2s;
    flex-shrink: 0;
  }

  .setting-btn:hover {
    border-color: var(--primary);
    background-color: var(--primary-dark);
  }

  .bar {
    width: 16px;
    height: 2px;
    background-color: var(--primary-very-light);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    border-radius: 2px;
    transition: background-color 0.2s;
  }

  .setting-btn:hover .bar {
    background-color: var(--primary-light);
  }

  .bar::before {
    content: "";
    width: 5px;
    height: 5px;
    background-color: var(--primary-dark);
    position: absolute;
    border-radius: 50%;
    border: var(--border-small) solid var(--primary-very-light);
    transition: all 0.3s;
  }

  .setting-btn:hover .bar::before {
    background-color: var(--primary);
    border-color: var(--primary-light);
  }

  .bar1::before {
    transform: translateX(-5px);
  }

  .bar2::before {
    transform: translateX(5px);
  }

  .setting-btn:hover .bar1::before {
    transform: translateX(5px);
  }

  .setting-btn:hover .bar2::before {
    transform: translateX(-5px);
  }

  @media (max-width: 900px) {
    .header-meta {
      display: none;
    }

    .brand-text {
      font-size: var(--font-large);
      letter-spacing: 6px;
    }
  }

  @media (max-width: 680px) {
    .status-pill {
      display: flex;
    }

    .status-pill:not(.active) {
      display: none;
    }

    .hide-sm {
      display: none !important;
    }

    .show-sm {
      display: flex !important;
    }

    header {
      padding: 0 12px;
      gap: 8px;
      min-height: 56px;
    }

    .brand-text {
      font-size: var(--font-medium);
      letter-spacing: 5px;
    }

    .search-wrap {
      max-width: none;
    }
  }

  @media (min-width: 681px) {
    .show-sm {
      display: none !important;
    }
  }
</style>