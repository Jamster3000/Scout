<script>
  export let indexing = false;
 
  export let onIndexClick = () => {};
  export let onSettingsClick = () => {};
  export let onManageFoldersClick = () => {};
  export let onCollectionsClick = () => {};
</script>
 
<div 
  class="hamburger-menu" 
  on:click|stopPropagation 
  on:keydown={(e) => {
    if (e.key === 'Escape') {
    }
  }} 
  role="menu" 
  tabindex="-1"
>
  <button class="hmenu-item" role="menuitem" on:click={onIndexClick} disabled={indexing}>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    Index Images
  </button>
  <button class="hmenu-item" role="menuitem" on:click={onManageFoldersClick}>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
    Manage Folders
  </button>
  <button class="hmenu-item" role="menuitem" on:click={onCollectionsClick}>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/><path d="M7 10h10M7 14h10M7 7h10"/></svg>
    Collections
  </button>
  <div class="hmenu-divider" role="separator"></div>
  <button class="hmenu-item" role="menuitem" on:click={onSettingsClick}>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
    Settings
  </button>
</div>
 
<style>
  .hamburger-menu {
	position: absolute;
	top: 72px;
	right: 12px;
	background: var(--background);
	border: var(--border-small) solid var(--primary-dark);
	border-radius: 10px;
	padding: 6px;
	min-width: 200px;
	box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(124, 58, 237, 0.1);
	display: flex;
	flex-direction: column;
	gap: 1px;
	z-index: 150;
}
 
.hmenu-item {
	display: flex;
	align-items: center;
	gap: 10px;
	padding: 10px 12px;
	border-radius: 6px;
	background: transparent;
	border: none;
	color: var(--primary-very-light);
	font-size: var(--font-small);
	cursor: pointer;
	text-align: left;
	transition: background 0.15s, color 0.15s;
	width: 100%;
}
 
.hmenu-item:hover:not(:disabled) {
	background: rgba(124, 58, 237, 0.1);
	color: var(--text-colour);
}
 
.hmenu-item:disabled {
	opacity: 0.3;
	cursor: not-allowed;
}
 
.hmenu-divider {
	height: 1px;
	background: var(--primary-dark);
	margin: 4px 0;
}
 
@media (max-width: 680px) {
	.hamburger-menu {
		top: 56px;
	}
}
</style>