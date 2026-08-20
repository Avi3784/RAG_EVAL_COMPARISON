// Theme Management (Light / Dark mode) with LocalStorage persistence
(function() {
  const savedTheme = localStorage.getItem('rag_eval_theme') || 'light';
  if (savedTheme === 'dark') {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
})();

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('rag_eval_theme', isDark ? 'dark' : 'light');
  updateThemeIcon();
  
  // Dispatch custom event so Chart.js and canvas can update dynamically
  window.dispatchEvent(new CustomEvent('themeChanged', { detail: { isDark } }));
}

function updateThemeIcon() {
  const isDark = document.documentElement.classList.contains('dark');
  const btn = document.getElementById('themeToggleBtn');
  if (btn) {
    btn.innerHTML = isDark 
      ? `<i data-lucide="sun" class="w-4 h-4 text-amber-400 transition-transform duration-300 rotate-0 hover:rotate-90"></i>` 
      : `<i data-lucide="moon" class="w-4 h-4 text-slate-600 transition-transform duration-300 rotate-0 hover:-rotate-12"></i>`;
    if (window.lucide) {
      lucide.createIcons();
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  updateThemeIcon();
});
