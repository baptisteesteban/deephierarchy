window.MathJax = {
  tex2jax: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  }
};

document$.subscribe(() => { 
  if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Queue) {
    window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub]);
    return;
  }

  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.startup.output.clearCache();
    window.MathJax.typesetClear();
    window.MathJax.texReset();
    window.MathJax.typesetPromise();
  }
})