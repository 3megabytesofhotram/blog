document.addEventListener('DOMContentLoaded', function() {
  const content = document.querySelector('.post-content');
  if (!content) return;

  // Regex for matching URLs
  const urlRegex = /(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/g;

  // Get all text nodes in the content
  const walk = document.createTreeWalker(
      content,
      NodeFilter.SHOW_TEXT,
      null,
      false
  );

  const nodes = [];
  let node;
  while (node = walk.nextNode()) {
      nodes.push(node);
  }

  // Process each text node
  nodes.forEach(node => {
      if (!node.parentElement.closest('a') && // Skip if already in a link
          !node.parentElement.closest('pre') && // Skip code blocks
          !node.parentElement.closest('code')) { // Skip inline code
          
          const html = node.textContent.replace(urlRegex, function(url) {
              return `<a href="${url}" target="_blank">${url}</a>`;
          });

          if (html !== node.textContent) {
              const span = document.createElement('span');
              span.innerHTML = html;
              node.replaceWith(span);
          }
      }
  });
});