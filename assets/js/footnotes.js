// Hovering footnotes

document.addEventListener('DOMContentLoaded', function() {
  // Get all footnote references
  const footnoteRefs = document.querySelectorAll('a.footnote');
  
  // Get all footnote contents
  const footnotes = document.querySelectorAll('.footnotes li');
  
  footnoteRefs.forEach(ref => {
      // Get the footnote ID from the href
      const footnoteId = ref.getAttribute('href').substring(1);
      
      // Find the corresponding footnote content
      const footnote = document.getElementById(footnoteId);
      
      if (footnote) {
          // Create tooltip container
          const tooltip = document.createElement('span');
          tooltip.className = 'footnote-tooltip';
          
          // Get the footnote content (excluding the back-reference)
          const footnoteContent = footnote.cloneNode(true);
          const backRef = footnoteContent.querySelector('.reversefootnote');
          if (backRef) {
              backRef.remove();
          }
          tooltip.innerHTML = footnoteContent.innerHTML;
          
          // Wrap the superscript in a span for positioning
          const wrapper = document.createElement('span');
          wrapper.className = 'footnote-ref';
          ref.parentNode.insertBefore(wrapper, ref);
          wrapper.appendChild(ref);
          wrapper.appendChild(tooltip);
      }
  });
});