chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'capture') {
      const content = {
        title: document.title,
        url: window.location.href,
        text: document.body.innerText,
        html: document.body.innerHTML
      };
      
      // Send captured content to background script
      chrome.runtime.sendMessage({
        action: 'sendToServer',
        content: content
      });
    }
    return true;
  });