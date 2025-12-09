/*
// background.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'captureTab') {
    chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 100 }, dataUrl => {
      sendResponse({ dataUrl: dataUrl });
    });
    return true; // Will respond asynchronously
  }
});
*/
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'captureTab') {
    chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 100 }, dataUrl => {
      chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        sendResponse({ 
          dataUrl: dataUrl,
          url: tabs[0].url,
          title: tabs[0].title
        });
      });
    });
    return true; // Will respond asynchronously
  }
});