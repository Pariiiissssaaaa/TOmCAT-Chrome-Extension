// Saves options to chrome.storage
function save_options() {
  var numReviewPages = document.getElementById('numReviewPages').value;
  chrome.storage.sync.set({
    numReviewPages: numReviewPages
  }, function() {
    // Update status to let user know options were saved.
    var status = document.getElementById('status');
    status.textContent = 'Options saved.';
    setTimeout(function() {
      status.textContent = '';
    }, 1500);
  });
}

// Restores select box and checkbox state using the preferences
// stored in chrome.storage.
function restore_options() {
  // Use default value numReviewPages = 20.
  chrome.storage.sync.get({
    numReviewPages: 20
  }, function(items) {
    document.getElementById('numReviewPages').value = items.numReviewPages;
  });
}
document.addEventListener('DOMContentLoaded', restore_options);
document.getElementById('save').addEventListener('click',
    save_options);
