var ratingServer = 'http://0.0.0.0:80';

// HELPERS
function scrapeReviews(htmlPage, prodId) {
  let reviews = [];
  let $htmlPage = $(htmlPage);
  for (let reviewDiv of $htmlPage.find("div[data-hook=review]")) {
    let review = {};
    let $reviewDiv = $(reviewDiv);

    review.prodId = prodId;
    review.reviewId = $reviewDiv.attr("id");
    review.title = $reviewDiv.find("a[data-hook='review-title']")[0].innerText;
    review.user = $reviewDiv.find('.a-profile-name')[0].innerText;
    review.reviewDate = $reviewDiv.find("span[data-hook='review-date']")[0].innerText;
    review.text = $reviewDiv.find('.review-text')[0].innerText;
    review.starRating = parseFloat($reviewDiv.find("i[data-hook='review-star-rating']")[0].innerText.split(' ')[0]);
    try {
      review.helpfulVotes = $reviewDiv.find("span[data-hook='helpful-vote-statement']")[0].innerText.split(' ')[0];
      if (review.helpfulVotes === "One") {
        review.helpfulVotes = 1;
      }
      else {
        review.helpfulVotes = parseInt(review.helpfulVotes);
      }
    }
    catch (err) {
      review.helpfulVotes = 0;
    }
    try {
      review.verified = $reviewDiv.find("span[data-hook='avp-badge']")[0].innerText === "Verified Purchase";
    }
    catch (err) {
      review.verified = false;
    }
    reviews.push(review);
  }
  return reviews;
}

// CONTROL FLOW

window.onload = chrome.tabs.query({
  active: true,
  currentWindow: true,
  url: ["*://*.amazon.com/*/product-reviews/*", "*://*.amazon.com/*/dp/*", "*://*.amazon.com/dp/*"]
}, canRunScript);

// Checks if the active tab is an Amazon page and executes script if true
function canRunScript(tabs) {
  console.log(tabs);
  if (tabs.length === 0) {
    score_description.innerText = "Oops, you can only use this extension on an Amazon item or reviews page.";
    page_score.style.color = "brown";
    return;
  }

  let activeTab = tabs[0];

  // If page is done loading, get page score. Otherwise, wait until page is done loading
  if (activeTab.status === 'complete') {
    chrome.tabs.sendMessage(activeTab.id, {"message": "clicked_browser_action"}, getPageScore);
  }
  else {
    score_description.innerText = "Waiting for page to load...";
    chrome.tabs.onUpdated.addListener(function (tabID, info) {
      if (info.status === 'complete' && tabID === activeTab.id) {
        chrome.tabs.sendMessage(activeTab.id, {"message": "clicked_browser_action"}, getPageScore);
      }
    });
  }

}

function getPageScore(productPageInfo) {

  chrome.storage.sync.get({ numReviewPages: 20 }, function(settings) {

    score_description.innerText = "Scraping reviews...";

    let domain = "https://www.amazon.com/";
    let reviewsPath = "/product-reviews/";
    let query = "/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=";

    // Put it all together
    let url = domain + 'ss' + reviewsPath + productPageInfo.prodID + query;

    let ajaxPromises = [];
    for (let pageNum = 1; pageNum <= Math.min(productPageInfo.lastPage, settings.numReviewPages); pageNum++) {
      let reviewPageURL = url + pageNum;
      ajaxPromises.push(
        $.get(reviewPageURL, "html")
      );
    }
    // Get the last page of reviews
    // ajaxPromises.push(
    //   $.get(url+productPageInfo.lastPage, "html")
    // );

    $.when(...ajaxPromises).then(function() {
      let reviews = [];
      for (let i = 0; i < arguments.length; i++) {
          let pageReviews = scrapeReviews(arguments[i][0], productPageInfo.prodID);
        reviews = reviews.concat(pageReviews);
      }
      console.log(reviews);

      $.ajax({
          url: ratingServer + '/api/v1.0/',
          type: 'post',
          dataType: 'json',
          contentType: 'text/plain',
          traditional: true,
          success: updatePopup,
          data: JSON.stringify({"prodId": productPageInfo.prodId, "reviews": reviews})
      });

    });

  });

}

function updatePopup(serverResponse) {
  let score = serverResponse.score*10;
  page_score.innerText = score;

  let score_denominator = document.createElement('h3');
  score_denominator.style.cssText = "color:gray; display:inline;";
  score_denominator.innerText = " / 10";
  score_display.appendChild(score_denominator);

  let learn_more_link = document.createElement('a');
  learn_more_link.href = "http://people.tamu.edu/~kaghazgaran/";
  learn_more_link.target = "_blank";
  learn_more_link.innerText = "(How do we calculate this number?)";
  learn_more.appendChild(document.createElement('br'));
  learn_more.appendChild(learn_more_link);

  if (score <= 3) {
    score_description.innerText = "The reviews for this item appear to be real!";
    page_score.style.color = "green";
  }
  else if (score > 3 && score <= 6) {
    score_description.innerText = "The reviews for this item may be fake.";
    page_score.style.color = "brown";
  }
  // 6 < Score <= 10
  else {
    score_description.innerText = "The reviews for this item appear to be fake!";
    page_score.style.color = "red";
  }
}
