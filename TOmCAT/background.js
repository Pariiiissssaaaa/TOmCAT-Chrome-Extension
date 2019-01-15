'use strict';

function scrapeReviews(htmlPage, prodId) {
  let reviews = [];
  let $htmlPage = $(htmlPage);
  for (let reviewDiv of $htmlPage.find("div[data-hook=review]")) {
    // console.log(reviewDiv);
    let review = {};
    let $reviewDiv = $(reviewDiv);
    // console.log($reviewDiv.find("a[data-hook='review-title']"));
    review.prodId = prodId;
    review.reviewId = $reviewDiv.attr("id");
    review.title = $reviewDiv.find("a[data-hook='review-title']")[0].innerText;
    review.user = $reviewDiv.find('.a-profile-name')[0].innerText;
    review.reviewDate = $reviewDiv.find("span[data-hook='review-date']")[0].innerText;
    review.text = $reviewDiv.find('.review-text')[0].innerText;
    review.starRating = $reviewDiv.find("i[data-hook='review-star-rating']")[0].innerText.split(' ')[0];
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


chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if (request.message === "get_reviews") {

      let domain = "https://www.amazon.com/";
      let reviewsPath = "/product-reviews/";
      let query = "/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber=";

      // Put it all together
      let url = domain + request.prodName + reviewsPath + request.prodID + query;
      let ajaxPromises = [];
      for (let pageNum = 1; pageNum <= Math.min(request.lastPage, 4); pageNum++) {
        let reviewPageURL = url + pageNum;
        ajaxPromises.push(
          $.get(reviewPageURL, "html")
        );
        // chrome.tabs.create({url: reviewPageURL, pinned: false}, function(tab) {
        //   // console.log("sent message to " + tab.id);
        //   // chrome.tabs.sendMessage(tab.id, {"message": "scrape_reviews"}, function(htmlReviews) {
        //   //   console.log(tab);
        //   //   console.log(htmlReviews);
        //   // });
        // });
      }

      // console.log(ajaxPromises);
      $.when(...ajaxPromises).then(function() {
        let reviews = [];
        for (let i = 0; i < arguments.length; i++) {
            let pageReviews = scrapeReviews(arguments[i][0], request.prodID);
          reviews = reviews.concat(pageReviews);
        }

        $.ajax({
            url: 'http://0.0.0.0:80/api/v1.0/',
            type: 'post',
            dataType: 'json',
            contentType: 'text/plain',
            traditional: true,
            success: function (data) {
              console.log("Received data: ");
              console.log(data);
                // $('#target').html(data.msg);
            },
            data: JSON.stringify({"prodId": request.prodId, "reviews": reviews})
        });
      });

      // $.post("http://0.0.0.0:80/api/v1.0/", {"prodId": request.prodId, "reviews": reviews})
      //   .done(function( data ) {
      //     console.log("Received data: " + data);
      //   });

    }
  }
);

// chrome.tabs.onUpdated.addListener(function (tabId , info) {
//   console.log(info.status);
//   if (info.status === 'complete') {
//     chrome.tabs.sendMessage(tabId, {"message": "scrape_reviews"}, function(htmlReviews) {
//       console.log(htmlReviews);
//     });
//   }
// });
