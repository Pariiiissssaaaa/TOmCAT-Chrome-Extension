'use strict';

chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    console.log(request.message);
    if( request.message === "clicked_browser_action" ) {

      // Get number of review pages
      let lastPageButton = $( "li.a-last" ).prev().children()[0];
      let seeAllReviewsLink = $( "a[data-hook='see-all-reviews-link-foot']" )[0];

      let lastPage;
      // Last page button is available if we are already on the 'see all reviews' page
      if (lastPageButton) {
        lastPage = parseInt(lastPageButton.innerText);
      }
      // If we are on the product detail page, get total number of reviews to calculate number of review pages
      else if (seeAllReviewsLink) {
        // Parse number of reviews, removing any commas in the string
        let numReviews = parseInt(seeAllReviewsLink.innerText.split(" ")[2].split(",").join(""));
        // Number of pages is numReviews / 10 since there are 10 reviews per page
        lastPage = Math.ceil( numReviews / 10);
      }
      // We are not on either of these pages or the Amazon layout has changed
      else {
        console.log("Error: Could not find number of review pages or reviews.");
      }

      // Get prodID and prodName from URL path
      let path = window.location.pathname.split('/');

      let prodID;
      // URL has schema amazon.com/dp/[prod_id]
      if (path[1] === 'dp') {

        prodID = path[2];
      }
      // URL has schema amazon.com/[prod_name]/dp/[prod_id]
      else {
        prodID = path[3];
      }

      let response = {
        "prodID": prodID,
        "lastPage": lastPage,
      };

      // console.log("sending response");
      // console.log(response);

      sendResponse(response);
    }
  }
);
