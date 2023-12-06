Issue: NOS crawling generate duplicate results

Production:
If one attempt to get search results from nos.nl using search urls repeateadly (for example, https://nos.nl/zoeken?q=abortus&page=1 then https://nos.nl/zoeken?q=abortus&page=2), the website will jump to previous opened page rather than the new page (in this case, the second link will still be redirected to the first one). This is the case on both opening the links with web browsers and using python requests.get. It seems to be the anti-crawling mechanism the website has. It has a very long waiting time for the website to stop this auto redirect and thus is not feasible for large scale crawling.

Fix:
Rather than the previously used request method, the metadata including urls is now crawled with selenium that imitates human actions. It opens a browser window and click on the next page button ('Volgende'). This is implemented in function get_nos_metadata_with_selenium, by driver.find_element("xpath", '/html/body/div[2]/main/div/form/div[2]/div/ul[1]/li[9]/a').click().

Unsolved Issues:
However, if the cookie consent window finished loading, selenium cannot find the next page button anymore, yet the closing button of the cookie consent window is also impossible to be clicked by selenium. The current program needs user to close that window manually after the crawling starts (Chrome browser will start and before the crawling starts it has 6 seconds for the cookie window to load and user to close it).

Data:
The data now contains correct nld data with metadata and train/test split.