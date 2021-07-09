
# Graphing retail behaviour

## Introduction

All the world and pretty much everything else, is a graph.

And so of course is transactional data.  

My transactional data  have two classes of nodes, the client and the merchant. 

Card transactions (by clients) at retail outlets (merchants) leave deep data in its wake, amount, time merchant name and such like. I explore some ideas in this article, such as classifing clients along shared behaviour or shared preferences.  In the same way merchants may be classified.  Shared client behaviour may come about because of physical presence/location, habit or brand preference.  all of which must be invaluable for banks, brand owners, franchise owners, merketers etc. 

<!-- The bank has rich data pertaining to the shopping behaviour of its clients.  Every transaction by a client is stored and contains such detail as, amongst other, date and time, amount and a brief text description or location of the transaction.   In most instances the text field contains the name of a merchant or owner of the point-of-sale device (POS) used to capture the transaction.  -->

I investigate the insights the graph analysis of transaction data may reveal of such shared behaviour. One obvious exercise is to cluster clients and merchants on transactional behaviour.  Clients 'share' merchants when they transact at the same merchant, and merchants in turn 'share' clients.  There are hence two distinct approaches possible - one isolating clients and the other doing the same for merchants.

Clustering clients will group like clients and allow extraction of similar and repeated transactional behaviour.  I introduce the concept of a transactional center and a transactional address.

Classifying bank clients is traditionally done using 'hard' rules, derived from attributes such as demographics (age, address etc.), over which a client has little or no influence and lead to some unwanted biases . Some attributes are circumstantial, such as income, and may be subject to change and is accurate only in as much as the client chooses to reveal.  Some detail is stale as it is only updated when there is a new interaction between the bank and the client.  

Transactional behaviour on the other hand is current and and largely involuntarily.

From a merchant point of view, a cluster of merchants will reveal true competitors.  Those who compete for product or service and wallet share.  Clustering merchants will reveal retail hubs, shaped by the transactional behaviour of clients, as opposed to physical location.  Merchant insights may be a rich source of information for business banking, and advice to prospecting start-ups, credit extension to existing business clients and the changing nature of bank exposure to name a few! 

Graphing transactional data also offers an appealing visual representation not easily achieved with traditional columnar data, (such as that from a relational database).  Although almost all of the output presented here is ultimately possible with a traditional relational database, but only with great and time consuming effort.  A graph representation opens the door to new insights and new questions.  A graph database is a relatively novel way of exploiting bank data.  Pursuing a graph approach leverages graph mathematics which is well developed and accessible.  

### Merchant rank
A graph approach brings with it some novel concepts.  The behaviour of retail customers can be likened to people traversing the internet by visiting web pages.  The movement of people on the internet and their search detail is a rich source of information and yields invaluable information to marketers and advertisers.  Just ask Google.  Internet behaviour is thouroughly analyzed using a graph representation.  Ask LinkedIn, Facebook and probably any other social media company.   

To profit from internet behaviour, Google aims to produce precision search results.  Precision results lead to a more enjoyable and personalized experience (sometimes unnervingly so) but one which als yields greater revenue to Google via more satisfied advertisers. Google calculates a ranking for each web page. This ranking methodology is described in \cite{robinson2013graph} and the result is formally referred to as the PageRank, and is described in detail in \cite{brin_page_1998}.  PageRank is a model of internet user behaviour. PageRank of an internet page delivers a probability that a random surfer will visit.  But PageRank is more than that.  I show an application here to transactional data

Here is a novel analogy: a client becomes a web surfer while a merchant mimics a web page.  While a web surfer will go from page to page, a shopper will follow a (retail/transactional) path from merchant to merchant.  Some shoppers would visit a limited and fixed set of merchants.  Others may visit many with a wide variety.  Visits to merchants may differ in frequency, spend and date.  Admittedly, some shoppers may visit merchants without conducting a transaction or may do so using cash only.  Cash transactions are effectively hidden from the bank as there is no record of such events.  For our purpose a client is said to have visited a merchant only when a transaction took place. The web page-merchant and web surfer-client analogy is therefore a compelling one.  When a client transacts at a merchant it can be seen as a 'like' that gets amplified by repetition and transaction amount. Such a 'like' is not always available from a web page visit.  
% However, the detail the bank possesses is arguably more detailed that what a search engine may have. Web pages are extremely diverse and is an encyclopedia of human knowledge and behaviour, the retail network the bank identifies is about 500000, but is a representation of the retail experience bank clients pursue.  Many of these retail points are abroad and some are variations of the same merchant.

 

Now a MerchantRank in a retail graph, is the probability that a random shopper will visit a particular merchant.  This number holds new and important information for the owner of a merchant, also for the owner of a group of merchants, or a company.  This merchant rank brings a dimension which traditional accounting measures such as revenue or number of transactions cannot relay.  


% A graph approach to bank (transactional) data places the client at the center of the data, arguably where it should be as opposed to a field in a traditional database.  
% The clients of the bank constitute a sample of the population of all shoppers. Clients earn and spend money and conduct their affairs through a variety of methods.  A subset of these actions are the card 'swipes' at point of sales (POS) devices.  Originally these devices were a physical machine, an online purchase now would be indistinguishable from a physical sale.  The two actions are vastly different though.  The one required a physical presence of the client the other could have been conducted at any remote site.

% The merchant node is the other node class.    In a month a client traverses many merchant nodes.  In this manner merchants becomes connected via clients and when the client node is replace by such merchant link we end up with a graph of merchants.
% Similarly clients become linked via a merchant(s).  By dropping the merchant node we create a client graph.  A client graph will link together clients with similar preferences and similar shopping habits.

