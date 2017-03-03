import networkx as nx
import numpy as np

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
 
def textrank(document):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)

document = """ The decision struck veteran White House journalists as unprecedented in the modern era, and escalated tensions in the already fraught relationship between the Trump administration and the press.

The New York Times, the Los Angeles Times, Politico, BuzzFeed, the BBC and the Guardian were also among those excluded from the meeting, which was held in White House press secretary Sean Spicer's office. The meeting, which is known as a gaggle, was held in lieu of the daily televised Q-and-A session in the White House briefing room.

When reporters from these news organizations tried to enter Spicer's office for the gaggle, they were told they could not attend because they were not on the list of attendees.

In a brief statement defending the move, administration spokeswoman Sarah Sanders said the White House "had the pool there so everyone would be represented and get an update from us today."

The White House press pool usually includes representatives from one television outlet, one radio outlet and one print outlet, as well as reporters from a few wire services. In this case, four of the five major television networks -- NBC, ABC, CBS and Fox News -- were invited and attended the meeting, while only CNN was blocked.

And while The New York Times was kept out, conservative media organizations Breitbart News, The Washington Times and One America News Network were also allowed in.

"This is an unacceptable development by the Trump White House," CNN said in a statement. "Apparently this is how they retaliate when you report facts they don't like. We'll keep reporting regardless."

New York Times executive editor Dean Baquet wrote, "Nothing like this has ever happened at the White House in our long history of covering multiple administrations of different parties. We strongly protest the exclusion of The New York Times and the other news organizations. Free media access to a transparent government is obviously of crucial national interest."

The White House press office had informed reporters earlier that the traditional, on-camera press briefing would be replaced by a gaggle in Spicer's office, reporters in attendance said. Asked about the move by the White House Correspondents Association, the White House said it would take the press pool and invite others as well.

The WHCA protested that decision on the grounds that it would unfairly exclude certain news organizations, the reporters said. The White House did not budge, and when reporters arrived at Spicer's office, White House communications officials only allowed in reporters from specific media outlets.

CNN reporters attempted to access the gaggle when it began at about 1:45 p.m. ET. As they walked with a large group of fellow journalists from the White House briefing room toward Spicer's office, an administration official turned them around, informing them CNN wasn't on the list of attendees.

Reporters from The Associated Press, Time magazine and USA Today decided in the moment to boycott the briefing because of how it was handled.

Asked during the gaggle whether CNN and The New York Times were blocked because the administration was unhappy with their reporting, Spicer responded: "We had it as pool, and then we expanded it, and we added some folks to come cover it. It was my decision to expand the pool."

Several news outlets spoke out against the White House's decision.

"The Wall Street Journal strongly objects to the White House's decision to bar certain media outlets from today's gaggle," a Journal spokesman said. "Had we known at the time, we would not have participated and we will not participate in such closed briefings in the future."

The White House move was called "appalling" by Washington Post Executive Editor Marty Baron, who said the Trump administration is on "an undemocratic path."

Politico editor-in-chief John Harris said that "selectively excluding news organizations from White House briefings is misguided."

Said BuzzFeed editor-in-chief Ben Smith: "While we strongly object to the White House's apparent attempt to punish news outlets whose coverage it does not like, we won't let these latest antics distract us from continuing to cover this administration fairly and aggressively."

The Associated Press said it "believes the public should have as much access to the president as possible."

The White House Correspondents Association also protested the move.

"The WHCA board is protesting strongly against how today's gaggle is being handled by the White House," it said in a statement. "We encourage the organizations that were allowed in to share the material with others in the press corps who were not. The board will be discussing this further with White House staff." 
"""

ranked = textrank(document)
for i in range(20):
	print(ranked[i][1])