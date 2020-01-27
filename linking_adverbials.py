# Based on list in Liu (2008) linking adverbials (single-word or prepositional phrases, 4 meaning categories)
# from: https://www.jbe-platform.com/docserver/fulltext/ijcl.13.4.05liu.pdf?expires=1562041529&id=id&accname=cityhkg%2F1&checksum=6565FEBC19B36296767C0CC18D0837CA

# Frequency band 1 (frequency of 50 and above per million words)
linking_words_band1 = ["again", "also", "of course", "in addition", "too", "for example", "for instance",       # Additive
                       "however", "yet", "nevertheless", "of course", "though", "in fact", "on the other hand", # Adversative
                       "instead", "anyway", "despite",                                                          # Adversative
                       "as a result", "so", "therefore", "thus", "otherwise", "then",                           # Causal                                                                                    
                       "eventually", "first", "firstly", "finally", "then"]                                     # Sequential
# Frequency band 2 (frequency of 10 through 49.99 per million words)
linking_words_band2 = [ "above all", "as I say", "as they say", "as you say", "besides", "furthermore",         # Additive
                        "moreover", "in other words", "namely", "alternatively", "likewise", "similarly",       # Additive
                        "at the same time", "nonetheless", "actually", "in comparison", "by comparison" ,       # Adversative
                        "in contrast", "by contrast", "in reality", "rather", "after all", "all the same",      # Adversative
                        "in any case", "in spite of this", "in spite of that",                                  # Adversative
                        "accordingly", "consequently", "hence", "naturally",                                    # Causal
                        "afterwards", "first of all", "in the first place", "second", "secondly", "third",      # Sequential
                        "thirdly", "at the same time", "in the meantime", "meanwhile", "in short"]              # Sequential
# Frequency band 3 (with frequency under 10 per million words)
linking_words_band3 = ["additionally", "as a matter of fact", "as well", "further", "to crown it all",          # Additive
                        "not to mention", "to cap it all", "what’s more", "what is more", "i.e.", "that is",    # Additive 
                        "that is to say", "for one thing", "to put it another way", "to put it bluntly",        # Additive
                        "to put it mildly", "what I’m saying is"," what I mean is", "which is to say",          # Additive
                        "by the same token", "correspondingly",                                                 # Additive
                        "then again", "as a matter of fact", "conversely", "on the contrary", "admittedly",     # Adversative
                        "anyhow", "at any rate", "still",                                                       # Adversative
                        "all things considered", "as a consequence (of)", "because of", "in consequence",       # Causal
                        "in such a case", "in such cases", "in that case",                                      # Causal
                        "first and foremost", "to begin with", "fourth", "fourthly", "last of all", "last",     # Sequential
                        "lastly", "next", "all in all", "in a word", "in conclusion", "in summary", "in sum",      # Sequential
                        "to conclude", "to sum up", "to summarize", "by the by", "by the way", "incidentally"]  # Sequential

linking_adv = {"band1": linking_words_band1,
               "band2": linking_words_band2,
               "band3": linking_words_band3}

def add_low_fr_to_terms(terminology, linking_adv, unigrams, bigrams):
    link_words = list(set(linking_adv["band1"] + linking_adv["band2"] + linking_adv["band3"]))
    # move infrequent linking words to terminology to match them also without quotes
    infreq_uni = [] 
    infreq_bi = []
    false_positives = ["accordingly", "likewise", "afterwards", "fourth"]
    for lw in linking_adv["band2"] + linking_adv["band3"]:
        if len(lw.split(" ")) == 1 and lw not in false_positives:
            if unigrams.get(lw):
                if unigrams.get(lw) <= 10:
                    infreq_uni.append(lw)
                    terminology.append(lw)
                    link_words.remove(lw)
        elif len(lw.split(" ")) == 2:
            if bigrams.get(lw):
                if bigrams.get(lw) <= 5:
                    infreq_bi.append(lw)
                    terminology.append(lw)
                    link_words.remove(lw)
    print(infreq_uni)
    print(infreq_bi)
    return (terminology, link_words)

def is_linkadv_use(linking_expression, sent, nlp):
    """ Check whether expressions up to 2 words long are used as linking adverbial: 
    1. that they are not adjectival modifiers (e.g. "next time");
    2. are not followed by an adjectival complement (e.g. "too bad"); 
    3. if of two words, first is not subject (e.g. "that is meaningful"). 
    Helps filtering false positive linking words for expressions with multiple syntact functions.
    (Note: many comments are incorrectly parsed hence not filtering for 'advmod' only.)
    """
    words = linking_expression.split(" ")
    if len(words) <= 2:
        doc = nlp(sent.replace("[[", "").replace("]]", ""))
        for i, token in enumerate(doc):
            if len(words) == 1:
                if i < len(doc)-1:
                    if doc.vocab.strings[token.dep] == "advmod" and doc.vocab.strings[doc[i+1].dep] != "acomp":
                        return True
                    elif token.text == words[0] and doc.vocab.strings[token.dep] != "amod":
                       return True
                elif i == len(doc)-1:
                    if token.text == words[0]:
                        return True
            elif len(words) == 2:
                if token.text == words[0] and doc[i+1].text == words[1]:
                    if doc.vocab.strings[token.dep] != "nsubj": # e.g. that is
                        return True
    else:
        return True

def is_linking_adv(comment, terminology, link_words, unigrams={}, bigrams={}):
    """ Open-ended comment relevant to errors involving linking adverbials on the 
    lexica contained in the comment. List of adverbial based on Liu (2008).
    (Matches mostly, but not only "Coherence - sugnposting" error category). 
    TO DO: # nevertheless, firstly -> will miss somme, to do: get freq from data (bi / trigrams) 
    """
    comment = comment.lower()
    for term in terminology:
        if term in comment:
            return True
    for link_word in link_words:
        if "'" + link_word in comment or "\"" + link_word in comment \
                                      or "-"+ link_word in comment   \
                                      or "�" + link_word in comment: 
            return True

def is_linking_adv_stud(st_sent, st_rev_sent, linking_adv, nlp, error_cat, target_tokens):
    """ Scans errors with Commentbank tags and identifies the ones that contain  
    linking adverbials.
    """
    # Length restriction based on error category
    if error_cat == "Delete this (unnecessary)" and len(target_tokens) > 3:
        return False
    seed_exprs = linking_adv["band1"] + linking_adv["band2"] + linking_adv["band3"]
    for expr in seed_exprs:
        pattern = " ".join(["[["+word+"]]" for word in expr.split(" ")])
        if st_rev_sent:
            if pattern in st_rev_sent.lower():
                return is_linkadv_use(expr, st_rev_sent, nlp)
        if st_sent:
            if pattern in st_sent.lower():
                return is_linkadv_use(expr, st_sent, nlp)
