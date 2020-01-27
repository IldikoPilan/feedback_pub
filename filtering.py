###########
# Filtering  
###########

import re

def enough_alpha(string, threshold=0.4):
    """ Ensures that the string contains enough alphabetic characters.
    """
    if string:
        alpha_ratio = sum([1 for char in string.replace(" ", "").replace("#", "")
                           if char.isalpha()]) / len(string) 
        if alpha_ratio >= threshold:
            return True

def date(comment_words):
    """ Detects if a comment is just a date.
    """ 
    months = ["January, February", "March", "April", "May", "June", "July", 
              "August", "September", "October", "November", "December"]
    if len(comment_words) <= 4:
        for word in comment_words:
            if word.replace(",", "") in months:
                return True

def junk(comment):
    """ Identifies junk comments or the ones not containing linguistic feedback.
    """
    junk_case_sens = ["See memo", "BE ,T1", "BE, T2", "Quantity", "Discount", "Birthday",
    "Aspect", "Microsoft", "eV / kT", "if !vml",  "endif", "if !supportLists", "Object.Method()",
    "#NAME?", "https://", "http : / / ", "if !support", "if !  support", "page:Section", "7a-7b"]
    junk = ["salestime", "salesdate", "sales date", "sales time", "unitpricehk", 
    "unitinstock", "branchzone", "productid", "online"] 
    #junk += ["see comment", "refer note", "see above", "see as above", 
    #"see end comment", "same comment as", "same as above", "see note", "see end commnet", 
    #"refer to above", "see previous comment"] #used only when working with comment characteristics
    if len(comment) < 40:
        for sens_junk in junk_case_sens:
            if sens_junk in comment:
                return True
        for junk_el in junk:
            if junk_el in comment.lower():
                return True

def mostly_phonetic(comment_words, course):
    """ Identifies comments containing almost only phonetic symbols.
    """
    course_type = course[:3]
    phon_symbols = ["ʦ", "ʱ", "ʰ", "ʂ", "ɕ", "ʑ", "ʨ", "æ", "ɲ", "ʎ", "ɳ", "ṣ", "ẓ"] 
    vowels = ["a","e", "i", "o", "u"]
    has_vowel = sum([1 for char in "".join(comment_words) if char in vowels]) # TO DO: add lower()?
    if course_type == "CTL" and len(comment_words) <= 6:
        if not has_vowel:
            return True 
        # TO DO: check if redundant  
        for w in comment_words:
            for phon_symbol in phon_symbols:
                if phon_symbol in w:
                    return True

def get_praise_phrases(adjs, error_type=""):
    nouns = ["topic", "draft", "first draft", "second draft", "final draft",
             "work", "attempt", "try", "effort", "job", "response", "improvement",
             "writing", "essay", "piece", "report", "website",
             "opening", "begining", "introduction", "intro", "start", 
             "title", "tittle", "paragraph", "reference", "inclusion", "sentence", 
             "background", "discussion", "lead", "changes", "analysis",
             "conclusion", "summary", "overview", "structure", "grouping", "comparison", "definition",
             "expression", "explanation", "evaluation", "content",
             "insight", "point", "understanding", "teamwork", "cause", "clarity",
             "choice", "use", "way", "placement", "organisation"]
    specific = ["link", "linking", "signpost", "signposting", "connector", "connective"]
    if error_type == "LA":
        nouns += specific
    praise_phrases = ["good luck", "nicely"]
    for adj in adjs:
        praise_phrases.append(adj + "!")
        praise_phrases.append(adj + " !")
        for noun in nouns:
            praise_phrases.append(adj + " " + noun)
    #print(len(praise_phrases)) #280 or 317 with signposting
    #for pp in sorted(praise_phrases):
    #    print(pp)
    return praise_phrases

def no_local_rev_requirement(comment, praise_phrases): #error_type
    """ Comment does not require a local revision. Includes holistic
    and praise-only comments. (Low recall, minimizes false positives, 
    additional manual filtering expected.)
    It assesses student performance overall. Only positive for now.
    TO DO: add meh / negative?  e.g. acceptable, usatisfying, unsatisfying
    """
    comment = comment.lower().replace("  ", " ")
    rare_praise_phrases = ["end comment", "comments:", "dear"] # clear indication or less frequent items
    adjs = ["good", "great", "nice", "excellent", "wonderful", "lovely"] 
    for rare_praise_phrase in rare_praise_phrases:
        if rare_praise_phrase in comment or rare_praise_phrase == comment:
            return True
    for adj in adjs:
        if adj == comment.replace(" ", "") or "very " + adj == comment:
            return True
        elif adj in comment and len(comment) <= 100:    #50 for 0703 data
            #if 50 < len(comment) <= 100:
            #    print(comment)
            return True
    for praise_phrase in praise_phrases:
        if praise_phrase == comment:
            return True
        if praise_phrase in comment: 
            if len(comment) <= 100 and ("not" not in comment or "n't" not in comment):
                return True

def filter_comment(orig_comment, course, filter_no_lrr, praise_phrases,
                   max_len=300, show_bad=False):
    """ Applies the different comment filters.
    """
    if orig_comment:
        comment = re.sub(r"\[ ?\d* ?\]?", "", orig_comment.lstrip("{").replace("|", ""))
        comment = re.sub(r"Figure \d+", "", comment)
        comment = re.sub(r"photo \d+", "", comment)
        comment = re.sub(r"see \d+", "", comment)
        if comment:
            comment_words = list(filter(None, comment.split(" ")))
            is_enough_alpha = enough_alpha(comment)
            is_junk = junk(comment)
            is_phon = mostly_phonetic(comment_words, course)
            is_date = date(comment_words)
            if filter_no_lrr:
                no_lrr = no_local_rev_requirement(comment, praise_phrases) #error_type
            else:
                no_lrr = False
            if len(comment) < max_len and is_enough_alpha and not is_junk \
                                      and not is_phon and not is_date \
                                      and not no_lrr: 
                return comment
            else:
                if show_bad:
                   print(comment)

def is_sentence(string, nlp_pipeline):
    if string:
        if (string[0].isupper() or string[0] == "[") and string[-1] in [".", "!", "]"]:
            parsed_str = nlp_pipeline(string)
            pos = [parsed_str.vocab.strings[token.pos] for token in parsed_str]
            if 'VERB' in pos:
                return True

def has_encoding_prob(string):
    for elem in ["#", "?", "HYPERLINK & quot"]:
        if elem in string:
            return True

def filter_stud_resp(stud_resp, nlp_pipeline, min_len=10):
    if is_sentence(stud_resp, nlp_pipeline) and not has_encoding_prob(stud_resp) \
                                            and enough_alpha(stud_resp)          \
                                            and len(stud_resp) > min_len:
        return stud_resp

