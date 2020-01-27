# Functions processing the L2 feedback corpus 

import os
import xml.etree.ElementTree as ET
import csv
import pickle
import re
from linking_adverbials import linking_adv, is_linking_adv, is_linking_adv_stud, add_low_fr_to_terms
from filtering import filter_comment, get_praise_phrases, no_local_rev_requirement

#########################
# Load and save functions
#########################

def filter_files(path_to_folder):
    return sorted([file_name for file_name in os.listdir(path_to_folder) 
                   if "DS_Store" not in file_name]) 

def load_error_cats(file_name):
    """ Returns a dictionary of (semester, error_id) pair as key and 
    the corresponding error category as value.
    """ 
    error_cats = {}
    with open(file_name, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        header = []
        for row_ix,row in enumerate(csv_reader):
            if not row_ix:
                header = row
            elif row:
                mapped_cats = list(zip(header, row)) # e.g. ('2007-08A', '11')
                for mapped_cat in mapped_cats[1:]:
                    error_cats[mapped_cat] = mapped_cats[0][1]
    return error_cats

def write_to_csv(cvs_name, rows, mode="w"):
    """ 
    @ mode: "a" to append to file, "w" to write to file
    """
    try:
        with open(cvs_name, mode, newline="") as csvfile:
            csv_writer = csv.writer(csvfile) #, delimiter='\t'
            for row in rows:
                csv_writer.writerow(row) # row -> list of row elements
    except FileNotFoundError:
        print(cvs_name)

def load_xml(path_to_data_file):
    """ Returns the root node of an XML.
    """
    #try:
    with open(path_to_data_file) as f:
        tree = ET.parse(f)
    return tree.getroot()
    #except FileNotFoundError:
    #    return None

def load_grams(ngram_file):
    with open(ngram_file, "r") as f:
        lines = f.readlines()
    ngrams_dict = {}
    for line in lines:
        if line.replace(" ", "") != "\n":
            line_el = line.split("\t")
            try:
                count = int(line_el[0])
            except:
                print(line_el)
            ngram = line_el[1].strip("\n")
            ngrams_dict[ngram] = count
    return ngrams_dict

###########################
# Collecting info from data
###########################

def get_target_tokens(target_tokens):
    """ Collect token ids indicated for the error. 
    Returns a list of token ids (e.g. 'w3').
    """
    if target_tokens:
        if "(" in target_tokens:
            limits = target_tokens.replace("range(", "").strip(")").split(",")
            return ["w"+str(i) for i in range(int(limits[0].replace("w", "")), 
                    int(limits[1].replace("w", ""))+1)]
        else:
            return [target_tokens]
    else:
        return None

def fix_unicode(string):
    """ 
    """
    # Unicode Decimal Code 
    # equivalents from: https://www.codetable.net/decimal/8212
    code_to_char = { "& # 8211 ;" : "-",
                     "& # 8212 ;" : "--",
                     "& # 8217 ;" : "'",
                     "& # 8220 ;" : "\"", 
                     "& # 8221 ;" : "\"", 
                     "& # 8230 ;" : "...", 
                     "& # 160 ;" : " ", #"<>"
                     "& # 160" : " ",   #"<>"
                     "160" : " " }      #"<>"
    if string:
        for decimal_code, char in code_to_char.items():
            string = string.replace(decimal_code, char)
        #print(string.replace("& #", ""))
        return string.replace("& #", "")

def add_more_context(sents, target_sent, first_trg_ix, last_trg_ix, more_context=""):
    """ Add previous and next sentence to target sentence.
    """
    if first_trg_ix:
        prev_sent = sents[first_trg_ix-1][1]
        first_trg_ix -= 1
        if (prev_sent == "?" or len(prev_sent) < 20) and first_trg_ix >= 1:
            prev_sent = ""
            prev_sent = sents[first_trg_ix-1][1] + " " + sents[first_trg_ix][1]
            first_trg_ix -= 1
    else:
        prev_sent = ""
    if last_trg_ix <= len(sents)-2:
        next_sent = sents[last_trg_ix+1][1]
        last_trg_ix += 1
        if (next_sent == "?" or len(next_sent) < 20) and last_trg_ix <= len(sents)-2:
            next_sent = ""
            next_sent = sents[last_trg_ix][1] + " " + sents[last_trg_ix+1][1] 
            last_trg_ix += 1
    else:
        next_sent = ""
    more_context = " ".join([prev_sent, target_sent, next_sent])
    #if ("?" in more_context or more_context[0].islower()) and len(more_context) < 150:
    #    add_more_context(sents, target_sent, first_trg_ix, last_trg_ix, more_context)
    return (first_trg_ix, last_trg_ix, more_context)

def get_st_sentence(xml_root, target_tokens):
    """ Get student sentence in which most target 'tokens' for the error appear.
    Returns also the previous sentence if any.
    """
    sents = []
    target_sents = []
    for sentence in xml_root.iter("{http://www.tei-c.org/ns/1.0}s"):
        sent = []
        match_count = 0
        for element in sentence:
            if element.tag == "{http://www.tei-c.org/ns/1.0}w":
                if element.text:
                    if "{http://www.w3.org/XML/1998/namespace}id" in element.attrib:
                        el_id = element.attrib["{http://www.w3.org/XML/1998/namespace}id"]
                        if el_id in target_tokens:
                            sent.append("[[" + element.text + "]]")
                            match_count += 1
                        else:
                            sent.append(element.text)
            else: # handle if sentence is highlighted 
                for highlighted_w in element:
                    if highlighted_w.tag == "{http://www.tei-c.org/ns/1.0}w":
                        if highlighted_w.text:
                            if "{http://www.w3.org/XML/1998/namespace}id" in highlighted_w.attrib:
                                if highlighted_w.attrib["{http://www.w3.org/XML/1998/namespace}id"] in target_tokens:
                                    sent.append("[[" + highlighted_w.text + "]]")
                                    match_count += 1
                                else:
                                    sent.append(highlighted_w.text)
        sent_str = " ".join(sent)
        sents.append((match_count, sent_str))

    target_sents = [(ix, sent) for ix, (count,sent) in enumerate(sents) if count]
    target_sent = " ".join([sent for ix, sent in target_sents])
    if target_sents:
        first_trg_ix = target_sents[0][0]
        last_trg_ix = target_sents[-1][0]
        # complete incomplete target sentence (sentence split issues during data extraction due to encoding problems)
        if target_sent[0] in ["?", "[[?]]"] or target_sent[0].islower() or len(target_sent.replace("[", "").replace("]","")) < 20:
            first_trg_ix, last_trg_ix, target_sent = add_more_context(sents, target_sent, first_trg_ix, last_trg_ix)
        # get more context (previous and next sentence)
        more_context = add_more_context(sents, target_sent, first_trg_ix, last_trg_ix)[2]
    else:
        more_context = ""
    return target_sent, more_context

def load_revision(path_to_original):
    """ Loads the revised student essay corresponding to the original version
    under the provided path. Returns a tuple of the path to the revised version
    and its loaded XML. 
    # version0, 1, 2
    @ path_to_original: path to file with original (pre-revision) version of student essay 
    """ 
    fn_elem = path_to_original.split("_")
    version_ix = None
    for ix, elem in enumerate(fn_elem):
        if "version" in elem:
            version_ix = ix
    if version_ix:
        next_version = fn_elem[version_ix][:-1] + str(int(fn_elem[version_ix][-1])+1)
        fn_elem[version_ix] = next_version
        # Revisions in the non-final version
        try:
            rev_fn = "_".join(fn_elem)
            return (rev_fn,load_xml(rev_fn))
        except FileNotFoundError:
            # Revisions in the final version if any 
            try: 
                fn_elem[version_ix] = next_version[:-1] + "final"
                rev_fn = "_".join(fn_elem)
                return (rev_fn, load_xml(rev_fn))
            # no subsequent version
            except FileNotFoundError:       
                return None

def get_revision_cost(revision_types):
    """ Maps revision types to a cost reflecting the student's amount of effort 
    in terms of productive skills used for the revision. Per-token revision effort 
    value is normalised for span length.
    """
    cost_mapping = {"identical":0, "delete":1, "shift":2, "insert":3, "replace":3} 
    return round(sum([cost_mapping[revision_type] for revision_type in revision_types])
                 / len(revision_types))
    
def get_revision_type(revision_types):
    """ Maps multiple revision types from the error span to one single label.
    Revision types: identical, shift, replace, delete, insert.
    """
    unique_rev_types = list(set([rev_type for rev_type in revision_types if rev_type not in ["from_file", "to_file"]]))
    if len(unique_rev_types) > 1:
        unique_rev_types = [rt for rt in unique_rev_types if rt != "identical"]
        if len(unique_rev_types) > 1:
            revision_type = "multiple"
            #revision_type = "replace"   # any mixed revision type mapped to 'replace'
        else:
            revision_type = unique_rev_types[0]
    elif len(unique_rev_types) == 1:
        revision_type = unique_rev_types[0]
    else:
        revision_type = ""
    #if revision_type == "shift":
    #    revision_type = "replace"
    #revision_type = "-".join(revision_types)
    return revision_type   

def get_revision_info(loaded_alignments, loaded_revisions, target_tokens, max_window_size=3):
    """ Returns a tuple with (the list of revised tokens, revision_effort, revision type) 
    corresponding to the original target tokens of teacher's comment.
    """
    revision_types = []
    revised_tokens = []
    all_rev_types = []
    del_tok = []
    for target_token in target_tokens:
        for link in loaded_alignments.iter("{http://www.tei-c.org/ns/1.0}link"):
            if not link.attrib.get("target"):       # skip <link> in <sourceDesc>
                rev_type = link.attrib.get("type")
                original = link.attrib.get("prev")  # original token id
                revised = link.attrib.get("next")   # revised token id
                if revised:
                    revised = revised.split("#")[1]
                if original:
                    original = original.split("#")[1]
                    if original == target_token:
                        if revised:
                            revised_tokens.append(revised)
                        else:
                            if all_rev_types:
                                del_tok.append(all_rev_types[-1][0])
                        revision_types.append(rev_type)
                    if rev_type == "identical" and revised:
                        all_rev_types.append((int(revised[1:]), rev_type))
    
    # look for insertions around the revised tokens
    if revised_tokens:
        window_size = 1
        #window_size = len(target_tokens)
        #if window_size > 3:
        #    window_size = 3
        target_span_int = [int(tok_id[1:]) for tok_id in revised_tokens]
        for link in loaded_alignments.iter("{http://www.tei-c.org/ns/1.0}link"):
            if not link.attrib.get("target") and not link.attrib.get("prev") and link.attrib.get("next"): # skip <link> in <sourceDesc>
                insertion = link.attrib.get("next").split("#")[1]
                if min(target_span_int)-window_size <= int(insertion[1:]) \
                                                    <= max(target_span_int)+window_size: # window of 1 compared to span -> more?
                                                                                         # same window size as error span?
                    revision_types.append(link.attrib.get("type")) # 'insert' (no 'prev' attribute)
                    revised_tokens.append(insertion)
    
    # for deleted tokens, if previous or next token identical, 
    # add as revised tokens to be able to return revised sentence
    if del_tok and not revised_tokens:
        for (tok, rev) in all_rev_types:
            if tok == min(del_tok):     # token before deletion
                revised_tokens.append("w"+str(tok))
            elif tok == max(del_tok)+1: # token after delition
                revised_tokens.append("w"+str(tok))

    revision_type = get_revision_type(revision_types)
    revision_effort = str(get_revision_cost(revision_types))
    return (revision_type, revision_effort, revised_tokens) # to do: check if st_rev_sent token ids only 

def get_data(path_to_data, path_to_error_cats, result_folder, nlp_pipeline, filter_no_lrr=False, 
             low_fr_to_terms=True, feedback_type="open", error_type="ALL", linking_adv=linking_adv, 
             max_error_span=10):
    """ Collects student errors marked by teachers via error tags ('tagged') or 
    open-ended comments ('open'). 
    Collects informaiton and saves it to both a CSV and a pickled Python object. CSV columns:
    (A) on essay ID, (B) error tag / comment, (C) error location (relevant tokens), (D) revision type, 
    (E) revision effort, (F) original student sentence(s), (G) revised student sentence(s), 
    (H) original student sentence preceding the relevant sentence.
    @ path_to_data: 
    @ path_to_error_cats: CSV file with Commentbank category IDs per semester
    @ result_folder:      path to folder where output files should be saved
    @ filter_no_lrr:      discard comments where no local revision is required (holistic / only positive)
    @ low_fr_to_terms:    add low frequency linking words to terminology (to match also occurrence without quotes) 
    @ feedback_type:      'tagged' for errors with Commentbank categories,
                          'open' for errors with open-ended comments 
                          (mixed tag and open-ended are excluded in both cases)
    @ error_type:         whether to filter for any specific error types (only 'LA' linking adverbials implemented) 
                          or collect any error type ('ALL') that satisfies filtering criteria
    @ linking_adv:        linking adverbials (see linking_adverbials.py)
    @ max_error_span:     span of the error, i.e. how many tokens can be indicated for an error by teachers 
    """
    terminology = ["linker", "linking", "linked", "linkage", "linkng", "linkere", "logical link",
               "connector", "connective", "signpost", "signposting", "joining word", 
               "transition", "discourse marker", "sequence marker", "conjunct ", "adjunct",
               "linking adverbial"]
    connector_cats = ["Adverb needed - Part of speech Incorrect",
                      "Coherence - signposting", 
                      "Coherence - logical sequence",
                      #"Coherence - drawing a parallel between clauses", 
                      "Conjunction - Wrong Use", 
                      "Conjunction Missing", 
                      "Conjunction missing OR wrong use",
                      "Delete this (unnecessary)",
                      #"Reference",
                      #"Reference - missing or unclear",
                      #"Sentence - Fragment",
                      #"Sentence - New sentence",
                      "Word choice",
                      "Word choice - Level of formality",
                      "Word order"]
    adjs = ["good", "great", "nice", "excellent", "wonderful"]
    comments = {}
    anomalous_ecode = 0
    error_cats = load_error_cats(path_to_error_cats)
    unigrams = load_grams("freq_unigrams")
    bigrams = load_grams("freq_bigrams")
    praise_phrases = get_praise_phrases(adjs, error_type)
    #nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    if low_fr_to_terms:                                     
        terminology, link_words = add_low_fr_to_terms(terminology, linking_adv, unigrams, bigrams)
    else:
        link_words = list(set(linking_adv["band1"] + linking_adv["band2"] + linking_adv["band3"]))
    #exit()
    for semester in filter_files(path_to_data):
        for course in filter_files(os.path.join(path_to_data,semester)):
            for assignment in filter_files(os.path.join(path_to_data,semester,course)):
                print(semester, course, assignment)
                for data_file in filter_files(os.path.join(path_to_data,semester,
                                                           course,assignment)):
                    if data_file[-3:] == "xml":
                        #if "CTL_0011_3210_Asgn_2" in data_file:
                        if "fixed_notes" in data_file and "version0" not in data_file \
                                                      and "final" not in data_file:
                            #version 0 has no teacher comments 
                            path_to_comment_file = os.path.join(path_to_data,semester,course,
                                                                assignment, data_file)
                            # load student original version
                            path_to_st_file = path_to_comment_file.replace("_notes", "")
                            try:
                                st_resp = load_xml(path_to_st_file)
                            except FileNotFoundError:
                                st_resp = None
                            # load revised version
                            rev = load_revision(path_to_st_file)
                            if rev:
                                path_to_st_rev, st_rev = rev
                                # load word alignment file
                                path_to_alig = path_to_st_rev.replace("_fixed", "_fixed_wordAlign")
                                word_alignments = load_xml(path_to_alig)
                            else:
                                word_alignments = None
                                st_rev = None
                            with open(path_to_comment_file) as f:
                                tree = ET.parse(f)
                                root = tree.getroot()
                                for note in root.iter("{http://www.tei-c.org/ns/1.0}note"):
                                    #*if note.text:            # only <note> with an open-ended comment 
                                    if (note.text and feedback_type == "open") or (not note.text and feedback_type == "tagged"):                                            
                                        try:
                                            try:
                                                error_cat = error_cats[(semester, 
                                                            note.attrib.get("type").split(":")[1].lstrip("0"))]
                                            except KeyError: # handling anomalous error codes (e.g. multiple codes)
                                                #print(note.attrib.get("type").split(":")[1].lstrip("0"))
                                                anomalous_ecode += 1
                                        except AttributeError:        # handling lack of error category -> open-ended
                                            error_cat = "open_ended"   
                                        if feedback_type == "open":
                                            comment = filter_comment(note.text, course, filter_no_lrr, praise_phrases)
                                            #comment = note.text # to avoid filtering
                                        else:
                                            comment = error_cat
                                        if comment:
                                            #if linking_adv: # for getting ALL open-ended comments 
                                            if (feedback_type == "open" and error_cat == "open_ended") or  \
                                               (feedback_type == "tagged" and error_type == "ALL") or      \
                                               (feedback_type == "tagged" and error_type == "LA" and comment in connector_cats): # including only open ended comments
                                                try:
                                                    target_tok = note.attrib.get("target").split("#")[1] #range(w29,w32) or w32
                                                except AttributeError:
                                                    target_tok = None
                                                target_tok_list = get_target_tokens(target_tok)
                                                if target_tok_list: # exclude instances without target tokens or too many target tokens
                                                    if len(target_tok_list) < max_error_span:
                                                        essay_id = "_".join([semester] + data_file.split("_")[:6])
                                                        if st_resp:
                                                            s = get_st_sentence(st_resp, target_tok_list)
                                                            if s:
                                                                st_sent, more_context = s 
                                                            else:
                                                                st_sent = s
                                                            if word_alignments and st_rev:
                                                                rev_type, rev_effort, revised_tokens = get_revision_info(word_alignments, st_rev, 
                                                                                                        target_tok_list)
                                                                # get revised student sentence
                                                                rs = get_st_sentence(st_rev, revised_tokens)
                                                                if rs:
                                                                    st_rev_sent, more_context_rev = rs
                                                                    st_rev_sent = fix_unicode(st_rev_sent)
                                                                else:
                                                                    st_rev_sent = rs
                                                                buggy_sent = False
                                                                if st_rev_sent:
                                                                    if "[[?]]" in st_rev_sent and len(st_rev_sent) < 10:
                                                                        buggy_sent = True
                                                                else:
                                                                    rev_type = "removed"
                                                                if rev_type and st_sent and not buggy_sent:
                                                                                                #stud only: and is_linking_adv(comment, terminology, link_words, unigrams, bigrams))
                                                                                                #both: or is_linking_adv_stud(st_sent, st_rev_sent, linking_adv, nlp_pipeline, error_cat, target_tok_list))
                                                                    if error_type == "ALL" or (error_type == "LA" and feedback_type == "open"              \
                                                                                               and (is_linking_adv(comment, terminology, link_words, unigrams, bigrams) \
                                                                                               or is_linking_adv_stud(st_sent, st_rev_sent, linking_adv, nlp_pipeline, error_cat, target_tok_list))) \
                                                                                           or (error_type == "LA" and feedback_type == "tagged" and        \
                                                                                              is_linking_adv_stud(st_sent, st_rev_sent, linking_adv, nlp_pipeline, error_cat, target_tok_list)):
                                                                        if error_cat in comments:
                                                                            comments[error_cat].append([essay_id, comment, 
                                                                                                        ",".join(target_tok_list), 
                                                                                                        rev_type, rev_effort, st_sent, st_rev_sent, more_context, more_context_rev])
                                                                        else:
                                                                            comments[error_cat] = [[essay_id, comment, 
                                                                                                    ",".join(target_tok_list), 
                                                                                                    rev_type, rev_effort, st_sent, st_rev_sent, more_context, more_context_rev]]
                                                                        #if is_linking_adv(comment, terminology, link_words, unigrams, bigrams):
                                                                        #    if no_local_rev_requirement(comment, praise_phrases, error_type):
                                                                        #        pass
                                                                                #print("S:\t", comment)
                                                                            #else:
                                                                            #    print("NS:\t", comment)
    output = []
    for error_cat, comments_list in comments.items():
        print(error_cat, len(comments_list))
        output.extend(comments_list)
    out_file_name = feedback_type + "_" + error_type
    write_to_csv(result_folder + out_file_name + ".csv", output)
    with open(result_folder + out_file_name + ".pkl", "wb") as pickle_file:
        pickle.dump(comments, pickle_file)
    print("Output saved to {}.pkl/.csv".format(out_file_name))
    return comments

