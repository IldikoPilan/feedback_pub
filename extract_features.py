import csv
import process_corpus
import nltk

def extract_features_LA(data_file, features_file, label_file, fname_file, nlp, 
                        add_extra_var=True, target="rev_success"):
    """ Extract features for the linking adverbial (LA) dataset.
    @ data_file: CSV file with all annotation information summed (directness, revision success)
    @ features_file: file name to save feature values to 
    @ label_file: file name to save labels to predict to (= gold revision success annotation)  
    @ fname_file: file name for saving feature names to
    @ nlp: loaded Spacy NLP processing pipeline
    @ add_extra_var: include characteristics not related to comments
    @ target: dependent variable ('rev_success' or 'edit_dist')
    """
    with open("metaling.txt", newline='') as metafile:
        meta_ling_terms = [l.strip("\n") for l in metafile.readlines()]
    hedging_words = ['indicate', 'suggest', 'propose', 'predict', 'assume', 'speculate', 'suspect', 'believe', 
    'imply', 'estimate', 'calculate', 'report', 'note', 'appear', 'seem', 'attempt', 'seek', 'quite', 'partially', 
    'rarely', 'almost', 'approximately', 'generally', 'likely', 'probably', 'presumably', 'apparently', 'evidently', 
    'essentially', 'potentially', 'unlikely', 'possible', 'apparent', 'probable', 'most', 'would', 'may', 'could', 
    'might', 'possibility', 'estimate'] + \
    ["usually", "normally", "slightly", "occasionally", "virtually", "relatively"] + \
    ["assumption", "claim", "suggestion"] + \
    ["try", "sound", "perhaps", "possibly", "little"] # own from most frequent unigrams (55 items)
    with open(data_file, newline='') as csvfile:
        csv_reader = list(csv.reader(csvfile, delimiter=','))
        header = ["ID", "essay ID", "target token", "comment", 
                  "direct_A1", "direct_A2", "gold_dir", 
                  "rev_succ_A1", "rev_succ_A2", "gold_rev_succ", 
                  "original", "revised", "original+", "revised+"]
        revision_success_mapping = {"same":0, "rem":1, "bad":2, "good":3, "alt":4, "?":5}
        feature_values = []
        target_values = []
        feature_names = []
        for row in csv_reader[1:]:
            item_id = row[header.index("ID")]
            if int(item_id[1:]) < 700: # open-ended comments only
                values_per_instance = {}
                #print(row)
                rev_succ = row[header.index("gold_rev_succ")]
                if rev_succ == 'alt':
                    rev_succ = 'good'
                # change_ratio
                edit_dist = nltk.edit_distance(row[header.index("original")], row[header.index("revised")])
                change_ratio = round(edit_dist / len(row[header.index("original")]),2)
                if rev_succ and rev_succ not in ["?", "rem", "skip"]:
                    if target == 'edit_dist':
                        target_values.append(str(change_ratio))
                    else:
                        if rev_succ != "same":
                            mapped_rev_succ = str(revision_success_mapping[rev_succ])
                            target_values.append(mapped_rev_succ)
                            values_per_instance["change_ratio"] = change_ratio
                    comment = row[header.index("comment")]
                    parsed_comment = nlp(comment)
                    comment_sents = list(parsed_comment.sents)
                    values_per_instance["comment_len_char"] = len(comment)
                    avg_sent_len = len(parsed_comment) / len(comment_sents)
                    values_per_instance["avg_sent_len"] = avg_sent_len
                    
                    # interrog_ratio # TO DO: look into spacy error (how to access tokens from sents)
                    #interrogatives = [comment_sent for comment_sent in comment_sents if comment_sent[-1] == '?']
                    #values_per_instance.append(len(interrogatives)/len(comment_sents))
                    #case
                    case_info = [char.isupper() for char in comment if char.isalpha()]
                    values_per_instance["upper_ratio"] = len([ch for ch in case_info if ch]) / len(case_info)
                    tkn_len = 0
                    puncts = {".":1, "?":1, "!":1} # additive smoothing
                    nr_1SG = 0
                    #nr_you = 0
                    #nr_it = 0
                    nr_hedge = 0
                    nr_meta = 0
                    nr_quote = 0
                    nr_pron = 0 
                    nr_noun = 1 # additive smoothing
                    nr_verb = 1 # additive smoothing
                    nr_adj = 0
                    nr_adv = 0
                    for token in parsed_comment:
                        tkn_len += len(token)
                        wordform = token.text
                        lemma = parsed_comment.vocab.strings[token.lemma]
                        pos = parsed_comment.vocab.strings[token.pos]
                        #print(pos)
                        #is_upper.append(wordform.isupper())
                        # has 1SG (subjectivity) 
                        if wordform == "I":
                            nr_1SG += 1
                            pos = "PRON"
                        #elif wordform.lower() == "you":
                        #    nr_you += 1
                        #    pos = "PRON"
                        #elif wordform.lower() == "it":
                        #    nr_it += 1
                        #    pos = "PRON"
                        if wordform in puncts:
                            puncts[wordform] += 1
                        if lemma.lower() in hedging_words:
                            nr_hedge += 1
                        if lemma.lower() in meta_ling_terms:
                            nr_meta += 1
                        if "'" == wordform or "\"" == wordform:
                            nr_quote += 1
                        if pos == "PRON":
                            nr_pron += 1
                        if pos == "NOUN":
                            nr_noun += 1
                        if pos == "VERB":
                            nr_verb += 1
                        if pos == "ADJ":
                            nr_verb += 1
                        if pos == "ADV":
                            nr_verb += 1
                    lexical_tokens = nr_noun + nr_verb + nr_adj + nr_adv - 2 #for smoothing
                    avg_tok_len = tkn_len/len(parsed_comment)
                    values_per_instance["avg_tok_len"] = avg_tok_len
                    if nr_1SG:
                        values_per_instance["1SG_ratio"] = nr_1SG / nr_pron
                    else:
                        values_per_instance["1SG_ratio"] = 0
                    #if nr_you:
                    #    values_per_instance["you_ratio"] = nr_you / nr_pron
                    #else:
                    #    values_per_instance["you_ratio"] = 0
                    #if nr_it:
                    #    values_per_instance["it_ratio"] = nr_it / nr_pron
                    #else:
                    #    values_per_instance["it_ratio"] = 0
                    values_per_instance["hedge_ratio"] = nr_hedge / len(parsed_comment)
                    #values_per_instance["hedge_ratio_lex"] = nr_hedge / len(parsed_comment)
                    values_per_instance["meta_ratio"] = nr_meta / len(parsed_comment)
                    values_per_instance["quote_ratio"] = nr_quote / len(comment.replace(" ",""))
                    values_per_instance["interrog_ratio"] = puncts["?"]/puncts["."]
                    values_per_instance["excl_ratio"] = puncts["!"]/puncts["."]
                    values_per_instance["nn_to_vb"] = nr_noun / nr_verb

                    if add_extra_var:
                        # LEARNER VARIABLES
                        file_info = row[header.index("essay ID")].split("_")
                        #values_per_instance.append(file_info[1]) # course code (e.g. CS, MS, SS, BCH)
                        
                        # version
                        version = ""
                        for elem in file_info:
                            if "version" in elem:
                                version = elem.replace("version", "")
                                values_per_instance["version"] = version
                        if not version:
                            print("no version in: ", row[header.index("essay ID")])

                        span_length = len(row[header.index("target token")].split(","))
                        values_per_instance["nr_target_tokens"] = span_length # lenght of target token span
                        error_pos = int(row[header.index("target token")].split(",")[0][1:])
                        values_per_instance["error_position"] = error_pos
                    # print info on features of each instance
                    #print([t for t in parsed_comment])
                    #for fname, val in values_per_instance.items():
                    #    print("\t", val, fname)

                    # add feature values per instance
                    feature_values.append(",".join([str(v) for k,v in sorted(values_per_instance.items())]))
                    feature_names = sorted(values_per_instance.keys())
                    

        assert len(feature_values), len(target_values)
        # save data
        with open(features_file, "w") as features_f:
            features_f.write("\n".join(feature_values))
        with open(label_file, "w") as target_f:
            target_f.write("\n".join(target_values))
        with open(fname_file, "w") as fn_f:
            fn_f.write("\n".join(feature_names))

def extract_features(data_file, features_file, label_file, fname_file, nlp, add_extra_var=False):
    """ Extract features for the sentence aligned dataset.
    @ data_file: CSV file with all annotation information summed (directness, revision success)
    @ features_file: file name to save feature values to 
    @ label_file: file name to save labels to predict to (= gold revision success annotation)  
    @ fname_file: file name for saving feature names to
    @ nlp: loaded Spacy NLP processing pipeline
    @ add_extra_var: 
    """
    with open("metaling.txt", newline='') as metafile:
        meta_ling_terms = [l.strip("\n") for l in metafile.readlines()]
    with open(data_file, newline='') as csvfile:
        csv_reader = list(csv.reader(csvfile, delimiter=','))
        header = ["align_type", "comment_type", "original", "revised", "comment/tag", "essay_id", 
                  "orig_sent_id", "target_token", "bug", "annotation"]
        align_mapping = {"identical":0, "delete":1, "split":2, "swap":2, "merge":2, "replace":3}
        feature_values = []
        target_values = []
        feature_names = []
        hedging_words = ['indicate', 'suggest', 'propose', 'predict', 'assume', 'speculate', 'suspect', 'believe', 
        'imply', 'estimate', 'calculate', 'report', 'note', 'appear', 'seem', 'attempt', 'seek', 'quite', 'partially', 
        'rarely', 'almost', 'approximately', 'generally', 'likely', 'probably', 'presumably', 'apparently', 'evidently', 
        'essentially', 'potentially', 'unlikely', 'possible', 'apparent', 'probable', 'most', 'would', 'may', 'could', 
        'might', 'possibility', 'estimate'] + \
        ["usually", "normally", "slightly", "occasionally", "virtually", "relatively"] + \
        ["assumption", "claim", "suggestion"]    + \
        ["try", "sound", "perhaps", "possibly", "little"] # own from most frequent unigrams
        # Hyland? +  Hyland + # http://www-di.inf.puc-rio.br/~endler/students/Hedging_Handout.pdf
        #print(len(hedging_words))
        for ix, row in enumerate(csv_reader[1:3060]): # file contains open-ended comments only
            print(ix)
            if row[0]:
                values_per_instance = {}
                annotation = row[header.index("annotation")]
                if len(annotation) > 1 and annotation[-1] == "?":
                    annotation = annotation[:-1]
                if annotation not in ["?", "irrel", "skip", "emb_com"]:
                    mapped_target = str(align_mapping[row[header.index("align_type")]])
                    if mapped_target != "0":
                        mapped_target = "3"
                    target_values.append(mapped_target)
                    comment = row[header.index("comment/tag")]
                    parsed_comment = nlp(comment)
                    comment_sents = list(parsed_comment.sents)
                    values_per_instance["comment_len_char"] = len(comment)
                    avg_sent_len = len(parsed_comment) / len(comment_sents)
                    values_per_instance["avg_sent_len"] = avg_sent_len
                    #span_length = len(row[header.index("target token")].split(","))
                    #values_per_instance["nr_target_tokens"] = span_length # lenght of target token span
                    case_info = [char.isupper() for char in comment if char.isalpha()]
                    values_per_instance["upper_ratio"] = len([ch for ch in case_info if ch]) / len(case_info)
                    tkn_len = 0
                    puncts = {".":1, "?":1, "!":1} # additive smoothing
                    nr_1SG = 0
                    nr_you = 0
                    nr_it = 0
                    nr_hedge = 0
                    nr_meta = 0
                    nr_quote = 0
                    nr_pron = 0 
                    nr_noun = 1 # additive smoothing
                    nr_verb = 1 # additive smoothing
                    nr_adj = 0
                    nr_adv = 0
                    for token in parsed_comment:
                        tkn_len += len(token)
                        wordform = token.text
                        lemma = parsed_comment.vocab.strings[token.lemma]
                        pos = parsed_comment.vocab.strings[token.pos]
                        #print(pos)
                        #is_upper.append(wordform.isupper())
                        # has 1SG (subjectivity) 
                        if wordform == "I":
                            nr_1SG += 1
                            pos = "PRON"
                        elif wordform.lower() == "you":
                           nr_you += 1
                           pos = "PRON"
                        elif wordform.lower() == "it":
                           nr_it += 1
                           pos = "PRON"
                        if wordform in puncts:
                            puncts[wordform] += 1
                        if lemma.lower() in hedging_words:
                            nr_hedge += 1
                        if lemma.lower() in meta_ling_terms:
                            nr_meta += 1
                        if "'" == wordform or "\"" == wordform:
                            nr_quote += 1
                        if pos == "PRON":
                            nr_pron += 1
                        if pos == "NOUN":
                            nr_noun += 1
                        if pos == "VERB":
                            nr_verb += 1
                        if pos == "ADJ":
                            nr_verb += 1
                        if pos == "ADV":
                            nr_verb += 1
                    lexical_tokens = nr_noun + nr_verb + nr_adj + nr_adv - 2 #for smoothing
                    avg_tok_len = tkn_len/len(parsed_comment)
                    values_per_instance["avg_tok_len"] = avg_tok_len
                    if nr_1SG:
                        values_per_instance["1SG_ratio"] = nr_1SG / nr_pron
                    else:
                        values_per_instance["1SG_ratio"] = 0
                    if nr_you:
                       values_per_instance["you_ratio"] = nr_you / nr_pron
                    else:
                       values_per_instance["you_ratio"] = 0
                    if nr_it:
                       values_per_instance["it_ratio"] = nr_it / nr_pron
                    else:
                       values_per_instance["it_ratio"] = 0
                    values_per_instance["hedge_ratio"] = nr_hedge / len(parsed_comment)
                    #values_per_instance["hedge_ratio_lex"] = nr_hedge / len(parsed_comment)
                    values_per_instance["hedge_nr"] = nr_hedge
                    values_per_instance["meta_ratio"] = nr_meta / len(parsed_comment)
                    values_per_instance["quote_ratio"] = nr_quote / len(comment.replace(" ",""))
                    values_per_instance["interrog_ratio"] = puncts["?"]/puncts["."]
                    values_per_instance["excl_ratio"] = puncts["!"]/puncts["."]
                    values_per_instance["nn_to_vb"] = nr_noun / nr_verb

                    if add_extra_var:
                        # LEARNER VARIABLES
                        file_info = row[header.index("essay ID")].split("_")
                        #values_per_instance.append(file_info[1]) # course code (e.g. CS, MS, SS, BCH)
                        
                        # version
                        version = ""
                        for elem in file_info:
                            if "version" in elem:
                                version = elem.replace("version", "")
                                values_per_instance["version"] = version
                        if not version:
                            print("no version in: ", row[header.index("essay ID")])

                        # change_ratio
                        edit_dist = nltk.edit_distance(row[header.index("original")], row[header.index("revised")])
                        change_ratio = edit_dist / len(row[header.index("original")])
                        values_per_instance["change_ratio"] = change_ratio

                    # print info on features of each instance
                    #print([t for t in parsed_comment])
                    #for fname, val in values_per_instance.items():
                    #    print("\t", val, fname)

                    # add feature values per instance
                    feature_values.append(",".join([str(v) for k,v in sorted(values_per_instance.items())]))
                    feature_names = sorted(values_per_instance.keys())
        assert len(feature_values), len(target_values)
        # save data
        with open(features_file, "w") as features_f:
            features_f.write("\n".join(feature_values))
        with open(label_file, "w") as target_f:
            target_f.write("\n".join(target_values))
        with open(fname_file, "w") as fn_f:
            fn_f.write("\n".join(feature_names))      


