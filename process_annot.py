# Functions for processing the annotated data (directness and revision success)

from collections import Counter
import csv
import os

def get_directness(annot_csv_file, delimiter=",", print_info=True):
    """ Collects directness annotation of an annotator from the 
    annotation file. Returns a dictionary with annotation item 
    id as key and annotation label as value.
    """
    stats = Counter()
    annotations = {}
    with open(annot_csv_file, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in list(csv_reader)[1:]:
            if row:
                if row[0]:
                    if len(row) > 2:
                        try:
                            annot_val = row[2].replace("?", "")
                        except IndexError:
                            annot_val = ""
                        if annot_val:
                            annotations[row[0]] = annot_val
                            stats[annot_val] += 1
    if print_info:
        print("Nr annotations {:<12}: {:<5}".format(annot_csv_file.split("/")[-1].split(".")[0][10:], 
                                                  len(annotations)))
    return annotations

def get_revision_success(annot_csv_file, delimiter=",", print_info=True):
    """ Collects revision success annotation of an annotator from the 
    annotation file. Returns a dictionary with annotation item 
    id as key and annotation label as value.
    """
    stats = Counter()
    annotations = {}
    with open(annot_csv_file, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter)
        for row in list(csv_reader)[1:]:
            #print(row)
            if row:
                try:
                    annot_val = row[4]
                except IndexError:
                    annot_val = "" # "-"
                if annot_val: 
                    if len(annot_val) > 1 and annot_val[-1] == "?":
                        annot_val = annot_val[:-1]    
                    annotations[row[0]] = annot_val
                    stats[annot_val] += 1

    if print_info:
        print("Nr annotations {:<12}: {:<5}".format(annot_csv_file.split("/")[-1].split(".")[0][10:], 
                                                  len(annotations)))
    return annotations

def prep_iaa_data(all_annotations):
    # Returns a list of (annotator, item, label) tuples.
    # NLTK AnnotationTask input data format.
    iaa_data = []
    for ix, annotations in enumerate(all_annotations):
        annotator_id = "a" + str(ix+1)
        for annotation_item, label in annotations.items():
            iaa_data.append((annotator_id, annotation_item, label))
    return iaa_data

def sum_annot_data(all_annotations):
    """ Collects annotation labels per item from all annotators.
    Returns a dictionary with annotation item id as key and a 
    list of annotation labels assigned by all annotators.
    """
    iaa_data = prep_iaa_data(all_annotations)
    data = {}
    for (annotator, item, label) in iaa_data:
        if label not in ["same", "rem", "skip", "-", ""]:
            if item in data:
                data[item].append(label)
            else:
                data[item] = [label]
    return data

def get_disagreements(summed_annot_data, print_info=True):
    """ Collects item ids with disagreement between annotators and 
    computes percentage agreement. Returns a dictionary with item 
    ids as keys and the list of labels assigned by annotators. 
    """
    disagr = 0
    disagr_items = {}
    nr_double_annot = 0
    for item, labels in summed_annot_data.items():
        if len(labels) > 1:
            #print(labels)
            nr_double_annot += 1
            if len(set(labels)) > 1:
                disagr += 1
                disagr_items[item] = labels
    if print_info:
        print("Nr of disagreements:    {}".format(disagr))
        print("Nr of double annotated: {}".format(nr_double_annot))
        print("Percentage agreement:   {}".format(round(100-(disagr/nr_double_annot*100),2)))
    return disagr_items

def get_ordered_labels(summed_annot_data):
    """ Returns a tuple of list of annotation labels per annotator.
    """
    labels_A1 = []
    labels_A2 = []
    for item, labels in summed_annot_data.items():
        if len(labels) > 1:
            labels_A1.append(labels[0])
            labels_A2.append(labels[1])
    return (labels_A1, labels_A2)

def get_kappa(labels_A1, labels_A2, kappa_metric, print_info=True):
    """ Computes Cohen's kappa.
    """
    kappa = round(kappa_metric(labels_A1, labels_A2),2)
    if print_info:
        print("Kappa: \t", kappa)
    return kappa

def get_confusion_matrix(labels_A1, labels_A2, label_names, confusion_matrix, np, sns, plt):
    """ Creates a confusion matrix over the annotation labels used.
    @ np: imported numpy module
    @ sns: imported sns module
    Returns an array with [n_lables, n_labels], see the sklearn documentation for more:
    <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix>
    """
    cm = confusion_matrix(np.array(labels_A1), np.array(labels_A2))  # first annotator on x axis
    with sns.axes_style("white"):
        ax = sns.heatmap(cm, vmin=0, vmax=50, xticklabels=label_names, yticklabels=label_names,
                    annot=True, fmt="d", cmap="Blues")
    plt.show()
    return cm

def compute_iaa(all_annotations, kappa_metric, label_names, confusion_matrix, np, sns, plt):
    """ Sums annotation data across annotators, computes percentage 
    agreement and Cohen's kappa. 
    @ all_annotations: list of dictionary objects with annotation
    data per annotator.
    @ kappa_metric:    sklearn.metrics.cohen_kappa_score
    """
    summed_annot_data = sum_annot_data(all_annotations)
    get_disagreements(summed_annot_data)
    labels_A1, labels_A2 = get_ordered_labels(summed_annot_data)
    get_kappa(labels_A1, labels_A2, kappa_metric)
    get_confusion_matrix(labels_A1, labels_A2, label_names, confusion_matrix, np, sns, plt)


def merge_all_annot_info(open_info_csv, tag_info_csv, same_rem_annot, annot_DIR_A1_csv, annot_DIR_A2_csv, 
                         annot_RS_A1_csv, annot_RS_A2_csv, annot_RS_A2_csv_part2):
    """ Creates a CSV file containing all annotation information and the original location
    of each annotation item.
    Items can be located in the original corpus data based on (i) column B ('essay ID') which 
    corresponds to the semester folder + the XML filename and (ii) column C ('target token')
    indicating the relevant XML <w> elements' xml:id attribute.
    @ open_info_csv:  file with info (e.g. location) for feedback with open-ended comments
    @ tag_info_csv:   file with info (e.g. location) for feedback with Commentbank tags,
                      ids c700 up  
    @ same_rem_annot: includes items with 'same' / 'rem' annotation (preceeding the actual annotation)
                      ids from c1300 up
    """
    info_list = []
    ids = []
    nr_open = 0
    # directness
    annotations_DIR_A1 = get_directness(annot_DIR_A1_csv)
    annotations_DIR_A2 = get_directness(annot_DIR_A2_csv)
    summed_annot_data_DIR = sum_annot_data([annotations_DIR_A1, annotations_DIR_A2])
    #revision success
    annotations_RS_A1 = get_revision_success(annot_RS_A1_csv) 
    annotations_RS_A2 = get_revision_success(annot_RS_A2_csv)
    annotations_RS_A1.update(get_revision_success(annot_RS_A2_csv_part2))
    summed_annot_data_RS = sum_annot_data([annotations_RS_A1, annotations_RS_A2])
    # annotated open-ended and tagged feedback
    with open(open_info_csv, newline="") as csvfile:
        csv_reader = list(csv.reader(csvfile))#[:10]
        with open(tag_info_csv, newline="") as csvfile:
            csv_reader_tag = list(csv.reader(csvfile))
        csv_reader.extend(csv_reader_tag[1:])    
        new_row = []
        for ix, row in enumerate(csv_reader):
            new_row = [row[0], row[3], row[4], row[1]]      # ids, location, comment
            if ix:
                if row[3] + row[4] not in ids:
                    ids.append(row[3] + row[4])
                    # directness annot
                    if int(row[0][1:]) < 700:               # directness for open
                        nr_open += 1
                        directness_annot = summed_annot_data_DIR.get(row[0])
                        if directness_annot:     
                            for i in range(2-len(directness_annot)):
                                directness_annot.append("")
                        else:
                            directness_annot = ["", ""]
                        gold_directness = directness_annot[1]
                        directness_annot.append(gold_directness)
                        new_row.extend(directness_annot)
                    else:
                        new_row.extend(["", "", row[2]])    # directness for tagged
                    # revision success annot
                    rev_succ_annot = summed_annot_data_RS.get(row[0])
                    if rev_succ_annot:     
                        for i in range(2-len(rev_succ_annot)):
                            rev_succ_annot.append("")
                    else:
                        rev_succ_annot = ["", ""]
                    gold_rev_succ = rev_succ_annot[0]
                    rev_succ_annot.append(gold_rev_succ)
                    new_row.extend(rev_succ_annot)
                    new_row.extend([row[7], row[8], row[10], row[11]]) # original and revised
                    info_list.append(new_row)
            else:
                new_row.extend(["direct_A1", "direct_A2", "gold_dir", 
                                "rev_succ_A1", "rev_succ_A2", "gold_rev_succ"])
                new_row.extend([row[7], row[8], row[10], row[11]]) # original and revised
                info_list.append(new_row)

    # adding tagged pre-annotated with 'same' and 'rem'
    with open(same_rem_annot, newline="") as csvfile:
        csv_reader_tag_pre_annot = list(csv.reader(csvfile))#[:10]
        new_row = []
        nr_preannot = 0
        for ix, row in enumerate(csv_reader_tag_pre_annot[1:]):
            # revision success
            rev_succ_pre_annot = row[5]
            if rev_succ_pre_annot in ["same", "rem"]:
                nr_preannot += 1
                item_id = "c"+str(1300+nr_preannot)
                if item_id not in ids:
                    ids.append(row[0] + row[2])
                    new_row = [item_id, row[0], row[2], row[1]] # ids, location, comment
                    # directness
                    if row[1] == "Delete this (unnecessary)":            
                        new_row.extend(["", "", "dir"])
                    else:
                        new_row.extend(["", "", "ind"])
                    new_row.extend(["", "", rev_succ_pre_annot])     # revision success (same/rem)
                    new_row.extend([row[6], row[7], row[8], row[9]]) # original and revised
                    info_list.append(new_row)
    print("Nr of LA-relevant datapoints: ", len(info_list))
    print("Nr open-ended:", nr_open)
    print(len(ids), len(set(ids)))
    # save output    
    out_csvfile_name = "all_annot_info.csv"
    with open(out_csvfile_name, "w", newline="") as out_csvfile:
        csv_writer = csv.writer(out_csvfile)
        for out_row in info_list:
            csv_writer.writerow(out_row) 
    print("Annotation results saved in {}".format(out_csvfile_name))
    