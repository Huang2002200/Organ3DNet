# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import os, sys, argparse
from copy import deepcopy
from uuid import uuid4
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
import benchmark.util as util
import benchmark.util_3d as util_3d



# ---------- Label info ---------- #
CLASS_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([100])
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])


def evaluate_matches(matches):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros(
        (len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float
    )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 1000
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = (
                                float(num_ignore) / pred["vert_count"]
                            )
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    num_true_examples = (
                        y_true_sorted_cumsum[-1]
                        if len(y_true_sorted_cumsum) > 0
                        else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(
                        recall_for_conv[0], recall_for_conv
                    )
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(
                        recall_for_conv, [-0.5, 0, 0.5], "valid"
                    )
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (
        pred["pred_classes"].shape[0]
        == pred["pred_scores"].shape[0]
        == pred["pred_masks"].shape[1]
    )
    for i in range(len(pred["pred_classes"])):
        info = {}
        info["label_id"] = pred["pred_classes"][i]
        info["conf"] = pred["pred_scores"][i]
        info["mask"] = pred["pred_masks"][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str):
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error("unable to load " + gt_file + ": " + str(e))

    # get gt instances
    gt_instances = util_3d.get_instances(
        gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL
    )
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt["matched_pred"] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]["label_id"])
        conf = pred_info[uuid]["conf"]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]["mask"]
        assert len(pred_mask) == len(gt_ids)
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance["uuid"] = uuid
        pred_instance["pred_id"] = num_pred_instances
        pred_instance["label_id"] = label_id
        pred_instance["vert_count"] = num
        pred_instance["confidence"] = conf
        pred_instance["void_intersection"] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask)
        )

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask)
            )
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ","
    with open(filename, "w") as f:
        f.write(
            _SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n"
        )
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(
                _SPLITTER.join(
                    [str(x) for x in [class_name, class_id, ap, ap50, ap25]]
                )
                + "\n"
            )


def evaluate(
    preds: dict, gt_path: str, output_file: str, dataset: str = "5plant", output_premask: bool=False,
):
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt
    output_premask = output_premask
    if dataset == "5plant":

        CLASS_LABELS = [
            #"no class"
            "benthi_stem",
            "benthi_leaf",
            "tomato_stem",
            "tomato_leaf",
            "sorghum_stem",
            "sorghum_leaf",
            "soybean_stem",
            "soybean_leaf",
            "pepper_stem",
            "pepper_leaf",

        ]
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6,7,8,9,10])
        ID_TO_LABEL = {}
        LABEL_TO_ID = {}
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    total_true = 0
    total_seen = 0
    NUM_CLASSES = len(VALID_CLASS_IDS)

    true_positive_classes = np.zeros(NUM_CLASSES)
    positive_classes = np.zeros(NUM_CLASSES)
    false_negative_classes = np.zeros(NUM_CLASSES)
    false_positive_classes = np.zeros(NUM_CLASSES)
    gt_classes = np.zeros(NUM_CLASSES)

    # precision & recall
    total_gt_ins = np.zeros(NUM_CLASSES)
    at = 0.5
    tpsins = [[] for _ in range(NUM_CLASSES)]
    fpsins = [[] for _ in range(NUM_CLASSES)]
    # mucov and mwcov
    all_mean_cov = [[] for _ in range(NUM_CLASSES)]
    all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

    print("evaluating", len(preds), "scans...")
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        gt_file = os.path.join(gt_path, k + ".txt")
        coords_path = 'data/processed/5plant/Area_2'
        coords_file= os.path.join(coords_path, k+'.npy')
        gt_filename = os.path.basename(gt_file)
        output_filename = os.path.join('data/pred_5plant_best', gt_filename)
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.isfile(gt_file):
            util.print_error(
                "Scan {} does not match any gt file".format(k), user_fault=True
            )

        if dataset == "5plant":
            coords_data = np.load(coords_file)
            coords = coords_data[:, :3]
            gt_ids = util_3d.load_ids(gt_file)
            gt_sem = (gt_ids // 1000) - 1  #origin

            gt_ins = gt_ids - (gt_ids // 1000) * 1000  #min:1

            pred_sem = np.full(v["pred_masks"].shape[0], -1, dtype=np.int)  # -1 means uncovered points
            pred_ins = np.full(v["pred_masks"].shape[0], -1, dtype=np.int)

            for inst_id in reversed(range(v["pred_masks"].shape[1])):
                point_ids = np.argwhere(v["pred_masks"][:, inst_id] == 1.0)[:, 0]
                pred_ins[point_ids] = inst_id + 1
                pred_sem[point_ids] = v["pred_classes"][inst_id] - 1

            instance_counts = Counter(pred_ins)

            small_instances = {inst_id for inst_id, count in instance_counts.items() if count < 100 and inst_id > 0}
            for inst_id in small_instances:
                point_ids = np.argwhere(pred_ins == inst_id)[:, 0]
                pred_ins[point_ids] = -1  # Set the instance label of small instances to -1

            knn = NearestNeighbors(n_neighbors=5)

            valid_points = np.argwhere(pred_ins != -1)[:,0]
            valid_coords = coords[valid_points]
            valid_ins = pred_ins[valid_points]
            valid_sem = pred_sem[valid_points]
            invalid_points = np.argwhere(pred_ins == -1)[:, 0]
            invalid_coords = coords[invalid_points]
            if len(invalid_points) == 0:
                print("skip KNN ")
            else:
                knn.fit(valid_coords)

                # Find the nearest points for invalid points
                _, nearest_indices = knn.kneighbors(invalid_coords)

                for i, invalid_idx in enumerate(invalid_points):
                    neighbor_ins_labels = valid_ins[nearest_indices[i]]  # Shape: (5,)
                    majority_ins_label = int(mode(neighbor_ins_labels, keepdims=False)[0])
                    pred_ins[invalid_idx] = majority_ins_label
                    pred_sem[invalid_idx] = valid_sem[valid_ins == majority_ins_label][0]
            if output_premask:
                pred_label = np.column_stack((coords, pred_ins, pred_sem))
                np.savetxt(output_filename, pred_label, fmt='%f %f %f %d %d', delimiter=' ')


            # semantic acc
            total_true += np.sum(pred_sem == gt_sem)
            total_seen += pred_sem.shape[0]
            uniq, counts = np.unique(pred_sem, return_counts=True)
            positive_classes[uniq] += counts

            uniq, counts = np.unique(gt_sem, return_counts=True)
            gt_classes[uniq] += counts

            uniq, counts = np.unique(
                gt_sem[pred_sem == gt_sem], return_counts=True
            )
            true_positive_classes[uniq] += counts

            uniq, counts = np.unique(pred_sem[pred_sem != gt_sem], return_counts=True)
            false_positive_classes[uniq] += counts

            uniq, counts = np.unique(
                gt_sem[pred_sem != gt_sem], return_counts=True
            )
            false_negative_classes[uniq] += counts

            # instance
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):  # each object in prediction
                if g == -1:
                    continue
                tmp = pred_ins == g
                sem_seg_i = int(stats.mode(pred_sem[tmp], keepdims=True)[0])
                pts_in_pred[sem_seg_i] += [tmp]

            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):
                tmp = gt_ins == g
                sem_seg_i = int(stats.mode(pred_sem[tmp], keepdims=True)[0])
                pts_in_gt[sem_seg_i] += [tmp]

            # instance mucov & mwcov
            for i_sem in range(NUM_CLASSES):
                sum_cov = 0
                mean_cov = 0
                mean_weighted_cov = 0
                num_gt_point = 0
                max_iou = []
                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    ovmax = 0.0
                    num_ins_gt_point = np.sum(ins_gt)
                    num_gt_point += num_ins_gt_point

                    for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)
                        if iou > ovmax:
                            ovmax = iou
                    max_iou.append(ovmax)

                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point
                if len(pts_in_gt[i_sem]) != 0:
                    mean_cov = sum_cov / len(pts_in_gt[i_sem])
                    all_mean_cov[i_sem].append(mean_cov)
                    mean_weighted_cov /= num_gt_point
                    all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

        if dataset == "5plant":
            # instance precision & recall
            for i_sem in range(NUM_CLASSES):
                tp = [0.0] * len(pts_in_pred[i_sem])
                fp = [0.0] * len(pts_in_pred[i_sem])
                gtflag = np.zeros(len(pts_in_gt[i_sem]))
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.0

                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            igmax = ig

                    if ovmax >= at:
                        tp[ip] = 1  # true
                    else:
                        fp[ip] = 1  # false positive

                tpsins[i_sem] += tp
                fpsins[i_sem] += fp

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(v, gt_file)
        matches[matches_key] = {}
        matches[matches_key]["gt"] = gt2pred
        matches[matches_key]["pred"] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print("")
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    # print
    print_results(avgs)
    write_result_file(avgs, output_file)

    if dataset == "5plant":
        MUCov = np.zeros(NUM_CLASSES)
        MWCov = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
            MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

        ins_precision = np.zeros(NUM_CLASSES)
        ins_recall = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            tp = np.asarray(tpsins[i_sem]).astype(float)
            fp = np.asarray(fpsins[i_sem]).astype(float)
            tp = np.sum(tp)
            fp = np.sum(fp)
            rec = tp / total_gt_ins[i_sem]
            prec = tp / (tp + fp)

            ins_precision[i_sem] = prec
            ins_recall[i_sem] = rec
        print('ins_precision:',ins_precision)
        print('ins_recall:',ins_recall)
        print('MUCov:',MUCov)
        print('MWCov:',MWCov)

         # semantic results
        iou_list = []
        for i in range(NUM_CLASSES):
            iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
            iou_list.append(iou)

        precision_sem = true_positive_classes / (false_positive_classes + true_positive_classes)
        print('precision_sem:',precision_sem)
        recall_sem = true_positive_classes / (false_negative_classes + true_positive_classes)
        print('recall_sem:',recall_sem)
        f1_score = 2*precision_sem*recall_sem/(precision_sem+recall_sem)
        print('f1-score:',f1_score)
        print('iou:', iou_list)
        """
        LOG_FOUT = open(os.path.join('results_a5.txt'), 'w')
    
        def log_string(out_str):
            LOG_FOUT.write(out_str + '\n')
            LOG_FOUT.flush()
            print(out_str)
        """

        return (np.mean(ins_precision[[1, 3, 5, 7, 9]]), np.mean(ins_recall[[1, 3, 5, 7, 9]]),
                np.mean(MUCov[[1, 3, 5, 7, 9]]), np.mean(MWCov[[1, 3, 5, 7, 9]]), np.mean(precision_sem),
                np.mean(recall_sem), np.mean(f1_score), 1.*sum(iou_list)/NUM_CLASSES)
