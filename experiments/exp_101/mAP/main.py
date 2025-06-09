#!/usr/bin/env python
# coding=utf-8

import os
# Suppress TensorFlow INFO and WARNING messages (including oneDNN messages)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import json
import shutil
import operator
import sys
import argparse
import time
import copy
import numpy as np
import warnings

# Silence matplotlib tight_layout warning.
warnings.filterwarnings("ignore", category=UserWarning, message="Tight layout not applied.*")

# Set working directory to this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# mAP@50 threshold (VOC style)
MINOVERLAP = 0.5  # default value

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

if args.ignore is None:
    args.ignore = []

specific_iou_flagged = args.set_class_iou is not None

# Check for images (for animation)
img_path = 'images'
if os.path.exists(img_path):
    for _, _, files in os.walk(img_path):
        if not files:
            args.no_animation = True
else:
    args.no_animation = True

show_animation = False
if not args.no_animation:
    try:
        import cv2
        show_animation = True
    except ImportError:
        print("\"opencv-python\" not found, please install to visualize the results.")
        args.no_animation = True

draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt
        draw_plot = True
    except ImportError:
        print("\"matplotlib\" not found, please install it to get the resulting plots.")
        args.no_plot = True

def error(msg):
    print(msg)
    sys.exit(0)

def is_float_between_0_and_1(value):
    try:
        val = float(value)
        return 0.0 < val < 1.0
    except ValueError:
        return False

def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = [i for i in range(1, len(mrec)) if mrec[i] != mrec[i-1]]
    ap = sum((mrec[i]-mrec[i-1])*mpre[i] for i in i_list)
    return ap, mrec, mpre

def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    return [x.strip() for x in content]

def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    cv2.putText(img, text, pos, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*proportion])

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    if true_p_bar != "":
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        plt.legend(loc='lower right')
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_str_val = " " + str(fp_sorted[i])
            tp_str_val = fp_str_val + " " + str(tp_sorted[i])
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)
    # Set window title using the manager
    fig.canvas.manager.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize='large')
    fig.tight_layout()
    fig.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()

def draw_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, output_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def draw_evaluation_summary(metrics_dict, output_path, to_show=True):
    # Create a summary plot showing all overall metrics
    if not draw_plot:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    summary_text = "Evaluation Summary\n" + ("=" * 50) + "\n"
    for key, value in metrics_dict.items():
        summary_text += f"{key}: {value}\n"
    ax.text(0.5, 0.5, summary_text, horizontalalignment='center', verticalalignment='center', fontsize=12, transform=ax.transAxes)
    plt.title("Evaluation Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    if to_show:
        plt.show()
    plt.close()

# Create temporary directories and results directory
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path):
    os.makedirs(tmp_files_path)
results_files_path = "results"
if os.path.exists(results_files_path):
    shutil.rmtree(results_files_path)
os.makedirs(results_files_path)
if draw_plot:
    os.makedirs(os.path.join(results_files_path, "classes"))
if show_animation:
    os.makedirs(os.path.join(results_files_path, "images"))
    os.makedirs(os.path.join(results_files_path, "images", "single_predictions"))

# -------------------------
# Read parameters from config file (assumed to be ../config.txt)
# -------------------------
config_path = os.path.join("..", "config.txt")
parameters_info = "N/A"
fps_config = "N/A"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config_lines = f.read().strip().splitlines()
    parameters_info = "\n".join(config_lines)
    for line in config_lines:
        if line.lower().startswith("fps"):
            fps_config = line.split(":", 1)[1].strip()

# -------------------------
# Process Ground-Truth Files
# -------------------------
ground_truth_files_list = glob.glob('ground-truth/*.txt')
if len(ground_truth_files_list) == 0:
    error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
gt_counter_per_class = {}

for txt_file in ground_truth_files_list:
    file_id = os.path.basename(os.path.splitext(txt_file)[0])
    if not os.path.exists('predicted/' + file_id + ".txt"):
        error("Error. File not found: predicted/" + file_id + ".txt\n(You can avoid this error by running extra/intersect-gt-and-pred.py)")
    lines_list = file_lines_to_list(txt_file)
    bounding_boxes = []
    is_difficult = False
    for line in lines_list:
        try:
            if "difficult" in line:
                class_name, left, top, right, bottom, _ = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()
        except ValueError:
            error("Error: File " + txt_file + " in the wrong format.\nExpected: <class_name> <left> <top> <right> <bottom> ['difficult']\nReceived: " + line)
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if is_difficult:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
            gt_counter_per_class[class_name] = gt_counter_per_class.get(class_name, 0) + 1
    with open(os.path.join(tmp_files_path, file_id + "_ground_truth.json"), 'w') as outfile:
        json.dump(bounding_boxes, outfile)

gt_classes = sorted(list(gt_counter_per_class.keys()))
n_classes = len(gt_classes)

if specific_iou_flagged:
    n_args = len(args.set_class_iou)
    error_msg = '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)
    specific_iou_classes = args.set_class_iou[::2]
    iou_list = args.set_class_iou[1::2]
    if len(specific_iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)
    for tmp_class in specific_iou_classes:
        if tmp_class not in gt_classes:
            error('Error, unknown class "' + tmp_class + '". Flag usage:' + error_msg)
    for num in iou_list:
        if not is_float_between_0_and_1(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

# -------------------------
# Process Predicted Files
# -------------------------
predicted_files_list = glob.glob('predicted/*.txt')
predicted_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in predicted_files_list:
        file_id = os.path.basename(os.path.splitext(txt_file)[0])
        if class_index == 0:
            if not os.path.exists('ground-truth/' + file_id + ".txt"):
                error("Error. File not found: ground-truth/" + file_id + ".txt\n(You can avoid this error by running extra/intersect-gt-and-pred.py)")
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
            except ValueError:
                error("Error: File " + txt_file + " in the wrong format.\nExpected: <class_name> <confidence> <left> <top> <right> <bottom>\nReceived: " + line)
            if tmp_class_name[1:] == class_name[1:]:
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
    with open(os.path.join(tmp_files_path, class_name + "_predictions.json"), 'w') as outfile:
        json.dump(bounding_boxes, outfile)

# -------------------------
# Compute mAP@50 (VOC style)
# -------------------------
start_eval = time.time()
sum_AP = 0.0
ap_dictionary = {}
count_true_positives = {}
with open(os.path.join(results_files_path, "results.txt"), 'w') as results_file:
    results_file.write("# AP and precision/recall per class (mAP@50)\n")
    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0
        predictions_file = os.path.join(tmp_files_path, class_name + "_predictions.json")
        predictions_data = json.load(open(predictions_file))
        nd = len(predictions_data)
        tp = [0] * nd
        fp = [0] * nd
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction["file_id"]
            gt_file = os.path.join(tmp_files_path, file_id + "_ground_truth.json")
            ground_truth_data = json.load(open(gt_file))
            ovmax = -1
            gt_match = None
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in ground_truth_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = ((bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) +
                              (bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1) - iw*ih)
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            min_overlap = MINOVERLAP
            if specific_iou_flagged and class_name in specific_iou_classes:
                index = specific_iou_classes.index(class_name)
                min_overlap = float(iou_list[index])
            if ovmax >= min_overlap:
                if gt_match is not None and "difficult" not in gt_match:
                    if not bool(gt_match.get("used", False)):
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = [float(tp[i]) / gt_counter_per_class[class_name] for i in range(len(tp))]
        prec = [float(tp[i]) / (fp[i] + tp[i]) if (fp[i]+tp[i]) > 0 else 0 for i in range(len(tp))]
        ap, mrec, mprec = voc_ap(rec, prec)
        ap_dictionary[class_name] = ap
        sum_AP += ap
        text = "{0:.2f}% = {1} AP".format(ap*100, class_name)
        results_file.write(text + "\n Precision: " + str(prec) + "\n Recall: " + str(rec) + "\n\n")
        if not args.quiet:
            print(text)
    mAP50 = sum_AP / n_classes if n_classes > 0 else 0
    results_file.write("\n# mAP@50 of all classes\n")
    results_file.write("mAP@50 = {0:.2f}%\n".format(mAP50*100))
eval_time = time.time() - start_eval
fps = len(ground_truth_files_list) / eval_time if eval_time > 0 else 0

# -------------------------
# Rebuild ground truth data for mAP@50-95 computation
# (Reset "used" flags by re-reading the original ground-truth files)
# -------------------------
gt_data_master = {}
for txt_file in ground_truth_files_list:
    file_id = os.path.basename(os.path.splitext(txt_file)[0])
    lines_list = file_lines_to_list(txt_file)
    bounding_boxes = []
    is_difficult = False
    for line in lines_list:
        try:
            if "difficult" in line:
                class_name, left, top, right, bottom, _ = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()
        except ValueError:
            error("Error: File " + txt_file + " in the wrong format.\nExpected: <class_name> <left> <top> <right> <bottom> ['difficult']\nReceived: " + line)
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " + bottom
        if is_difficult:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
            is_difficult = False
        else:
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
    gt_data_master[file_id] = bounding_boxes

# -------------------------
# Compute mAP@50-95 (averaging over thresholds 0.50 to 0.95)
# -------------------------
def compute_mAP_for_threshold(threshold, gt_data_master):
    sum_AP_local = 0.0
    ap_dict_local = {}
    for class_name in gt_classes:
        predictions_file = os.path.join(tmp_files_path, class_name + "_predictions.json")
        predictions_data = json.load(open(predictions_file))
        nd = len(predictions_data)
        tp = [0] * nd
        fp = [0] * nd
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction["file_id"]
            gt_data = copy.deepcopy(gt_data_master[file_id])
            ovmax = -1
            gt_match = None
            bb = [float(x) for x in prediction["bbox"].split()]
            for obj in gt_data:
                if obj["class_name"] == class_name:
                    bbgt = [float(x) for x in obj["bbox"].split()]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = ((bb[2]-bb[0]+1)*(bb[3]-bb[1]+1) +
                              (bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1) - iw*ih)
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
            if ovmax >= threshold:
                if gt_match is not None and "difficult" not in gt_match:
                    if not bool(gt_match.get("used", False)):
                        tp[idx] = 1
                        gt_match["used"] = True
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = [float(tp[i]) / gt_counter_per_class[class_name] for i in range(len(tp))]
        prec = [float(tp[i]) / (fp[i] + tp[i]) if (fp[i]+tp[i]) > 0 else 0 for i in range(len(tp))]
        ap, _, _ = voc_ap(rec, prec)
        ap_dict_local[class_name] = ap
        sum_AP_local += ap
    mAP_local = sum_AP_local / n_classes if n_classes > 0 else 0
    return mAP_local, ap_dict_local

thresholds = [round(0.5 + 0.05*i, 2) for i in range(10)]  # thresholds 0.50, 0.55, ..., 0.95
mAP_list = []
for th in thresholds:
    mAP_th, _ = compute_mAP_for_threshold(th, gt_data_master)
    mAP_list.append(mAP_th)
mAP50_95 = sum(mAP_list) / len(mAP_list) if mAP_list else 0

# -------------------------
# Compute Overall Recall and Precision
# -------------------------
total_tp = sum(count_true_positives.get(c, 0) for c in gt_classes)
total_gt = sum(gt_counter_per_class[c] for c in gt_classes)
total_pred = 0
for txt_file in predicted_files_list:
    lines = file_lines_to_list(txt_file)
    total_pred += len(lines)
overall_recall = total_tp / total_gt if total_gt > 0 else 0
overall_precision = total_tp / total_pred if total_pred > 0 else 0

# -------------------------
# Compute Confusion Matrix
# -------------------------
confusion_matrix = {gt: {pred: 0 for pred in gt_classes} for gt in gt_classes}
for txt_file in ground_truth_files_list:
    file_id = os.path.basename(os.path.splitext(txt_file)[0])
    gt_data = json.load(open(os.path.join(tmp_files_path, file_id + "_ground_truth.json")))
    predicted_file = os.path.join('predicted', file_id + ".txt")
    if os.path.exists(predicted_file):
        lines = file_lines_to_list(predicted_file)
        predictions = []
        for line in lines:
            try:
                cls, conf, left, top, right, bottom = line.split()
            except ValueError:
                continue
            predictions.append({"class": cls, "bbox": [float(left), float(top), float(right), float(bottom)]})
        for pred in predictions:
            best_iou = 0
            best_gt = None
            for gt in gt_data:
                if gt["class_name"] in args.ignore:
                    continue
                bb_pred = pred["bbox"]
                bb_gt = [float(x) for x in gt["bbox"].split()]
                bi = [max(bb_pred[0], bb_gt[0]), max(bb_pred[1], bb_gt[1]),
                      min(bb_pred[2], bb_gt[2]), min(bb_pred[3], bb_gt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = ((bb_pred[2]-bb_pred[0]+1)*(bb_pred[3]-bb_pred[1]+1) +
                          (bb_gt[2]-bb_gt[0]+1)*(bb_gt[3]-bb_gt[1]+1) - iw*ih)
                    iou = iw * ih / ua
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_iou >= 0.5 and best_gt is not None:
                gt_cls = best_gt["class_name"]
                pred_cls = pred["class"]
                if gt_cls in confusion_matrix and pred_cls in confusion_matrix[gt_cls]:
                    confusion_matrix[gt_cls][pred_cls] += 1

# Convert confusion matrix dict into a numpy array for plotting
cm_array = np.zeros((len(gt_classes), len(gt_classes)), dtype=int)
for i, gt in enumerate(gt_classes):
    for j, pred in enumerate(gt_classes):
        cm_array[i, j] = confusion_matrix[gt][pred]

if draw_plot:
    draw_confusion_matrix(cm_array, gt_classes, title="Confusion Matrix", output_path=os.path.join(results_files_path, "confusion_matrix.png"))

# -------------------------
# Write Overall Metrics and Parameters Info to File and Prepare Summary Data
# -------------------------
overall_metrics = {
    "mAP@50": f"{mAP50*100:.2f}%",
    "mAP@50-95": f"{mAP50_95*100:.2f}%",
    "Overall Recall": f"{overall_recall*100:.2f}%",
    "Overall Precision": f"{overall_precision*100:.2f}%",
    "FPS": f"{fps:.2f}",
    "Parameters Info": parameters_info.replace("\n", " | ")
}

with open(os.path.join(results_files_path, "results.txt"), 'a') as results_file:
    results_file.write("\n# Overall Metrics\n")
    for key, value in overall_metrics.items():
        results_file.write(f"{key}: {value}\n")

# Also print overall metrics to terminal in a formatted manner
print("\n=== Evaluation Metrics ===")
for key, value in overall_metrics.items():
    print(f"{key}: {value}")

# -------------------------
# Draw Summary Plot (shows all overall metrics)
# -------------------------
if draw_plot:
    summary_output = os.path.join(results_files_path, "Evaluation_Summary.png")
    draw_evaluation_summary(overall_metrics, summary_output, to_show=True)

# -------------------------
# Plot Ground-Truth and Predicted Objects Info
# -------------------------
if draw_plot:
    window_title = "Ground-Truth Info"
    plot_title = f"Ground-Truth\n({len(ground_truth_files_list)} files and {n_classes} classes)"
    x_label = "Number of objects per class"
    output_path = os.path.join(results_files_path, "Ground-Truth Info.png")
    draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title, x_label, output_path, False, 'forestgreen', '')

with open(os.path.join(results_files_path, "results.txt"), 'a') as results_file:
    results_file.write("\n# Number of ground-truth objects per class\n")
    for class_name in sorted(gt_counter_per_class):
        results_file.write(f"{class_name}: {gt_counter_per_class[class_name]}\n")

pred_counter_per_class = {}
for txt_file in predicted_files_list:
    lines_list = file_lines_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        if class_name in args.ignore:
            continue
        pred_counter_per_class[class_name] = pred_counter_per_class.get(class_name, 0) + 1

pred_classes = list(pred_counter_per_class.keys())

if draw_plot:
    window_title = "Predicted Objects Info"
    detected_classes = sum(1 for x in pred_counter_per_class.values() if x > 0)
    plot_title = f"Predicted Objects\n({len(predicted_files_list)} files and {detected_classes} detected classes)"
    x_label = "Number of objects per class"
    output_path = os.path.join(results_files_path, "Predicted Objects Info.png")
    draw_plot_func(pred_counter_per_class, len(pred_counter_per_class), window_title, plot_title, x_label, output_path, False, 'forestgreen', count_true_positives)

with open(os.path.join(results_files_path, "results.txt"), 'a') as results_file:
    results_file.write("\n# Number of predicted objects per class\n")
    for class_name in sorted(pred_classes):
        n_pred = pred_counter_per_class[class_name]
        tp_val = count_true_positives.get(class_name, 0)
        results_file.write(f"{class_name}: {n_pred} (tp:{tp_val}, fp:{n_pred - tp_val})\n")

if draw_plot:
    window_title = "mAP"
    plot_title = f"mAP@50 = {mAP50*100:.2f}%"
    x_label = "Average Precision"
    output_path = os.path.join(results_files_path, "mAP.png")
    draw_plot_func(ap_dictionary, n_classes, window_title, plot_title, x_label, output_path, True, 'royalblue', "")

# Remove temporary files directory
shutil.rmtree(tmp_files_path)
