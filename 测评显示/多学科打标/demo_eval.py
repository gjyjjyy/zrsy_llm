import torch
import pandas as pd
from datasets import load_dataset, ClassLabel


# match method where both side must match exactly
def strict_matching_tensors(list1, list2):
    # Ensure both lists have the same length
    assert len(list1) == len(list2), "Lists must have the same length"
    # Count the number of matching tensors
    matching_count = sum(torch.equal(tensor1, tensor2) for tensor1, tensor2 in zip(list1, list2))
    return matching_count


# generate match method where only masked fields must match exactly
def generate_masked_matching_fn(mask):
    def count_masked(preds, targets):
        assert len(preds) == len(targets), 'prediction array and target array must be same size'
        matching_count = 0

        for pred, target in zip(preds, targets):
            masked_pred = pred * mask
            masked_target = target * mask
            if torch.all(masked_pred == masked_target):
                matching_count += 1
        return matching_count
    return count_masked


# match method where prediction is counted as long as all target are covered
# use only for external models where confidence cannot be adjusted, for benchmark purpose
def subset_matching_tensors(predict, target):
    return sum(torch.all(p[t]) for p, t in zip(predict, target))


def target_align(target_values):
    targets = [torch.Tensor(tgt) for tgt in target_values]
    return [tensor.byte().bool() for tensor in targets]


def rate_model_output(model_name, file, match_fn, predict_field='prediction', target_field='target'):
    dataset = load_dataset('json', data_files=file)
    count = len(dataset['train'])
    data_frame = pd.DataFrame(dataset['train'])

    targets = target_align(data_frame[target_field].values)
    preds = target_align(data_frame[predict_field].values)
    matched = match_fn(preds, targets)

    print('%s Acc: %.4f\n' % (model_name, matched / count))


# read labels from a designated file using standard format
def get_labels(file):
    fr = open(file, 'r', encoding='utf-8')
    raw_labels = fr.readline()
    fr.close()
    # label return
    return [s.strip() for s in raw_labels.split(';')]


def labels_to_binary(cur_labels, full_labels, label_class):
    filtered_labels = [label for label in cur_labels if label in full_labels]
    label_indices = label_class.str2int(filtered_labels)
    labels_binary = [0] * len(full_labels)
    for idx in label_indices:
        labels_binary[idx] = 1
    return labels_binary


# given the total label space, generate a mask binary matrix listed under mask file.
def get_mask_binary(mask_file, label_file):
    masks = get_labels(mask_file)
    labels = get_labels(label_file)
    label_class = ClassLabel(names=labels)
    mask_binary = labels_to_binary(masks, labels, label_class)
    return torch.tensor(mask_binary)


if __name__ == '__main__':
    mask_tensor = get_mask_binary('./category_priority_mask.txt', './category_label.txt')
    priority_matching_tensors = generate_masked_matching_fn(mask_tensor)

    # compete model output
    rate_model_output('gpt 3.5 turbo', './gpt3_category.json', subset_matching_tensors)
    rate_model_output('gpt 4 turbo', './gpt4_category.json', subset_matching_tensors)
    rate_model_output('brainstorm baseline', './baseline/eval_result.json', strict_matching_tensors)
    rate_model_output('brainstorm ensemble', './ensemble/eval_result.json', strict_matching_tensors)
    rate_model_output('brainstorm optimized', './optimized/eval_result.json', priority_matching_tensors,
                      predict_field='biz_prediction')
