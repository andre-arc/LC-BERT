import torch

# Forward function for word classification
def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    (subword_batch, mask_batch, label_batch) = batch_data[:-1]
    token_type_batch = None
    
    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # print(subword_batch.size())
    # print(mask_batch.size())
    # print(label_batch.size())
    # exit()

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]

    
    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.argmax(logits, dim=-1).tolist()

    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        label = label_batch[i].item()  # Convert tensor to scalar

        list_hyp = i2w[hyps]  # Predicted class label
        list_label = i2w[label]  # Actual class label

        list_hyps.append(list_hyp)
        list_labels.append(list_label)
            
    return loss, list_hyps, list_labels

# Forward function for word classification
def modified_forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):

    # Unpack batch data
    (vector_batch, label_batch) = batch_data[:-1]
    token_type_batch = None
    
    # Prepare input & label
    vector_batch = torch.FloatTensor(vector_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        vector_batch = vector_batch.cuda()
        label_batch = label_batch.cuda()

    # print(subword_batch.size())
    # print(mask_batch.size())
    # print(label_batch.size())
    # exit()

    # Forward model
    outputs = model(vector_batch, labels=label_batch)
    loss, logits = outputs[:2]

    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.argmax(logits, dim=-1).tolist()


    for i in range(len(hyps_list)):
        hyps = hyps_list[i]
        label = label_batch[i].item()  # Convert tensor to scalar

        list_hyp = i2w[hyps]  # Predicted class label
        list_label = i2w[label]  # Actual class label

        list_hyps.append(list_hyp)
        list_labels.append(list_label)
            
    return loss, list_hyps, list_labels