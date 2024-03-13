import random, re
import numpy as np
import torch
from transformers import BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity


def predict_proba(texts, victim):
    # Adapt this function to work with your victim model
    # It should return a 2D numpy array with prediction probabilities
    return victim.get_prob(texts)


# right now we are supporting only the untargeted attack
def attack_text_bfs(
    text,
    explainer,
    victim,
    device,
    roberta_model,
    roberta_tokenizer,
    target_class=None,
    k=3,
    width=10,
    verbose=True,
    early_stop=10,
    imp="lime",
    lime_num_features=5000,
    lime_num_samples=5000,
    only_positive_importances=False,
    temperature=1,
    filter_beams_with_only_negative=False,
    remove_words=False,
    keep_original=False,
    add_semantic_pruning=False,
    bleurt_model=None,
    bleurt_tokenizer=None,
    semantic_pruning_ratio=0.3,
    positional=True
):
    outputs = []  # it will have (text, orig_proba)]

    orig_probs = victim.get_prob([text])
    orig_proba = orig_probs[0]
    initial_proba = orig_proba.max()
    target_class = orig_proba.argmax()
    if verbose:
        print(orig_proba)
        print("Using temperature: ", temperature)
    if verbose:
        print(
            f"Original prediction for class {target_class}: {orig_proba[target_class]}"
        )

    if imp == "lime":
        # explainer.bow => use TfIdf instead of BOW to capture positions => return positions in explanation
        if positional:
            if verbose:
                print("Using positions in the explanation.")
        exp = explainer.explain_instance(
            text,
            lambda texts: predict_proba(texts, victim),
            num_features=lime_num_features,
            num_samples=lime_num_samples,
            labels=[0, 1],
        )
        importances = exp.as_list(label=target_class, positions=positional)
    elif imp == "bert":
        importances = get_scores_new(
            text, victim, orig_probs, positional=positional
        )
    # elif imp == "random":
    #     importances = get_random_scores(text, sort=True, bert_tokenizer=bert_tokenizer)

    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    if only_positive_importances:
        importances = [
            item for item in importances if item[1] > 0
        ]  # keep only the words associated with the class whose probability we are minimizing
    if verbose:
        print("--------")
        print(importances)
        print("--------")

    beam = [(text, initial_proba)]
    for depth, (word, _) in enumerate(importances):
        new_beam = []
        for beam_text, beam_proba in beam:
            # if there is any punctuation at the beginning or end of the word_to_replace, strip them
            # including (, ), [, ], {, }, <, >, ", ', :, ;, ., ,, !, ?, -, _, =, +, *, /, \, |, &, %, $, #, @, ~, `
            if positional:
                parts = word.rsplit("_", 1)
                word_to_replace = parts[0]
                start_position = parts[1]
                word_to_replace = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word_to_replace) + "_" + start_position
            else:
                word_to_replace = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word)

            # alternatives = generate_alternatives(beam_text, word, k)
            alternatives = generate_alternatives_with_chunking(
                device=device,
                text=beam_text,
                word_to_replace=word_to_replace,
                width=width,
                temperature=temperature,
                roberta_model=roberta_model,
                tokenizer=roberta_tokenizer,
                remove_words=remove_words,
                keep_original=keep_original,
                positional=positional,
                chunk_size=512
            )
            for alt_text, proba_changed, replacement in alternatives:
                if proba_changed:
                    alt_proba = predict_proba([alt_text], victim)[0][target_class]
                else:
                    alt_proba = beam_proba
                proba_change = alt_proba - beam_proba
                if (not filter_beams_with_only_negative) or (filter_beams_with_only_negative and proba_change < 0):
                    new_beam.append((alt_text, alt_proba))
                    if verbose:
                        print(
                            f"Depth {depth+1}: '{word_to_replace}' -> '{replacement}', New Probability: {alt_proba} - {beam_proba} = {proba_change:.4f}"
                        )
                    # if successful, remove the beam back from the new_beam (which means we stop that branch) and add to the outputs
                    if new_beam[-1][1] < 0.5:
                        outputs.append(new_beam.pop(-1))
                        if verbose:
                            print("Added to the outputs: ", outputs[-1])
                        if len(outputs) >= early_stop:
                            return outputs, target_class, initial_proba

        if add_semantic_pruning:
            beam_outputs = [x[0] for x in new_beam]
            semantic_similarities = calculate_bleurt_score(bleurt_model, bleurt_tokenizer, 
                            [text]*len(beam_outputs), [beam[0] for beam in beam_outputs], device=device)
            # scale semantic similarities to [0, 1] linearly by x - max / min formula usign numpy functions
            # consider the edge cases, for example if max and min are the same
            max_semantic = np.max(semantic_similarities)
            min_semantic = np.min(semantic_similarities)
            if max_semantic != min_semantic:
                semantic_similarities = (semantic_similarities - min_semantic) / (max_semantic - min_semantic)
            
        else:
            semantic_similarities = [0]*len(new_beam)

        # sorted_beams = sorted(
        #     zip(new_beam, semantic_similarities), key=lambda x: np.log(x[0][1][target_class]*(1-semantic_pruning_ratio)-x[1]*semantic_pruning_ratio), reverse=False
        # )[:k]  # sorted low to high
        # beam = [x[0] for x in sorted_beams]

        sorted_beams = sorted(
            zip(new_beam, semantic_similarities), key=lambda x: x[0][1]*(1-semantic_pruning_ratio)-x[1]*semantic_pruning_ratio, reverse=False
        )[:k]  # sorted low to high
        beam = [x[0] for x in sorted_beams]

        # when one text flips it, return the whole beam
        # if beam[0][1][target_class] < 0.5:
        #   break

    return outputs, target_class, initial_proba

def generate_alternatives_with_chunking(
    device,
    text,
    word_to_replace,
    width,
    temperature,
    tokenizer,
    roberta_model,
    chunk_size=512,
    remove_words=False,
    keep_original=False,
    positional=False,
    mask_token="<mask>"
):
    chunk_size -= 2  # reserve space for the special tokens
    start_position = 0
    if positional:
        parts = word_to_replace.rsplit("_", 1)
        word_to_replace = parts[0]
        start_position = int(parts[1])

    word_pos = text.lower().find(word_to_replace.lower(), start_position)
    if word_pos == -1:
        word_pos = text.lower().find(word_to_replace.lower())
        if word_pos == -1:
            return [
                (text, False, "WORD TO REPLACE NOT FOUND")
            ]  # Return original if not found

    # Mask the word to replace
    masked_text = (
        text[:word_pos]
        + mask_token
        + text[word_pos + len(word_to_replace):]
    )

    input_ids = tokenizer.encode(masked_text, return_tensors="pt")[0]
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[0]

    start_index = max(0, mask_token_index - chunk_size // 2)
    end_index = min(len(input_ids), mask_token_index + (chunk_size + 1) // 2)
    chunk_ids = input_ids[start_index:end_index].unsqueeze(0)

    mask_token_index = torch.where(chunk_ids == tokenizer.mask_token_id)[0]
    with torch.no_grad():
        outputs = roberta_model(chunk_ids.to(device))
        predictions = torch.nn.functional.softmax(outputs.logits / temperature, dim=-1)
        # get probabilities for all predicted tokens
        
    top_k_indices = (
        torch.topk(predictions[0, mask_token_index, :], width, dim=1).indices[0].tolist()
    )

    alternatives = []
    predicted_tokens = [tokenizer.decode([token_id]) for token_id in top_k_indices]
    for predicted_token in predicted_tokens:
        predicted_token = predicted_token.strip()
        if predicted_token not in tokenizer.all_special_tokens and predicted_token != word_to_replace:
            predicted_sequence = re.sub(tokenizer.mask_token, predicted_token, masked_text)
            alternatives.append(
                (predicted_sequence, True, predicted_token)
            )
    if keep_original:
        alternatives.append((text, False, word_to_replace))
    if remove_words:
        predicted_sequence = (
            text[:word_pos].rstrip()
            + text[word_pos + len(word_to_replace):]
        )
        alternatives.append(
            (predicted_sequence, True, ["WORD REMOVED"])
        )

    return alternatives

def get_masked(words):
    len_text = max(len(words), 2)
    masked_words = []
    for i in range(len_text):
        masked_words.append(words[0:i] + ["[UNK]"] + words[i + 1 :])
    # list of words
    return masked_words

def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs):
    masked_words = get_masked(words)
    texts = [" ".join(words) for words in masked_words]  # list of text of masked words
    leave_1_probs = torch.Tensor(tgt_model.get_prob(texts))
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

    import_scores = (
        (
            orig_prob
            - leave_1_probs[:, orig_label]
            + (leave_1_probs_argmax != orig_label).float()
            * (
                leave_1_probs.max(dim=-1)[0]
                - torch.index_select(orig_probs, 0, leave_1_probs_argmax)
            )
        )
        .data.cpu()
        .numpy()
    )

    return import_scores

def get_scores_new(text, victim, orig_probs, positional=True, chunk_size=510):
    # first cut the sentence in a chunk_size piece
    # input_ids = roberta_tokenizer.encode(text, add_special_tokens=False)
    # input_ids = input_ids[:chunk_size]
    # text = roberta_tokenizer.decode(input_ids)
    words = text.replace("\n", " ").split(" ")
    if positional:
        starting_char_ids = []
        start_index = 0
        starting_char_ids.append(start_index)
        prev_word = words[0]
        for word in words[1:]:
            start_index = text.find(word, start_index+len(prev_word))
            starting_char_ids.append(start_index)
            prev_word = word
    orig_proba = orig_probs[0]
    initial_proba = orig_proba.max()
    target_class = orig_proba.argmax()

    important_scores = get_important_scores(
        words, victim, initial_proba, target_class, torch.Tensor(orig_probs)[0].squeeze()
    )
    word_score_pairs = []

    for top_index, score in enumerate(important_scores):
        tgt_word = words[top_index]
        if len(tgt_word) > 0:
            if positional:
                tgt_word = tgt_word + f"_{starting_char_ids[top_index]}"
            word_score_pairs.append((tgt_word, score))
    return word_score_pairs

# def get_random_scores(x, bert_tokenizer: BertTokenizerFast, sort=False):
#     # Tokenize the input text
#     words = tokenize_sentence(x)
#     # words, _, _ = tokenize(x, bert_tokenizer)

#     # Generate random scores for each word
#     word_score_pairs = [(word, random.random()) for word in words]

#     return word_score_pairs

def calculate_bleurt_score(bleurt_model, bleurt_tokenizer, references, predictions, device='cpu', batch_size=16):
    bleurt_model.eval()
    bleurt_model.to(device)
    
    # Tokenize inputs
    tokenized_inputs = bleurt_tokenizer(references, predictions, padding='longest', max_length=512,
                                        truncation=True, return_tensors='pt')
    tokenized_inputs = {key: tensor.to(device) for key, tensor in tokenized_inputs.items()}
    
    # Calculate BLEURT scores
    bleurt_scores = []
    for i in range(0, len(references), batch_size):
        batch_inputs = {key: tensor[i:i+batch_size] for key, tensor in tokenized_inputs.items()}
        with torch.no_grad():
            scores = bleurt_model(**batch_inputs).logits.flatten().cpu().numpy().tolist()
            bleurt_scores.extend(scores)
    
    return np.array(bleurt_scores)