import torch
from utils import *
import math
from model.SARI import calculate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
sf = SmoothingFunction()


def sample(complex_sentences, simple_sentences, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob):
    count = 0
    sari_scorel = 0
    keepl = 0
    deletel = 0
    addl = 0
    b_scorel = 0
    p_scorel = 0
    fkgl_scorel = 0
    fre_scorel = 0
    stats = {'ls': 0, 'dl': 0, 'las': 0, 'rl': 0}
    lm_forward.load_state_dict(torch.load(config['lm_name'] + '.pt',
                                          map_location='cuda:%s' % config['gpu']))
    if config['double_LM']:
        lm_backward.load_state_dict(torch.load(
            'structured_lm_backward_300_150_0_4.pt'))
    lm_forward.eval()
    lm_backward.eval()

    # resume system
    offset_sample = 0
    if os.path.isfile(config['file_name']):
        simplfied = open(config['file_name'], encoding='utf-8').read()
        simplfied = simplfied.split('\n')
        simplfied = len(
            [line for i, line in enumerate(simplfied) if i % 8 == 2])
        # if output file exist and not finished, continue simplification
        # else create new file
        if simplfied < len(complex_sentences):
            offset_sample = simplfied
        else:
            open(config['file_name'], "w", encoding='utf-8')

    for i in range(offset_sample, len(complex_sentences)):
        sl, kl, dl, al, bl, pl, fkl, frl = mcmc(complex_sentences[i], simple_sentences[i], input_lang, tag_lang,
                                                dep_lang, lm_forward, lm_backward, embedding_weights, idf,
                                                unigram_prob, stats)
        # print('\n')
        # print("Average sentence level SARI till now for sentences")
        # sari_scorel += sl
        # keepl += kl
        # deletel += dl
        # addl += al
        # p_scorel += pl
        # print(sari_scorel / (count + 1))
        # print(keepl / (count + 1))
        # print(deletel / (count + 1))
        # print(addl / (count + 1))
        # print("Average sentence level BLEU till now for sentences")
        # b_scorel += bl
        # print(b_scorel / (count + 1))
        # print("Average perplexity of sentences")
        # print(p_scorel / (count + 1))
        # fkgl_scorel += fkl
        # fre_scorel += frl
        # print('Average sentence level FKGL and FRE till now for sentences')
        # print(fkgl_scorel / (count + 1))
        # print(fre_scorel / (count + 1))
        # print('\n')
        print(i + 1)

        with open(config['file_name'], "a", encoding='utf-8') as file:
            file.write("Average Sentence Level Perplexity, Bleu, SARI \n")
            file.write(str(p_scorel / (count + 1)) + " " + str(b_scorel /
                                                               (count + 1)) + " " + str(sari_scorel / (count + 1)) + "\n\n")
        count += 1

    print(stats)


def mcmc(input_sent, reference, input_lang, tag_lang, dep_lang, lm_forward, lm_backward, embedding_weights, idf, unigram_prob, stats):
    print(stats)
    reference = reference.lower()
    given_complex_sentence = input_sent.lower()
    orig_sent = input_sent

    original_emb = nli_model.encode(orig_sent)
    original_mag = np.linalg.norm(original_emb)

    beam = {}
    entities = get_entities(input_sent)
    perplexity = -10000
    perpf = -10000
    synonym_dict = {}
    sent_list = set()
    spl = input_sent.lower().split(' ')
    # the for loop below is just in case if the edit operations go for a very long time
    # in almost all the cases this will not be required
    for iter in range(2 * len(spl)):

        synonym_dict = {}
        print(input_sent)

        '''if len(input_sent.split(' ')) <= 3:
            print('sentence length already at min, so cannot do deletion')
            # it could be debatable where do we get 85 from, is it from aligned text
            continue'''
        doc = nlp(input_sent)
        elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for
                                                                                                                                        tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
        prob_old = calculate_score(lm_forward, elmo_tensor, input_sent_tensor, tag_tensor, dep_tensor,
                                   input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob)
        if config['double_LM']:
            elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
                                                                                                                                                                               tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
            prob_old += calculate_score(lm_backward, elmo_tensor_b, input_sent_tensor_b, tag_tensor_b, dep_tensor_b,
                                        input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob)
            prob_old /= 2.0
        # for the first time step the beam size is 1, just the original complex sentence
        if iter == 0:
            beam[input_sent] = [prob_old, 'original']
        print('Getting candidates for iteration: ', iter)

        new_beam = {}
        # intialize the candidate beam
        for key in beam:
            # get candidate sentence through different edit operations
            candidates = get_subphrase_mod(
                key, sent_list, input_lang, idf, spl, entities, synonym_dict)
            print('Scoring {} candidates...'.format(len(candidates)))

            if len(candidates) > 0:
                cosine_similarities = sentence_cosine_similarity(
                    [list(candid.keys())[0] for candid in candidates], original_emb, original_mag)
                if len(orig_sent) > 275:
                    print(len(orig_sent), 'characters, taking a 6 second break.')
                    time.sleep(6)
                # print(len([1 for cos in cosine_similarities if cos <
                #            config['cos_similarity_threshold']]), 'will be eliminated')

            for i in range(len(candidates)):
                sent = list(candidates[i].keys())[0]
                operation = candidates[i][sent]
                doc = nlp(list(candidates[i].keys())[0])
                elmo_tensor, candidate_tensor, candidate_tag_tensor, candidate_dep_tensor = tokenize_sent_special(sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for
                                                                                                                                                             tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
                # calculate score for each candidate sentence using the scoring function
                p = calculate_score(lm_forward, elmo_tensor, candidate_tensor, candidate_tag_tensor,
                                    candidate_dep_tensor, input_lang, sent, orig_sent, embedding_weights, idf, unigram_prob, cosine_similarities[i])
                if config['double_LM']:
                    elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b = tokenize_sent_special(reverse_sent(sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
                                                                                                                                                                                                    tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
                    p += calculate_score(lm_backward, elmo_tensor_b, candidate_tensor_b, candidate_tag_tensor_b, candidate_dep_tensor_b,
                                         input_lang, reverse_sent(sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob, cosine_similarities[i])
                    p /= 2.0

                # if the candidate sentence is able to increase the score by a threshold value, add it to the beam
                if p > prob_old * config['threshold'][operation]:
                    new_beam[sent] = [p, operation]
                    # record the edit operation by which the candidate sentence was created
                    stats[operation] += 1
                else:
                    # if the threshold is not crossed, add it to a list so that the sentence is not considered in the future
                    sent_list.add(sent)
        if new_beam == {}:
            # if there are no candidate sentences, exit
            break

        new_beam_sorted_list = sorted(
            new_beam.items(), key=lambda x: x[1])[-config['beam_size']:]
        # sort the created beam on the basis of scores from the scoring function
        new_beam = {}
        # top k top scoring sentences selected. In our experiments the beam size is 1
        # copying the new_beam_sorted_list into new_beam
        for key in new_beam_sorted_list:
            new_beam[key[0]] = key[1]

        # we'll get top beam_size (or <= beam size) candidates

        # get the top scoring sentence. This will act as the source sentence for the next iteartion
        maxvalue_sent = max(new_beam.items(), key=lambda x: x[1])[0]
        perpf = new_beam[maxvalue_sent][0]
        input_sent = maxvalue_sent

        sent_list.add(maxvalue_sent)

        # for the next iteration
        beam = new_beam.copy()

    input_sent = input_sent.lower()

    # print("Input complex sentence")
    # print(given_complex_sentence)
    # print("Reference sentence")
    # print(reference)
    # print("Simplified sentence")
    # print(input_sent)

    # scorel, keepl, deletel, addl = calculate(
    #     given_complex_sentence, input_sent.lower(), [reference])

    # bleul = sentence_bleu([convert_to_blue(reference)], convert_to_blue(
    #     input_sent.lower()), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)

    # if (perpf == -10000):
    #     doc = nlp(input_sent)
    #     elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor = tokenize_sent_special(input_sent.lower(), input_lang, convert_to_sent([(tok.tag_).upper() for
    #                                                                                                                                               tok in doc]), tag_lang, convert_to_sent([(tok.dep_).upper() for tok in doc]), dep_lang)
    #     perpf = calculate_score(lm_forward, elmo_tensor, best_input_tensor, best_tag_tensor, best_dep_tensor,
    #                             input_lang, input_sent, orig_sent, embedding_weights, idf, unigram_prob)
    #     if config['double_LM']:
    #         elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b = tokenize_sent_special(reverse_sent(input_sent.lower()), input_lang, reverse_sent(convert_to_sent([(tok.tag_).upper() for
    #                                                                                                                                                                                      tok in doc])), tag_lang, reverse_sent(convert_to_sent([(tok.dep_).upper() for tok in doc])), dep_lang)
    #         perpf += calculate_score(lm_backward, elmo_tensor_b, best_input_tensor_b, best_tag_tensor_b, best_dep_tensor_b,
    #                                  input_lang, reverse_sent(input_sent), reverse_sent(orig_sent), embedding_weights, idf, unigram_prob)

    # fkgl_scorel = sentence_fkgl(input_sent)
    # fre_scorel = sentence_fre(input_sent)

    scorel = 0
    keepl = 0
    deletel = 0
    addl = 0
    bleul = 0
    perpf = 0
    fkgl_scorel = 0
    fre_scorel = 0

    with open(config['file_name'], "a", encoding='utf-8') as file:
        file.write(given_complex_sentence + "\n")
        file.write(reference + "\n")
        file.write(input_sent.lower() + "\n")
        file.write(str(perpf) + " " + str(bleul) + " " + str(scorel) + " " + str(keepl) + " " +
                   str(deletel) + " " + str(addl) + " " + str(fkgl_scorel) + " " + str(fre_scorel) + "\n")
        file.write("\n")
    return scorel, keepl, deletel, addl, bleul, perpf, fkgl_scorel, fre_scorel
