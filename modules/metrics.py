from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = []
    try:
        scorers.append((Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]))
    except Exception as e:
        print("Warning: Failed to initialize BLEU scorer:", e)

    try:
        scorers.append((Meteor(), "METEOR"))
    except (FileNotFoundError, Exception) as e:
        print("Warning: METEOR scorer initialization failed (possibly because Java is not installed or not in PATH). Skipping METEOR metric. Error:", e)

    try:
        scorers.append((Rouge(), "ROUGE_L"))
    except Exception as e:
        print("Warning: Failed to initialize ROUGE scorer:", e)

    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
        except Exception as e:
            if method != "METEOR":
                print("Warning: Failed to compute score for {}: {}".format(method, e))
            continue

        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
