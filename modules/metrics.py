import concurrent.futures
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def _compute(scorer, gts, res):
    try:
        return scorer.compute_score(gts, res, verbose=0)
    except TypeError:
        return scorer.compute_score(gts, res)


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
    """

    scorers = []
    try:
        scorers.append((Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]))
    except Exception as e:
        print("Warning: Failed to initialize BLEU scorer:", e)

    import shutil
    if shutil.which('java') is not None:
        try:
            scorers.append((Meteor(), "METEOR"))
        except (FileNotFoundError, Exception) as e:
            print("Warning: METEOR scorer initialization failed. Skipping METEOR metric. Error:", e)
    else:
        print("Warning: Java is not found in PATH. Skipping METEOR metric.")

    try:
        scorers.append((Rouge(), "ROUGE_L"))
    except Exception as e:
        print("Warning: Failed to initialize ROUGE scorer:", e)

    eval_res = {}
    for scorer, method in scorers:
        label = method if isinstance(method, str) else ", ".join(method)
        print("Computing metric: {} ...".format(label))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_compute, scorer, gts, res)
                score, scores = future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            print("Warning: Metric calculation for {} timed out after 15s. Skipping.".format(label))
            if hasattr(scorer, 'meteor_p') and scorer.meteor_p is not None:
                try:
                    scorer.meteor_p.kill()
                except Exception:
                    pass
            continue
        except Exception as e:
            print("Warning: Failed to compute score for {}: {}".format(label, e))
            continue

        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
