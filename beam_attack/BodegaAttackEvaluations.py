import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

import OpenAttack
from OpenAttack.tags import TAG_Classification, Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer
from OpenAttack.utils import visualizer, result_visualizer
from tqdm import tqdm

from metrics.BODEGAScore import BODEGAScore
from utils.no_ssl_verify import no_ssl_verify


class BodegaAttackEval(OpenAttack.AttackEval):
    """
    wrapper for OpenAttack.AttackEval to produce a submission.tsv file for shared task evaluation

    To perform evaluation, we use a new method: eval_and_save_tsv() rather than the usual AttackEval.eval()
    submission.tsv file consists of 4 columns for each sample in attack set: succeeded, num_queries, original_text and modified text (newlines are escaped)

    """

    def eval_and_save_tsv(
        self,
        dataset: Iterable[Dict[str, Any]],
        total_len: Optional[int] = None,
        visualize: bool = False,
        progress_bar: bool = False,
        num_workers: int = 0,
        chunk_size: Optional[int] = None,
        tsv_file_path: Optional[os.PathLike] = None,
    ):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_workers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.

            tsv_file_path: path to save submission tsv

        Returns:
            A dict of attack evaluation summaries.

        """

        if hasattr(dataset, "__len__"):
            total_len = len(dataset)

        def tqdm_writer(x):
            return tqdm.write(x, end="")

        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        # list for tsv
        x_orig_list = []
        x_adv_list = []
        num_queries_list = []
        succeed_list = []

        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])

            if TAG_Classification in self.victim.TAGS:
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        y_adv = probs[1]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])
                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                info["Succeed"] = res["success"]
                if visualize:
                    if progress_bar:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.tokenizer)
                    else:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.tokenizer)

                # list for tsv
                succeed_list.append(res["success"])
                num_queries_list.append(res["metrics"]["Victim Model Queries"])
                x_orig_list.append(x_orig)

                if res["success"]:
                    x_adv_list.append(x_adv)
                else:
                    x_adv_list.append("ATTACK_UNSUCCESSFUL")

            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]

        if visualize:
            result_visualizer(summary, sys.stdout.write)

        # saving tsv
        if tsv_file_path is not None:
            with open(tsv_file_path, "w") as f:
                f.write("succeeded" + "\t" + "num_queries" + "\t" + "original_text" + "\t" + "modified_text" + "\t" + "\n")  # header
                for success, num_queries, x_orig, x_adv in zip(succeed_list, num_queries_list, x_orig_list, x_adv_list):
                    escaped_x_orig = x_orig.replace("\n", "\\n")  # escaping newlines
                    escaped_x_adv = x_adv.replace("\n", "\\n")
                    f.write(str(success) + "\t" + str(num_queries) + "\t" + escaped_x_orig + "\t" + escaped_x_adv + "\t" + "\n")

        return summary


"""
This example code shows how to design a customized attack model (that shuffles the tokens in the original sentence).
Taken from https://github.com/thunlp/OpenAttack/blob/master/examples/custom_attacker.py
"""


class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
    @property
    def TAGS(self):
        # returns tags can help OpenAttack to check your parameters automatically
        return {self.lang_tag, Tag("get_pred", "victim")}

    def __init__(self, tokenizer=None):
        if tokenizer is None:
            with no_ssl_verify():
                tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer
        self.lang_tag = OpenAttack.utils.get_language([self.tokenizer])
        # We add parameter ``processor`` to specify the :py:class:`.TextProcessor` which is used for tokenization and detokenization.
        # By default, :py:class:`.DefaultTextProcessor` is used.

    def attack(self, victim, input_, goal):
        # Generate a potential adversarial example
        x_new = self.tokenizer.detokenize(self.swap(self.tokenizer.tokenize(input_, pos_tagging=False)))

        # Get the predictions of victim classifier
        y_new = victim.get_pred([x_new])

        # Check for attack goal
        if goal.check(x_new, y_new):
            return x_new
        # Failed
        return None

    def swap(self, sentence):
        # Shuffle tokens to generate a potential adversarial example
        random.shuffle(sentence)

        # Return the potential adversarial example
        return sentence


def setup_bodega_scorer_and_run_attacks(attacker, dataset, device, out_dir, victim, results_path: Path, task, victim_name: str="BERT"):
    raw_path = out_dir / ("raw_" + task + f"_targeted_{victim_name}.tsv")
    submission_path = out_dir / ("submission_" + task + f"_targeted_{victim_name}.tsv")
    with no_ssl_verify():
        scorer = BODEGAScore(device, task, align_sentences=True, semantic_scorer="BLEURT", raw_path=raw_path)
        attack_eval = BodegaAttackEval(attacker, victim, language="english", metrics=[scorer])  # , OpenAttack.metric.EditDistance()
    start = time.time()
    visualize_adv_examples = True  # prints adversarial samples as they are generated, showing the difference between original
    summary = attack_eval.eval_and_save_tsv(dataset, visualize=visualize_adv_examples, progress_bar=True, tsv_file_path=submission_path)
    end = time.time()
    attack_time = end - start

    with open(results_path, "a") as f:
        f.write(f"\nTimestamp: {time.time()} \n")
        f.write("Subset size: " + str(len(dataset)) + "\n")
        f.write("Queries per example: " + str(summary["Avg. Victim Model Queries"]) + "\n")
        f.write("Total attack time: " + str(end - start) + "\n")
        f.write("Time per example: " + str((end - start) / len(dataset)) + "\n")

    print("-")
    print("Submission file saved to", submission_path)

    return scorer


def calculate_metrics(scorer, results_path: Path):
    start = time.time()
    score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
    end = time.time()
    evaluate_time = end - start

    with open(results_path, "a") as f:
        f.write("Success score: " + str(score_success) + "\n")
        f.write("Semantic score: " + str(score_semantic) + "\n")
        f.write("Character score: " + str(score_character) + "\n")
        f.write("BODEGA score: " + str(score_BODEGA) + "\n")
        f.write("Total evaluation time: " + str(evaluate_time) + "\n")

    print("Bodega metrics saved to", results_path)
