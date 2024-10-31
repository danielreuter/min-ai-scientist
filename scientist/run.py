import json
import multiprocessing
import os
import shutil
import sys
import time
from logging import getLogger

import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from anyio import Path

from scientist.config import ai
from scientist.datasets import ExperimentDataset
from scientist.generate_ideas import generate_novel_ideas
from scientist.models import Experiment, ExperimentName, Idea
from scientist.perform_experiments import perform_experiments
from scientist.perform_review import load_paper, perform_review

logger = getLogger(__name__)

PROJECT_ROOT = Path(os.getcwd()).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def read_initial_code(experiment_name: ExperimentName) -> str:
    with open(f"{TEMPLATES_DIR}/{experiment_name.value}/experiment.py", "r") as f:  # Add .value
        return f.read()


async def run_scientist(
    experiment_name: ExperimentName = ExperimentName.GROKKING,
    model: str = "gpt-4o-mini",
    writeup: str = "latex",
    parallel: int = 0,
    improvement: bool = False,
    gpus: str = "0",  # Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.
    num_ideas: int = 50,
) -> None:
    available_gpus = get_available_gpus(gpus.split(","))
    if parallel > len(available_gpus):
        logger.warning(
            f"Requested {parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        parallel = len(available_gpus)
    logger.info(f"Using GPUs: {available_gpus}")

    dataset = ExperimentDataset(experiment_name.value)
    experiment = dataset[0].experiment
    assert isinstance(experiment, Experiment)

    novel_ideas = await generate_novel_ideas(
        experiment=experiment,
        max_num_iterations=2,
        max_num_generations=1,
        num_refinements=2,
    )

    if parallel > 0:
        logger.info(f"Running {parallel} parallel processes")

        queue: multiprocessing.Queue[Idea | None] = multiprocessing.Queue()
        for idea in novel_ideas[:num_ideas]:
            queue.put(idea)

        processes: list[multiprocessing.Process] = []
        for i in range(parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(queue, writeup, improvement, gpu_id),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(parallel):
            queue.put(None)

        for p in processes:
            p.join()

        logger.info("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            logger.info(f"Processing idea: {idea.name}")

            try:
                success = do_idea(
                    experiment=experiment,
                    idea=idea,
                    model=model,
                    writeup=writeup,
                    improvement=improvement,
                )
                logger.info(f"Completed idea: {idea.name}, Success: {success}")
            except Exception as e:
                logger.error(f"Failed to evaluate idea {idea.name}: {str(e)}")

    logger.info("All ideas evaluated.")


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def worker(
    experiment: Experiment,
    queue: multiprocessing.Queue[Idea],
    writeup: str,
    improvement: bool,
    gpu_id: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            experiment=experiment,
            idea=idea,
            model="gpt-4o-mini",
            writeup=writeup,
            improvement=improvement,
            log_file=True,
        )
        logger.info(f"Completed idea: {idea.name}, Success: {success}")
    logger.info(f"Worker {gpu_id} finished.")


def do_idea(
    experiment: Experiment,
    idea: Idea,
    model: str,
    writeup: str,
    improvement: bool,
    log_file: bool = False,
):
    base_dir = TEMPLATES_DIR / experiment.name.value
    idea_dir = ai.run_dir() / idea.name
    exp_file = idea_dir / "experiment.py"
    vis_file = idea_dir / "plot.py"
    notes = idea_dir / "notes.txt"

    assert not idea_dir.exists(), f"Folder {idea_dir} already exists."
    shutil.copytree(base_dir, idea_dir, dirs_exist_ok=True)
    with open(base_dir / "run_0" / "final_info.json", "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}

    with open(notes, "w") as f:
        f.write(f"# Title: {idea.title}\n")
        f.write(f"# Experiment description: {idea.experiment}\n")
        f.write("## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write("Description: Baseline results.\n")

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log = open(idea_dir / "log.txt", "a")
        sys.stdout = log
        sys.stderr = log
    try:
        logger.info(f"*Starting idea: {idea.name}*")
        io = InputOutput(yes=True, chat_history_file=f"{idea_dir}/{idea.name}_aider.txt")
        coder = Coder.create(
            main_model=Model(model),
            fnames=[exp_file, vis_file, notes],
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        logger.info("*Starting Experiments*")
        try:
            success = perform_experiments(idea, baseline_results, coder)
        except Exception as e:
            logger.info(f"Error during experiments: {e}")
            logger.info(f"Experiments failed for idea {idea.name}")
            return False

        if not success:
            logger.info(f"Experiments failed for idea {idea.name}")
            return False

        logger.info("*Starting Writeup*")
        if writeup == "latex":
            writeup_file = idea_dir / "latex" / "template.tex"
            coder = Coder.create(
                main_model=Model(model),
                fnames=[exp_file, writeup_file, notes],
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                # perform_writeup(idea, folder_name, coder, client, client_model)
                pass
            except Exception as e:
                logger.info(f"Failed to perform writeup: {e}")
                return False
            logger.info("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        logger.info("*Starting Review*")
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{idea_dir}/{idea.name}.pdf")
                review = perform_review(paper_text, model=model)
                # Store the review in separate review.txt file
                with open(idea_dir / "review.txt", "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                logger.info(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            logger.info("*Starting Improvement*")
            try:
                # perform_improvement(review, coder)
                # generate_latex(coder, idea_dir, f"{idea_dir}/{idea.name}_improved.pdf")
                paper_text = load_paper(f"{idea_dir}/{idea.name}_improved.pdf")
                review = perform_review(paper_text, model=model)
                # Store the review in separate review.txt file
                with open(idea_dir / "review_improved.txt", "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                logger.info(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        logger.info(f"Failed to evaluate idea {idea.name}: {str(e)}")
        return False
    finally:
        logger.info("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()
