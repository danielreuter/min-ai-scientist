import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from logging import getLogger
from textwrap import dedent

from aider.coders import Coder

from scientist.config import ai
from scientist.models import Idea

logger = getLogger(__name__)

MAX_ITERS = 4
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 1500


@dataclass(frozen=True)
class ExperimentResult:
    return_code: int
    message: str


@ai.task()
async def perform_experiments(
    idea: Idea,
    baseline_results: dict,
    coder: Coder,
    max_runs: int = MAX_RUNS,
    max_iters: int = MAX_ITERS,
) -> bool:
    run_number = 1
    current_iter = 0
    next_prompt = dedent(f"""\
        Your goal is to implement the following idea: {idea.title}.
        The proposed experiment is as follows: {idea.experiment}.
        You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

        First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

        Note that we already provide the vanilla baseline results, so you do not need to re-run it.

        For reference, the baseline results are as follows:
        {json.dumps(baseline_results, indent=4)}

        After you complete each change, we will run the command 'python experiment.py --out_dir=experiment_i' where i is the run number and evaluate the results.
        YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
        You can then implement the next thing on your list.
    """)

    while run_number <= max_runs:
        if current_iter >= max_iters:
            print("Max iterations reached")
            break

        coder_out = coder.run(next_prompt)
        print(coder_out)

        if "ALL_COMPLETED" in coder_out:
            break

        result = await run_experiment(run_number)
        if result.return_code == 0:
            run_number += 1
            current_iter = 0
            next_prompt = result.message
        else:
            current_iter += 1
            next_prompt = result.message

    if current_iter >= max_iters:
        print("Not all experiments completed.")
        return False

    # handle plotting
    current_iter = 0
    next_prompt = dedent("""
        Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 
        In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.
        Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.
        We will be running the command `python plot.py` to generate the plots.
    """)

    while True:
        _ = coder.run(next_prompt)
        result = await run_plotting()
        current_iter += 1
        if result.return_code == 0 or current_iter >= max_iters:
            break
        next_prompt = result.message

    # handle notes
    coder.run(
        dedent("""
            Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.
            Somebody else will be using `notes.txt` to write a report on this in the future.
        """)
    )

    return True


@ai.task()
async def run_experiment(
    run_num: int,
    timeout: int = 7200,
) -> ExperimentResult:
    cwd = ai.run_dir()
    exp_dir = cwd / f"experiment_{run_num}"
    exp_dir.mkdir(exist_ok=True, parents=True)

    # copy code so we can see it
    shutil.copy(
        cwd / "experiment.py",
        exp_dir / "experiment.py",
    )

    # launch command
    command = [
        sys.executable,
        "experiment.py",
        f"--out_dir=experiment_{run_num}",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            logger.warning(f"Run {run_num} failed with return code {result.returncode}")

            shutil.rmtree(exp_dir)

            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]

            message = f"Run failed with the following error:\n{stderr_output}"
        else:
            with open(exp_dir / "final_info.json", "r") as f:
                results = json.load(f)

            results_summary = {k: v["means"] for k, v in results.items()}

            message = dedent(f"""\
                Run {run_num} completed. Here are the results:

                {json.dumps(results_summary, indent=4)}
                
                Decide if you need to re-plan your experiments given the result (you often will not need to).

                Someone else will be using notes.txt to perform a writeup on this in the future. Please include all relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

                Then, implement the next thing on your list. We will then run the command 'python experiment.py --out_dir=run_{run_num + 1}'. YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS. If you are finished with experiments, respond with 'ALL_COMPLETED'.
            """)

        return ExperimentResult(return_code=result.returncode, message=message)

    except subprocess.TimeoutExpired:
        logger.warning(f"Run {run_num} timed out after {timeout} seconds")
        shutil.rmtree(exp_dir)
        message = f"Run timed out after {timeout} seconds"
        return ExperimentResult(return_code=1, message=message)


@ai.task()
async def run_plotting(
    timeout: int = 600,
) -> ExperimentResult:
    cwd = ai.run_dir()
    plot_dir = cwd / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # copy plot.py so we can see it
    shutil.copy(
        cwd / "plot.py",
        plot_dir / "plot.py",
    )

    # launch command
    command = [
        sys.executable,
        "plot.py",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            logger.warning("Plotting failed with return code {result.returncode}")

            shutil.rmtree(plot_dir)

            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]

            message = f"Plotting failed with the following error:\n{stderr_output}"
        else:
            message = "Plotting completed successfully."

        return ExperimentResult(return_code=result.returncode, message=message)

    except subprocess.TimeoutExpired:
        logger.warning(f"Plotting timed out after {timeout} seconds")
        shutil.rmtree(plot_dir)
        message = f"Plotting timed out after {timeout} seconds"
        return ExperimentResult(return_code=1, message=message)
