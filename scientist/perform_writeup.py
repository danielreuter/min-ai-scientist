# import json
# import os
# import os.path as osp
# import re
# import shutil
# import subprocess
# from logging import getLogger
# from typing import Dict

# from scientist.config import ai
# from scientist.generate_ideas import search_for_papers
# from scientist.models import CodeModification, Experiment, Idea, Think

# logger = getLogger(__name__)

# MAX_LATEX_ERROR_CORRECTIONS = 5
# MAX_LATEX_COMPILE_TIMEOUT = 30

# per_section_tips = {
#     "Abstract": """
# - **TL;DR of the paper**
# - What are we trying to do and why is it relevant?
# - Why is this hard?
# - How do we solve it (i.e., our contribution!)
# - How do we verify that we solved it (e.g., experiments and results)

# Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
# """,
#     "Introduction": """
# - **Longer version of the Abstract**, i.e., of the entire paper
# - What are we trying to do and why is it relevant?
# - Why is this hard?
# - How do we solve it (i.e., our contribution!)
# - How do we verify that we solved it (e.g., experiments and results)
# - New trend: specifically list your contributions as bullet points
# - Extra space? Future work!
# """,
#     "Related Work": """
# - **Academic siblings** of our work, i.e., alternative attempts in literature at trying to solve the same problem.
# - Goal is to **compare and contrast**—how does their approach differ in either assumptions or method?
# - Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
# """,
#     "Background": """
# - **Academic ancestors** of our work, i.e., all concepts and prior work that are required for understanding our method.
# - Usually includes a subsection, **Problem Setting**, which formally introduces the problem setting and notation for our method.
# - Highlights any specific assumptions that are made that are unusual.
# - Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate section.
# """,
#     "Method": """
# - **What we do and why we do it.**
# - Described using the general formalism introduced in the Problem Setting and building on top of the concepts introduced in Background.
# """,
#     "Experimental Setup": """
# - **How do we test that our stuff works?**
# - Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method.
# - Do not imagine unknown hardware details.
# - Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.
# """,
#     "Results": """
# - **Shows the results** of running Method on our problem described in Experimental Setup.
# - Includes statements on hyperparameters and other potential issues of fairness.
# - Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
# - Compares to baselines and includes statistics and confidence intervals.
# - Includes ablation studies to show that specific parts of the method are relevant.
# - Discusses limitations of the method.
# - Make sure to include all the results from the experiments, and include all relevant figures.
# """,
#     "Conclusion": """
# - **Brief recap of the entire paper.**
# - To keep going with the analogy, you can think of future work as (potential) academic offspring.
# """,
# }

# error_list = """- Unenclosed math symbols
# - Only reference figures that exist in our directory
# - LaTeX syntax errors
# - Numerical results that do not come from explicit experiments and logs
# - Repeatedly defined figure labels
# - References to papers that are not in the .bib file, DO NOT ADD ANY NEW CITATIONS!
# - Unnecessary verbosity or repetition, unclear text
# - Results or insights in the `notes.txt` that have not yet been included
# - Any relevant figures that have not yet been included in the text
# - Closing any \\begin{figure} with a \\end{figure} and \\begin{table} with a \\end{table}, etc.
# - Duplicate headers, e.g., duplicated \\section{Introduction} or \\end{document}
# - Unescaped symbols, e.g., shakespeare_char should be shakespeare\\_char in text
# - Incorrect closing of environments, e.g., </end{figure}> instead of \\end{figure}
# """


# @ai.task()
# async def perform_writeup(
#     idea: Idea,
#     experiment: Experiment,
#     folder_name: str,
#     num_cite_rounds: int = 20,
#     model: str = "openai:gpt-4-turbo",
# ) -> None:
#     messages = [
#         ai.message.system(
#             "You are assisting in writing a scientific paper based on the provided idea and experiment."
#         ),
#     ]

#     # Abstract prompt
#     messages.append(
#         ai.message.user(f"""\
#             We've provided the `latex/template.tex` file to the project. We will be filling it in section by section.

#             First, please fill in the "Title" and "Abstract" sections of the writeup.

#             Some tips are provided below:
#             {per_section_tips["Abstract"]}

#             Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

#             Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
#             """)
#     )

#     code_modification = await ai.generate_object(
#         model=model,
#         messages=messages,
#         type=CodeModification,
#     )

#     apply_code_modifications(folder_name, code_modification.files)
#     messages.append(ai.message.assistant(code_modification.message))

#     # Perform refinement for Abstract
#     messages.append(
#         ai.message.user(
#             f"""Great job! Now criticize and refine only the Abstract that you just wrote.
# Make this complete in this pass; do not leave any placeholders.

# Pay particular attention to fixing any errors such as:
# {error_list}
# """
#         )
#     )

#     code_modification = await ai.generate_object(
#         model=model,
#         messages=messages,
#         type=CodeModification,
#     )

#     apply_code_modifications(folder_name, code_modification.files)
#     messages.append(ai.message.assistant(code_modification.message))

#     # Handle other sections
#     for section in [
#         "Introduction",
#         "Background",
#         "Method",
#         "Experimental Setup",
#         "Results",
#         "Conclusion",
#     ]:
#         # Section prompt
#         messages.append(
#             ai.message.user(
#                 f"""Please fill in the {section} of the writeup. Some tips are provided below:
# {per_section_tips[section]}

# Be sure to use \\cite or \\citet where relevant, referring to the works provided in the file.
# Do not cite anything that is not already in `references.bib`. Do not add any new entries to this.

# Keep the experimental results (figures and tables) only in the Results section, and make sure that any captions are filled in.
# In this pass, do not reference anything in later sections of the paper.

# Before every paragraph, please include a brief description of what you plan to write in that paragraph in a comment.

# Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
# """
#             )
#         )

#         code_modification = await ai.generate_object(
#             model=model,
#             messages=messages,
#             type=CodeModification,
#         )

#         apply_code_modifications(folder_name, code_modification.files)
#         messages.append(ai.message.assistant(code_modification.message))

#         # Perform refinement for the section
#         messages.append(
#             ai.message.user(
#                 f"""Great job! Now criticize and refine only the {section} that you just wrote.
# Make this complete in this pass; do not leave any placeholders.

# Pay particular attention to fixing any errors such as:
# {error_list}
# """
#             )
#         )

#         code_modification = await ai.generate_object(
#             model=model,
#             messages=messages,
#             type=CodeModification,
#         )

#         apply_code_modifications(folder_name, code_modification.files)
#         messages.append(ai.message.assistant(code_modification.message))

#     # Handle Related Work separately
#     section = "Related Work"
#     messages.append(
#         ai.message.user(
#             f"""Please fill in the {section} of the writeup. Some tips are provided below:

# {per_section_tips[section]}

# For this section, very briefly sketch out the structure of the section, and clearly indicate what papers you intend to include.
# Do this all in LaTeX comments using %.
# The related work should be concise; only plan to discuss the most relevant work.
# Do not modify `references.bib` to add any new citations; this will be filled in at a later stage.

# Be sure to first name the file and use *SEARCH/REPLACE* blocks to perform these edits.
# """
#         )
#     )

#     code_modification = await ai.generate_object(
#         model=model,
#         messages=messages,
#         type=CodeModification,
#     )

#     apply_code_modifications(folder_name, code_modification.files)
#     messages.append(ai.message.assistant(code_modification.message))

#     # Perform citation rounds
#     draft_file = osp.join(folder_name, "latex", "template.tex")
#     with open(draft_file, "r") as f:
#         draft = f.read()

#     draft = await add_citations(
#         draft=draft,
#         total_rounds=num_cite_rounds,
#         model=model,
#     )

#     with open(draft_file, "w") as f:
#         f.write(draft)

#     # Perform refinement of Related Work section
#     messages.append(
#         ai.message.user(
#             f"""Great job! Now criticize and refine only the Related Work that you just wrote.
# Make this complete in this pass; do not leave any placeholders.

# Pay particular attention to fixing any errors such as:
# {error_list}
# """
#         )
#     )

#     code_modification = await ai.generate_object(
#         model=model,
#         messages=messages,
#         type=CodeModification,
#     )

#     apply_code_modifications(folder_name, code_modification.files)
#     messages.append(ai.message.assistant(code_modification.message))

#     # Second refinement loop
#     messages.append(
#         ai.message.user(
#             """Great job! Now that there is a complete draft of the entire paper, let's refine each section again.
# First, re-think the Title if necessary. Keep this concise and descriptive of the paper's concept, but try to be creative with it."""
#         )
#     )

#     for section in [
#         "Abstract",
#         "Related Work",
#         "Introduction",
#         "Background",
#         "Method",
#         "Experimental Setup",
#         "Results",
#         "Conclusion",
#     ]:
#         messages.append(
#             ai.message.user(
#                 f"""Criticize and refine the {section} only. Recall the advice:
# {per_section_tips[section]}
# Make this complete in this pass; do not leave any placeholders.

# Pay attention to how it fits in with the rest of the paper.
# Identify any redundancies (e.g., repeated figures or repeated text); if there are any, decide where in the paper things should be cut.
# Identify where we can save space and be more concise without weakening the message of the text.
# Fix any remaining errors as before:
# {error_list}
# """
#             )
#         )

#         code_modification = await ai.generate_object(
#             model=model,
#             messages=messages,
#             type=CodeModification,
#         )

#         apply_code_modifications(folder_name, code_modification.files)
#         messages.append(ai.message.assistant(code_modification.message))

#     # Finally, generate LaTeX
#     await generate_latex(folder_name, f"{folder_name}/{idea.name}.pdf")


# def apply_code_modifications(folder_name: str, files: Dict[str, str]) -> None:
#     for filename, content in files.items():
#         filepath = osp.join(folder_name, filename)
#         with open(filepath, "w") as f:
#             f.write(content)
#         print(f"Updated {filepath}")


# @ai.task()
# async def add_citations(
#     draft: str,
#     total_rounds: int = 20,
#     model: str = "openai:gpt-4-turbo",
# ) -> str:
#     messages = [
#         ai.message.system(
#             """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
# You have already written an initial draft of the paper and now you are looking to add missing citations to related papers throughout the paper.
# The related work section already has some initial comments on which papers to add and discuss.

# Focus on completing the existing write-up and do not add entirely new elements unless necessary.
# Ensure every point in the paper is substantiated with sufficient evidence.
# Feel free to add more cites to a particular point if there is only one or two references.
# Ensure no paper is cited without a corresponding reference in the `references.bib` file.
# Ensure each paragraph of the related work has sufficient background, e.g., a few papers cited.
# You will be given access to the Semantic Scholar API; only add citations that you have found using the API.
# Aim to discuss a broad range of relevant papers, not just the most popular ones.
# Make sure not to copy verbatim from prior literature to avoid plagiarism."""
#         ),
#     ]

#     for current_round in range(1, total_rounds + 1):
#         messages.append(
#             ai.message.user(
#                 f"""Round {current_round}/{total_rounds}:

# You have written this LaTeX draft so far:

# {draft}

# Identify the most important citation that you still need to add, and the query to find the paper.
# """
#             )
#         )

#         result = await ai.generate_object(
#             model=model,
#             messages=messages,
#             type=Think[Dict],
#         )

#         if "No more citations needed" in result.thought:
#             print("No more citations needed.")
#             break

#         query = result.action.get("Query")
#         description = result.action.get("Description")

#         papers = await search_for_papers(query)

#         if not papers:
#             print("No papers found.")
#             continue

#         papers_str = "\n\n".join(
#             f"{i}: {paper.title}. {paper.authors}. {paper.venue}, {paper.year}.\nAbstract: {paper.abstract}"
#             for i, paper in enumerate(papers)
#         )

#         messages.append(
#             ai.message.user(f"Search has recovered the following articles:\n{papers_str}")
#         )

#         result = await ai.generate_object(
#             model=model,
#             messages=messages,
#             type=Think[Dict],
#         )

#         if "Do not add any" in result.thought:
#             print("Do not add any.")
#             continue

#         selected_indices_str = result.action.get("Selected", "[]")
#         selected_indices = json.loads(selected_indices_str)
#         description = result.action.get("Description")

#         if not selected_indices:
#             continue

#         bibtex_entries = "\n".join(papers[i].citationStyles["bibtex"] for i in selected_indices)
#         draft = insert_bibtex_into_draft(draft, bibtex_entries)
#         draft = apply_description_to_draft(draft, description)

#         messages.append(ai.message.assistant(result.thought))
#         messages.append(ai.message.assistant(json.dumps(result.action, indent=2)))

#     return draft


# def insert_bibtex_into_draft(draft: str, bibtex_entries: str) -> str:
#     pattern = r"(\end{filecontents})"
#     replacement = f"{bibtex_entries}\n\\end{{filecontents}}"
#     return re.sub(pattern, replacement, draft, count=1)


# def apply_description_to_draft(draft: str, description: str) -> str:
#     draft += f"\n% Applied modification:\n{description}"
#     return draft


# @ai.task()
# async def generate_latex(
#     folder_name: str,
#     pdf_file: str,
#     num_error_corrections: int = MAX_LATEX_ERROR_CORRECTIONS,
# ) -> None:
#     """Generate a PDF from the LaTeX files in the given folder."""
#     latex_dir = osp.join(folder_name, "latex")

#     # Ensure output directory exists
#     os.makedirs(osp.dirname(pdf_file), exist_ok=True)

#     for _ in range(num_error_corrections):
#         try:
#             # Run pdflatex twice to resolve references
#             subprocess.run(
#                 ["pdflatex", "-interaction=nonstopmode", "template.tex"],
#                 cwd=latex_dir,
#                 check=True,
#                 capture_output=True,
#                 timeout=MAX_LATEX_COMPILE_TIMEOUT,
#             )
#             subprocess.run(
#                 ["pdflatex", "-interaction=nonstopmode", "template.tex"],
#                 cwd=latex_dir,
#                 check=True,
#                 capture_output=True,
#                 timeout=MAX_LATEX_COMPILE_TIMEOUT,
#             )

#             # Copy the generated PDF to the target location
#             shutil.copy(osp.join(latex_dir, "template.pdf"), pdf_file)
#             return

#         except subprocess.CalledProcessError as e:
#             logger.error(f"LaTeX compilation failed: {e.output.decode()}")
#             continue
#         except subprocess.TimeoutExpired:
#             logger.error("LaTeX compilation timed out")
#             continue

#     raise RuntimeError(f"Failed to compile LaTeX after {num_error_corrections} attempts")
