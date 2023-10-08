
# Portability


## Methodology
* We sampled from the codeparrot/codeparrot-clean huggingface dataset and found all items that imported tensorflow and pytorch
* We used https://github.com/cedricrupb/code_tokenize to parse tokens in the filtered files.
* We iterated through the pytorch and tensorflow model structure to get function and class names
* We counted all identifiers found in the torch and tensorflow module structure to get a frequency dict for both frameworks.
* We sampled five functions/classes per decile for both frameworks. We also kept the top 20.
* We went through the tensorflow and pytorch test suites searching for relevant tests for each sampled function
* For some we could not find a relevant test. Sometimes these functions did just not have tests, other times it was a function that was seen frequently in the dataset but did not seem likely to actual show up much. These are often false positives like `append` or `range`.
* We resampled from the deciles to replace these making sure to avoid any already in the sampled set.
* We then ran each of these tests on TPUs and GPUs and recorded what worked and what did not.