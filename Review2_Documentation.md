# Review 2 - Multilingual Chatbot with Ayurvedic Domain

> Deadline: April 3, 2026  
> Deliverable: Documentation with project demo  
> Basis: This document is written from the current repository state. Wherever benchmark logs are not stored in the repo, the analysis is marked as qualitative or inferred from implementation artifacts.

---

## 1. Project Overview

This project is a bilingual Ayurvedic domain chatbot that supports Hindi and English queries through a Hindi-primary Retrieval-Augmented Generation (RAG) pipeline. The system combines:

- `google/mt5-small` for answer generation
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for multilingual embeddings
- FAISS for retrieval over an Ayurvedic knowledge base
- `langdetect` and `deep-translator` for language detection and English-Hindi conversion
- Streamlit for the interactive demo interface

The overall flow is:

1. Detect user language
2. Translate English query to Hindi when needed
3. Retrieve top relevant Ayurvedic passages using FAISS
4. Generate a Hindi response with mT5
5. Apply a grounded fallback if the generated answer is weak
6. Translate the final response back to English if the user asked in English

---

## 2. Data Preprocessing Steps

### 2.1 Data sources integrated into the project

The preprocessing pipeline is designed to consume five sources:

1. `hindi_dataset.csv` for clinical symptom-treatment style Q&A
2. `bhashbench_ayur_hindi.json` for Ayurvedic MCQ-style benchmark content
3. `himed_trad_bench.json` for benchmark questions and explanations
4. `himed_trad_corpus.json` for large-scale traditional medicine QA/dialogue/MCQ data
5. `ashtanga.txt` for classical Ayurvedic text passages

### 2.2 Source-specific preprocessing

#### A. Hindi clinical dataset

- Reads symptoms, treatment, diagnosis, and patient history
- Converts each record into one or more Hindi Q&A pairs
- Creates both direct treatment answers and diagnosis-oriented answers
- Adds these records to both the generative training set and the RAG knowledge base

#### B. BhashaBench-Ayur Hindi

- Extracts the correct option text from MCQ entries
- Appends topic/domain context to make answers more descriptive
- Routes these entries only to the RAG knowledge base
- Keeps them out of the final generative training split to reduce MCQ-style answer bias

#### C. HiMed benchmark

- Routes MCQs to the RAG knowledge base
- Routes QA and dialogue items to the generative training set
- Uses chain-of-thought text when it is richer than the short answer field

#### D. HiMed corpus

- Separates MCQ, QA, and dialogue entries
- Samples QA and dialogue data for generative training
- Uses a large MCQ sample for the retrieval knowledge base
- Prefers richer explanation fields when available

#### E. Classical Ayurvedic text processing

The file `chatbot/data/process_classical_text.py` performs custom processing over `ashtanga.txt`:

- Skips the first 900 lines to remove front matter and non-content pages
- Removes OCR junk, page numbers, and publisher noise
- Detects chapter and section headers
- Chunks long passages at sentence boundaries
- Uses:
  - minimum passage length: 80 characters
  - target chunk length: 500 characters
  - maximum chunk length: 2000 characters

### 2.3 Saved preprocessing outputs

Current repository artifacts show:

| Artifact | Current Count |
|---|---:|
| `train.json` | 22,552 |
| `val.json` | 2,653 |
| `test.json` | 1,327 |
| Total current generative dataset | 26,532 |
| `knowledge_base.json` | 55,974 |
| `classical_passages.json` | 2,183 |
| FAISS metadata entries | 58,157 |

The FAISS metadata count is larger than `knowledge_base.json` because `build_kb.py` merges the knowledge base with the separately processed classical passages during indexing.

### 2.4 Important preprocessing observation

The repository contains two dataset split definitions:

- `config.py` defines an older `80/10/10` split
- `preprocess_all.py` actually uses `85/10/5`

For Review 2, the implemented script should be treated as the source of truth because that is what produced the saved data files.

---

## 3. Model Implementation Details

### 3.1 Generator model

- Base model: `google/mt5-small`
- Architecture: multilingual sequence-to-sequence transformer
- Role: generate grounded Ayurvedic responses in Hindi

### 3.2 Parameter-efficient fine-tuning

The project uses LoRA-style fine-tuning over mT5:

- LoRA rank `r = 16`
- LoRA alpha `= 32`
- LoRA dropout `= 0.05`
- Target modules: `q`, `v`

### 3.3 Retrieval model

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Embedding size: `384`
- Retrieval backend: FAISS `IndexFlatIP`
- Retrieval strategy: cosine-style similarity via normalized embeddings

### 3.4 Language support

- Language detection: `langdetect`
- Translation: `deep-translator` with Google Translate backend
- Design choice: keep the core reasoning and retrieval pipeline Hindi-first, then translate at the edges

### 3.5 Runtime safety and fallback

The inference pipeline includes two practical safeguards:

1. If a local fine-tuned checkpoint is unavailable, the app falls back to the base mT5 model.
2. If the generated answer is too short or too generic, the system constructs a grounded response from retrieved passages.

This is important for demo stability because the chatbot can still respond even when generation quality drops.

### 3.6 Current model artifact status

The repository currently contains a local full model checkpoint:

- `chatbot/models/mt5_ayurvedic_lora/model.safetensors`
- Size: about `2.07 GB`

So the project is in a demo-ready state, not just a script-only state.

---

## 4. Hyperparameters Used

### 4.1 Fine-tuning hyperparameters

| Hyperparameter | Value |
|---|---:|
| Train batch size | 4 |
| Eval batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective train batch size | 16 |
| Learning rate | `3e-4` |
| Number of epochs | 3 |
| Max input length | 512 |
| Max target length | 256 |
| Warmup steps | 100 |
| Weight decay | 0.01 |

### 4.2 Generation hyperparameters

| Hyperparameter | Value |
|---|---:|
| Number of beams | 4 |
| Max generation length | 256 |
| Repetition penalty | 1.2 |
| Length penalty | 1.0 |

### 4.3 Retrieval hyperparameters

| Hyperparameter | Value |
|---|---:|
| Top-k retrieval | 5 |
| Embedding dimension | 384 |

---

## 5. Training and Validation Strategy

### 5.1 Training strategy

The fine-tuning script formats each sample as:

- Input: `Question: <Hindi question>`
- Target: `Hindi answer`

Training is performed using Hugging Face `Seq2SeqTrainer` with:

- periodic logging every 50 steps
- validation every 200 steps
- checkpoint saving every 500 steps
- `load_best_model_at_end=True`
- best model selected using `eval_loss`

### 5.2 Validation strategy

The repository also includes `chatbot/data/validate_data.py`, which validates:

- empty questions or answers
- extremely short answers
- unwanted `extra_id` token leakage
- BhashaBench answer extraction correctness
- source distribution
- duplicate questions
- script coverage for classical text data

### 5.3 Current validation evidence

From the saved validation report and current processed files:

- Missing questions: 0
- Missing answers: 0
- `extra_id` contamination: 0
- BhashaBench verification in saved report: 100/100 correct samples
- Current post-processed dataset size: 26,532 examples

---

## 6. Regularization Techniques Applied

The project uses several regularization and robustness strategies:

1. LoRA dropout (`0.05`) to reduce overfitting in adapter training
2. Weight decay (`0.01`) during training
3. Padding labels converted to `-100` so padded tokens do not contribute to loss
4. Retrieval grounding to keep answers tied to evidence passages
5. Post-processing fallback when the model output is too weak
6. Dataset balancing script to reduce over-dominance of short MCQ-style answers

The Colab notebook also contains an additional stability safeguard:

- `fp16=False` and `bf16=False` are used because mT5's large vocabulary can cause floating-point overflow and `NaN` loss under mixed precision in this setup.

---

## 7. Optimization Algorithms Used

### 7.1 What is explicitly implemented

In the current executable local fine-tuning script:

- optimizer is not explicitly overridden in `Seq2SeqTrainingArguments`
- warmup scheduling is explicitly used through `warmup_steps=100`
- gradient accumulation is used to simulate a larger effective batch size

### 7.2 Important repo observation

Some older project notes mention `paged_adamw_8bit` and cosine scheduling, but these are not explicitly configured in the current `chatbot/models/fine_tune_mt5.py`. Therefore, for review purposes:

- the local script should be treated as the actual implemented training setup
- the older documentation should be treated as design intent or earlier planning

---

## 8. Experimental Setup

### 8.1 Hardware targets

The project is designed for:

- RTX 3050 with 4 GB VRAM for compact fine-tuning or inference
- Google Colab T4 for more comfortable training
- CPU mode for fallback inference

### 8.2 Software stack

Core dependencies include:

- PyTorch
- Transformers
- PEFT
- Sentence Transformers
- FAISS CPU
- Streamlit
- Deep Translator
- Langdetect

### 8.3 Demo environment

The project includes:

- `run_chatbot.bat` for quick startup on Windows
- `chatbot/app.py` for Streamlit UI
- a sidebar showing retrieved passages and retrieval scores
- automatic bilingual query handling

### 8.4 Current data and retrieval scale

The live retrieval layer currently operates over:

- 55,974 knowledge-base entries from processed Q&A sources
- 2,183 classical text passages
- 58,157 total indexed FAISS vectors

---

## 9. Results and Performance Analysis

### 9.1 What is strongly supported by repository evidence

The current repository proves that:

- the full bilingual RAG pipeline is implemented end-to-end
- the FAISS index has already been built
- the Streamlit demo is available
- a local full model checkpoint exists
- the preprocessing pipeline has produced large-scale training and retrieval artifacts

### 9.2 Quantitative data quality evidence

Current artifact-backed quality indicators include:

| Metric | Value |
|---|---:|
| Current fine-tuning examples | 26,532 |
| Knowledge base entries | 55,974 |
| Total FAISS vectors | 58,157 |
| Average question length | 97.93 chars |
| Average answer length | 136.54 chars |
| Classical passages | 2,183 |
| Classical passage average length | 428.17 chars |
| Unique classical chapters | 259 |

### 9.3 Honest limitation on benchmark metrics

The repo does not currently store final benchmark outputs such as BLEU, ROUGE, exact match, F1, or human evaluation tables. Because of that, performance analysis for Review 2 should be presented as:

- implementation-complete
- data-prepared
- demo-capable
- partially validated

but not yet fully benchmarked in a reproducible evaluation report.

---

## 10. Convergence Analysis

### 10.1 What the training pipeline does to encourage convergence

The training setup includes:

- warmup steps
- best-model selection by `eval_loss`
- periodic evaluation
- gradient accumulation
- maximum sequence lengths to keep computation stable
- notebook-level checks for `NaN` and `Inf` loss before full training

### 10.2 Practical convergence interpretation

Because the repo does not include saved training curves or loss logs, convergence can only be analyzed indirectly:

- the project team anticipated instability with mT5 mixed precision and explicitly disabled it in Colab
- the notebook includes a forward-pass sanity test to verify valid loss before long training
- the existence of a working local checkpoint suggests training completed successfully at least once

### 10.3 Review statement

For presentation, it is accurate to say:

> The project includes convergence safeguards and a successfully saved model checkpoint, but a full loss-vs-epoch chart is not currently preserved in the repository.

---

## 11. Error Analysis

### 11.1 Data-level issues observed

The validation artifacts reveal the following challenges:

- a small number of very short answers still remain in the dataset
- duplicate questions are still present after preprocessing
- some benchmark-derived entries contain terse or option-like responses that are less suitable for generation

Current post-processed checks over the saved train/val/test files show:

| Issue | Current Count |
|---|---:|
| Missing questions | 0 |
| Missing answers | 0 |
| Very short answers under 10 chars | 18 |
| Entries with `extra_id` tokens | 0 |
| Duplicate questions | 3,305 |

### 11.2 System-level issues addressed in code

The implementation directly addresses several failure modes:

1. Weak or generic generation is replaced by a grounded fallback answer.
2. English input is translated to reduce retrieval mismatch.
3. Retrieved passages are shown in the UI for transparency.
4. Base-model-only inference is still supported when a fine-tuned model is missing.

### 11.3 Remaining error risks

- Translation may distort Ayurvedic terminology in English queries.
- Duplicate training questions may bias generation.
- Some short answers remain too sparse for high-quality explanatory output.
- Retrieval quality depends on embedding similarity and may surface semantically nearby but clinically incomplete passages.

---

## 12. Comparison with Baseline Model

### 12.1 Baseline used in practice

The practical baseline in this repository is the unfine-tuned base `mT5-small` model with retrieval assistance.

### 12.2 Comparison summary

| Aspect | Base mT5 / weak baseline | Current project system |
|---|---|---|
| Domain specialization | General multilingual | Ayurvedic-domain adapted |
| Retrieval grounding | Limited by prompt only | Explicit FAISS-based RAG |
| English handling | Not task-specific | Translation-assisted bilingual flow |
| Reliability | Can be generic or weak | Adds grounded fallback |
| Demo transparency | None by default | Retrieved passages shown in UI |

### 12.3 Evidence-based inference

Inference from the codebase:

- the project explicitly keeps a fallback for weak base-model outputs
- therefore the team already identified that raw base mT5 is not reliable enough for final-domain answers
- the fine-tuned and retrieval-grounded system is the intended production/demo path

This is a code-supported qualitative comparison, not a benchmark table.

---

## 13. Observations and Inferences

1. The project has moved beyond planning and into deployable prototype stage because the data pipeline, FAISS index, UI, and full checkpoint all exist.
2. The system is intentionally Hindi-first, which is a strong design fit for Ayurvedic terminology and source material.
3. Retrieval is not just an add-on here; it is a core reliability mechanism for both demo quality and hallucination reduction.
4. The training pipeline shows careful engineering for limited hardware constraints.
5. The repo still contains some documentation drift, so executable scripts should be treated as more trustworthy than earlier narrative files.

---

## 14. Challenges Faced and Solutions

| Challenge | Impact | Solution Applied |
|---|---|---|
| Limited GPU memory | Hard to fine-tune full multilingual models | Used LoRA-style parameter-efficient fine-tuning and compact training setup |
| mT5 mixed-precision instability | Risk of `NaN` loss during training | Disabled `fp16`/`bf16` in Colab notebook and added loss sanity checks |
| MCQ-heavy data skew | Model can learn short option-style answers | Routed many MCQs to RAG and added balancing over train/val files |
| OCR noise in classical text | Dirty passages reduce retrieval quality | Built custom text cleaning and chunking pipeline |
| Bilingual requirement | English queries can miss Hindi knowledge | Added language detection and translation at input/output stages |
| Weak generative answers | Poor demo reliability | Added grounded fallback using retrieved evidence |
| Documentation mismatch | Risk of inconsistent reporting | Review 2 should use code artifacts as the primary evidence source |

---

## 15. Future Enhancements

1. Add a formal evaluation suite with BLEU, ROUGE, retrieval recall, and human review.
2. Save training curves and evaluation logs automatically after every run.
3. Deduplicate the final training data more aggressively.
4. Replace generic translation with a more domain-aware Indic translation model for production.
5. Add source citation formatting in final answers, not just passage display in the UI.
6. Introduce safety filters and stronger medical disclaimers for high-risk health questions.
7. Align `config.py`, notebook settings, and training script so there is one single source of truth.
8. Benchmark the fine-tuned checkpoint directly against base mT5 and retrieval-only baselines.

---

## 16. Project Demo Notes

### 16.1 How to run the demo

From the repository root:

```powershell
run_chatbot.bat
```

Or manually:

```powershell
cd chatbot
streamlit run app.py
```

### 16.2 Demo features to show in Review 2

1. Ask one Hindi query and one English query.
2. Show automatic language handling.
3. Show retrieved passages in the sidebar/expander.
4. Explain that the answer is produced through retrieval plus generation.
5. Mention that the system can fall back to grounded passages when generation is weak.

### 16.3 Suggested demo questions

- What are the benefits of Ashwagandha?
- How can Vata dosha be balanced?
- What is Triphala?
- Explain an Ayurvedic remedy for digestive imbalance.

---

## 17. Final Conclusion

The Multilingual Chatbot with Ayurvedic Domain is a strong applied deep learning project with a clear domain focus, a practical bilingual design, and a complete RAG-based implementation. Its main strengths are the Hindi-first knowledge strategy, the large retrieval corpus, the classical-text integration pipeline, and the stable demo-oriented fallback design.

The main area still pending for a research-quality finish is reproducible quantitative evaluation. For Review 2, the project can be confidently presented as an implemented and demo-ready system, with formal benchmark reporting identified as the next major milestone.
