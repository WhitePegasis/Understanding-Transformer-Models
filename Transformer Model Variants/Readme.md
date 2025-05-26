# Transformer Architecture Variants: Encoder-Only vs Decoder-Only vs Encoder-Decoder

Modern Transformers come in **three main styles** – encoder-only, decoder-only, and encoder–decoder (seq2seq). Each has a distinct structure, attention pattern, and training objective, tailored to different tasks. In brief:

* **Encoder-only (e.g. BERT, RoBERTa)** – stacks of *encoder* layers only. Full self-attention (bidirectional), trained with masked-token objectives. Excels at understanding tasks (classification, NER, extractive QA).
* **Decoder-only (e.g. GPT series, LLaMA)** – stacks of *decoder* layers only. Causal self-attention (masked to previous tokens), trained autoregressively (next-token prediction). Optimized for text generation and completion.
* **Encoder–Decoder (e.g. T5, BART, Marian)** – combines encoders and decoders in one model. Encoder does full attention, decoder does causal + cross-attention to encoder outputs. Trained on sequence-to-sequence tasks (e.g. text→text), great for translation, summarization, etc..

Each variant’s design (layer types and attention masks) dictates how data flows through the model and what tasks it’s best at. We’ll illustrate each with diagrams, bullet lists, and an easy comparison table.

## 🔍 Encoder-Only Models (e.g. BERT, RoBERTa)

&#x20;*Figure: In encoder-only Transformers, the input flows through a stack of encoder blocks to produce contextual representations.*

Encoder-only Transformers consist **solely of encoder layers**.  In each encoder block, tokens attend to *all* other tokens in the input (full or “bi-directional” self-attention), allowing the model to build rich contextual embeddings of the entire sequence. There is no decoder or causal masking. Key points:

* **Layer composition:** A stack of *N encoder blocks*. Each block has **multi-head self-attention** (allowing every token to attend to every other token) followed by a feed-forward network (with LayerNorm and residuals).
* **Attention pattern:** *Full (bidirectional)* attention – every position can see the whole input (no future masking).
* **Data flow:** Input tokens → embedding & positional encoding → all encoder layers → output hidden states. Often the final hidden states feed into task-specific heads (classification or token prediction).
* **Training objective:** Usually **masked language modeling** – randomly mask out some input tokens and train the model to predict them. (E.g., BERT also used “next sentence prediction,” but masking is the core.)
* **Use cases:** Tasks that require understanding and analyzing text. Common downstream tasks include sentence/classification, named-entity recognition, extractive question answering, etc. (All tokens’ context is used for predictions.)
* **Example models:** BERT (Base and Large), RoBERTa, ALBERT, DistilBERT, and many language encoders (including multilingual or domain-specific variants). These are often used as “backbones” for classification or tagging tasks.

## 🤖 Decoder-Only Models (e.g. GPT, LLaMA, Gemma)

&#x20;*Figure: Decoder-only Transformers input a prompt and generate outputs one token at a time, using causal (masked) self-attention.*

Decoder-only Transformers consist **only of decoder layers**. They are **auto-regressive**: at each step, the model predicts the next token given all previous tokens. Crucially, **each position’s attention is masked** so tokens can only attend to earlier positions. Highlights:

* **Layer composition:** A stack of *N decoder blocks*. Each block has **self-attention (causal)** – tokens can only see earlier tokens – followed by a feed-forward network.  (Since there is no separate encoder, there is *no cross-attention* sublayer.)
* **Attention pattern:** *Causal (unidirectional)* self-attention.  This is implemented with a triangular mask so that token *i* cannot attend to tokens *j>i*. This enforces the “predict next token” rule.
* **Data flow:** The entire input sequence (or prompt) is fed into the decoder stack. During generation, the model iteratively feeds back its own outputs as new inputs (auto-regressively). There is no separate encoder context.
* **Training objective:** **Auto-regressive language modeling.** Typically trained to predict the next token in a large text corpus. All text is available as training data (no need for input–output pairs).
* **Use cases:** **Text generation and completion.** Because they generate fluently one token at a time, decoder-only models excel at tasks like free-form text generation, story or dialogue generation, and any task that can be cast as continuing a prompt. They can also be steered for certain tasks via prompt design (e.g., few-shot classification, QA) but inherently have less bidirectional understanding than encoders.
* **Example models:** OpenAI’s GPT family (GPT-1, GPT-2, GPT-3, GPT-4) – all are decoder-only models. Newer large models like Meta’s LLaMA, Google’s Gemma, Microsoft’s Megatron-Turing, etc., also follow this design.

## 🔄 Encoder–Decoder Models (Seq2Seq, e.g. T5, BART)

&#x20;*Figure: Encoder–Decoder Transformers use an encoder to read the input and a decoder (with cross-attention) to generate the output.*

Encoder–decoder (a.k.a. *sequence-to-sequence*) Transformers use **both** an encoder stack and a decoder stack. The encoder reads the entire input sequence first; the decoder then generates an output sequence token by token, attending to the encoder’s outputs as context. Key characteristics:

* **Layer composition:** The encoder is a stack of standard encoder blocks (self-attention + feed-forward). The decoder is a stack of modified decoder blocks: each has *causal self-attention*, **cross-attention** to the encoder outputs, and a feed-forward network.
* **Attention patterns:**  The encoder uses full (bidirectional) self-attention (like encoder-only). The decoder uses *causal* self-attention (like decoder-only) plus an extra multi-head cross-attention sublayer. In cross-attention, the decoder’s queries come from the previous decoder layer, while keys/values come from the encoder’s final hidden states.
* **Data flow:** Input → encoder → produces “context” features. Then the decoder takes the encoded features plus (shifted) output tokens, and alternates self-attending (to past outputs) and cross-attending (to encoder features) to generate each new token.
* **Training objective:** **Sequence-to-sequence (text-to-text)** training. Commonly, the model is fed input–output pairs: e.g., translate an input sentence into a target language or summarize an input paragraph. T5’s pre-training is done by corrupting input text (masking spans) and training the decoder to predict the missing spans. BART’s pre-training masks or deletes spans and the model must reconstruct them. In general, the loss is computed over the target sequence tokens.
* **Use cases:** Any task where one sequence must be transformed into another. Classic examples include machine translation, abstractive summarization, text simplification, question answering (generative answer given a context), code generation (specification → code), etc.. Because they have a bidirectional encoder, they understand input deeply, and because they have a causal decoder, they can generate fluent outputs.
* **Example models:** T5 and its variants (mT5, Flan-T5), BART, MBART, MarianMT (translation), Pegasus, ProphetNet, and other seq2seq pre-trained models.

## 📊 Comparison Table

| **Model Type**  | **Layers & Attention**                                        | **Training Objective**                   | **Typical Tasks**                                     | **Example Models**                 |
| --------------- | ------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------- | ---------------------------------- |
| Encoder-Only    | Only encoder blocks (self-attn + FF); full (bi-\*\*)          | Masked LM (predict masked tokens)        | Classification, NER, extractive QA                    | BERT, RoBERTa, ALBERT, DistilBERT  |
| Decoder-Only    | Only decoder blocks (self-attn \[causal] + FF)                | Next-token (auto-regressive) LM          | Text generation/completion, dialogue                  | GPT-1/2/3/4, LLaMA, Gemma, GPT-Neo |
| Encoder–Decoder | Encoder blocks + decoder blocks (self-attn + cross-attn + FF) | Seq2seq (predict output seq given input) | Machine translation, summarization, Q\&A (generative) | T5, BART, mBART, Pegasus, MarianMT |

**Key distinctions:** Encoder-only models have **bidirectional attention** and are trained to reconstruct or classify inputs. Decoder-only models use **causal (one-way) attention** and are trained to predict the next token. Encoder–decoder models combine both: the encoder does bidirectional self-attention, while the decoder does masked self-attention plus encoder–decoder cross-attention. This makes them powerful for “transforming” input sequences into outputs.

Each architecture optimizes for different needs, which is why large-model designers choose one over the other depending on task: e.g. **GPT**-style models went decoder-only for scalable text generation, **BERT**-style encoders dominate understanding tasks, and **T5/BART** seq2seq models excel when input and output differ in domain or format.

**Sources:** Authoritative discussions and model docs on Transformer variants. Each cited source details aspects of layer composition, attention masks, training objectives, and example uses for these model classes.

https://huggingface.co/learn/llm-course/en/chapter1/6

https://en.wikipedia.org/wiki/BERT_(language_model)

https://en.wikipedia.org/wiki/Generative_pre-trained_transformer

https://machinelearningmastery.com/encoders-and-decoders-in-transformer-models/

https://huggingface.co/blog/encoder-decoder





https://machinelearningmastery.com/encoders-and-decoders-in-transformer-models/
