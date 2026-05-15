# word3vec

It has been well observed that popular word embeddings (word2vec, GloVe) preserve analogy parallelograms. Indeed, if $v_a, v_b, v_c, v_d \in \mathbb{R}^d$ are some low-dimensional embeddings corresponding to an analogy quadruple $a : b :: c : d$, it has been emperically observed that $v_a - v_b \approx v_c - v_d$

Prior theoretical explanations for this phenomenon (Arora et al., 2016; Gittens, Achlioptas, and Mahoney, 2017; Korchinski et al., 2025) assume some latent probabilistic model for the generation of language which induces a certain structure on the word co-occurence matrix ... which in turn leads to analogy preserving embeddings. On the contrary, our research project proposes that analogy parallelograms already exist in a ground-truth representation of words-- even before text is generated. We start by defining words in the vocabulary $$\{w_1, ... ,w_m\}$$ in terms of an exhaustive set of concepts $C$ and then define analogies in terms of concept set differences. Our main claim is that the co-occurence probability of words $w_i, w_j$ for some window $\delta$ is (roughly) some monotonically increasing function of the number of words $w$ that share common concepts with $w_i, w_j$. We then argue that the resulting co-occurence matrix leads to embeddings which preserve analogy parallelograms (our project will also include some interesting results on the minimum dimension requiredf for embeddings in order to preserve analogy parallelograms).

Formally, we define a set of words $w_1, ... ,w_m$ and a set of concepts $C$. Each word has a cet of concepts associated with it, which we call $C(w)$; the set $W(c)$ represents the number of words associated with concept $c \in C$. Then we say that $w_i, w_j, w_k, w_l$ form an analogy if:

$$C(w_i)-C(w_j) = C(w_k)-C(w_l)$$ and
$$C(w_j)-C(w_i) = C(w_l)-C(w_k)$$

where the minus sign here represents set differences. Furthermore, we define word similarity $s(w_i,w_j)= |C(w_i) \cap C(w_j)|$ (note, this is just one way of defining word similarity; we are still working on this and have proposed alternate definitions as well) and claim that the probability of two words co-occuring in a window $\delta$ is given by $P_\delta[wi, w_j] \propto f(s(w_i,w_j))$ where $f$ is some monotonically increasing function. Our claim is that the resulting co-occurence matrix leads to embeddings that preserve analogy parallelograms.

This theoretical framework leaves a few questions that require some empirical answer: 1) can analogy preserving embeddings really be derived from
