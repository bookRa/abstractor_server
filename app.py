from flask import Flask, render_template, request
from flask_cors import CORS
import json
import sys
import random
import re
import numpy as np
import pandas as pd

import gensim.utils as genutils
from gensim import corpora, similarities
import gensim.parsing.preprocessing as genpre
import pickle
from gensim.models import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)
CORS(app)

####### RELATED TO PROCESSING AND COLoRIZING #######

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


lmtzr= WordNetLemmatizer()
def prep_text(text):
     #this regex removes LATEX formatting, numbers, citations, splits hyphens into two words
    myreg=r'\\[\w]+[\{| ]|\$[^\$]+\$|\(.+\, *\d{2,4}\w*\)|\S*\/\/\S*|[\\.,\/#!$%\^&\*;:{}=_`\'\"~()><\|]|\[.+\]|\d+|\b\w{1,2}\b'
    parsed_data = text.replace('-', ' ')
    parsed_data = re.sub(myreg, '', parsed_data)
    parsed_data = [lmtzr.lemmatize(w) for w in parsed_data.lower().split() if w not in genpre.STOPWORDS]
    # print("From PrepText: ", parsed_data)
    if len(parsed_data) ==1: return parsed_data[0]
    return parsed_data

full_ldaModel = LdaModel.load('./final_models/full_ldaModel')
full_dict = corpora.Dictionary.load('./final_models/full_dict')

def colorize(ab, model=full_ldaModel, dictionary=full_dict):
    """
    takes an abstract, an LDA model, and dictionary, and returns json with each word mapped to it's topic (or -1 if n/a)
    
    Still needs refinement to take advantage of bigrams
    """
    proc_ab = prep_text(ab)
    ab_bow = dictionary.doc2bow(proc_ab) # not using bigrams for now b/c it complicates things
    doc_topics, word_topics, phi_values = model.get_document_topics(ab_bow, per_word_topics = True)
    # print(word_topics)
    # topic_colors = {0: 'red', 1: 'yellow', 2:'green', 3:'blue', 4:'aqua', 5: "magenta", 6:"purple"} # Have to figure out how to not hard-code this for the future
    raw_ab = ab.split()
    raw_proc = ["".join(prep_text(w)) for w in raw_ab] #this is a hack b/c the preprocessing is off
    # print(raw_proc)
    result = []
    for i in range(len(raw_proc)):
        if not raw_proc[i] or raw_proc[i] not in dictionary.token2id:
            # print("Not in dictionary", raw_proc[i])
            result.append((raw_ab[i], -1))
        else:
            sample = [w for w in word_topics if w[0] == dictionary.token2id[raw_proc[i]]]
            # print("In dictionary: ",sample, dictionary.token2id[raw_proc[i]])
            if len(sample)==0 or len(sample[0][1])==0: result.append((raw_ab[i], -1))
            else: 
                match = sample[0] 
                # print(match)
                result.append((raw_ab[i], match[1][0]))
    final_dict={
        "words":result,
        "topics": [(int(t), float(p)) for t,p in doc_topics]
    }
    return json.dumps(final_dict)
####### END PROCESSING AND COLoRIZING #######

####### BEGIN SIMILARITIES #######
with open('./models/abstracts.json') as f:
    all_abstracts= json.load(f)
index_sims = similarities.MatrixSimilarity.load('./final_models/full_index')   
bigrams = genutils.SaveLoad.load('./models/bigram_phrases')
full_corp_matrix = corpora.MmCorpus('./final_models/full_bigram_corpus.mm')
def get_similar_docs(querydoc, model=full_ldaModel, index=index_sims, tokenizer=prep_text, phraser=bigrams, dictionary=full_dict, top_n_docs=5): 
    """
    does a (I think?) cosine similarity with all documents in my corpus
    """
    # curr_text is a list of strings (including bigrams)
    prep_text = tokenizer(querydoc)
    # print("prep_text: ", prep_text)
    curr_text = phraser[prep_text]
    # print("curr_text: ", curr_text)
    curr_bow = dictionary.doc2bow(curr_text)
    # print("curr_bow, ", curr_bow)
    lda_analyzed= model[curr_bow]
    # print("lda_analyzed, ", lda_analyzed)
    #fit the model to the bow - convert to lda space
    sims = index[lda_analyzed]
    # print("sims: ", sims)
    omg = enumerate(sims)
    # print("omg: ", omg)
    fiji = sorted(omg, key=lambda i: -i[1])
    # print("fiji: ", fiji)
    return fiji[:top_n_docs]
    # return sorted(enumerate(sims), key= lambda i: -i[1])[:top_n_docs]

def display_similars(tup_array, model=full_ldaModel, corp_matrix=full_corp_matrix, corpus= all_abstracts):
    the_tops = [model[corp_matrix[sim[0]]] for sim in tup_array]
    print("THE TOPS: ", the_tops)
    for doc in range(len(the_tops)):
        print(doc)
        for top in range(len(the_tops[doc])):
            curr_top= the_tops[doc][top]
            the_tops[doc][top]= (float(curr_top[0]), float(curr_top[1]))

    print("the_tops: ", the_tops)

    the_cents = [float(sim[1]) for sim in tup_array]
    # print("the_cents: ", the_cents)

    the_text = [corpus[sim[0]]["summary"] for sim in tup_array]
    # print("the_text: ", the_text)

    the_ids = [corpus[sim[0]]["article_id"] for sim in tup_array]
    # print("the_ids: ", the_ids)

    result={
        "topics" : the_tops, #[[(float(t[0]), float(t[1])) for t in curr_doc] for curr_doc the_tops],
        "percentage" : the_cents,
        "text" : the_text,
        "link" : the_ids
    }
    # result = [{
    #     "topics":model[corp_matrix[sim[0]]],
    #     "percentage": float(sim[1]),
    #     "text":corpus[sim[0]["summary"]],
    #     "link":corpus[sim[0]["article_id"]]
    #     } for sim in tup_array]
#     print(type(result[1]["topics"]), type(result[1]["percentage"]), type(result[1]["text"]))
#     sim = tup_array[0]
#     print(sim[0], type(sim[0]))
#     print(corp_matrix[sim[0]], type(corp_matrix[sim[0]]))
#     print(model[corp_matrix[sim[0]]], type(model[corp_matrix[sim[0]]]))
#     return json.dumps(result) # for some reason it won't json-serialize at the moment...hmmm
    return result

####### END SIMILARITIES #######




@app.route("/abst_subm", methods=['POST'])
def about():
    if request.method =='POST':
        my_abst = request.get_json()['abstract'] # request.data.abstract
        # cleaned_abst = prep_text(my_abst)
        # return f"abstract is {cleaned_abst}"
        return colorize(my_abst)
    else:
        return "WTF mate...from FLASK"
    # return render_template('about.html', arr= my_first, title = "teaseit")

@app.route("/get_similars", methods=['POST'])
def similars():
    if request.method =='POST':
        my_abst = request.get_json()['abstract']
        q= get_similar_docs(my_abst)
        # print("q is: ", q)
        results = display_similars(q)
        print("results are: ", results)
        return json.dumps(results)#, cls=NumpyEncoder)
        # return(json.dumps({"Hello": 1, "World": 2}))
    else:
        return "Nope nope nope"


@app.route("/")
@app.route("/home")
def hello():
    # return render_template('home.html', title='Googoo')
    return "Ssgoinon"


@app.route("/get_rand_abst")
def provide_rand():
    return random.choice(all_abstracts)['summary']

randoAbstracts =[
    "We consider a correlated multi-armed bandit problem in which rewards of arms are correlated through a hidden parameter. Our approach exploits the correlation among arms to identify some arms as sub-optimal and pulls them only O(1) times. This results in significant reduction in cumulative regret, and in fact our algorithm achieves bounded (i.e., O(1)) regret whenever possible; explicit conditions needed for bounded regret to be possible are also provided by analyzing regret lower bounds. We propose several variants of our approach that generalize classical bandit algorithms such as UCB, Thompson sampling, KL-UCB to the structured bandit setting, and empirically demonstrate their superiority via simulations.",
    "Stochastic variational inference is an established way to carry out approximate Bayesian inference for deep models. While there have been effective proposals for good initializations for loss minimization in deep learning, far less attention has been devoted to the issue of initialization of stochastic variational inference. We address this by proposing a novel layer-wise initialization strategy based on Bayesian linear models. The proposed method is extensively validated on regression and classification tasks, including Bayesian DeepNets and ConvNets, showing faster convergence compared to alternatives inspired by the literature on initializations for loss minimization.",
    "Deep learning has shown high performances in various types of tasks from visual recognition to natural language processing, which indicates superior flexibility and adaptivity of deep learning. To understand this phenomenon theoretically, we develop a new approximation and estimation error analysis of deep learning with the ReLU activation for functions in a Besov space and its variant with mixed smoothness. The Besov space is a considerably general function space including the Holder space and Sobolev space, and especially can capture spatial inhomogeneity of smoothness. Through the analysis in the Besov space, it is shown that deep learning can achieve the minimax optimal rate and outperform any non-adaptive (linear) estimator such as kernel ridge regression, which shows that deep learning has higher adaptivity to the spatial inhomogeneity of the target function than other estimators such as linear ones. In addition to this, it is shown that deep learning can avoid the curse of dimensionality if the target function is in a mixed smooth Besov space. We also show that the dependency of the convergence rate on the dimensionality is tight due to its minimax optimality. These results support high adaptivity of deep learning and its superior ability as a feature extractor.",
    "In this article we propose a novel ranking algorithm, referred to as HierLPR, for the multi-label classification problem when the candidate labels follow a known hierarchical structure. HierLPR is motivated by a new metric called eAUC that we design to assess the ranking of classification decisions. This metric, associated with the hit curve and local precision rate, emphasizes the accuracy of the first calls. We show that HierLPR optimizes eAUC under the tree constraint and some light assumptions on the dependency between the nodes in the hierarchy. We also provide a strategy to make calls for each node based on the ordering produced by HierLPR, with the intent of controlling FDR or maximizing F-score. The performance of our proposed methods is demonstrated on synthetic datasets as well as a real example of disease diagnosis using NCBI GEO datasets. In these cases, HierLPR shows a favorable result over competing methods in the early part of the precision-recall curve.",
    "By using the viewpoint of modern computational algebraic geometry, we explore properties of the optimization landscapes of the deep linear neural network models. After clarifying on the various definitions of \"flat\" minima, we show that the geometrically flat minima, which are merely artifacts of residual continuous symmetries of the deep linear networks, can be straightforwardly removed by a generalized L2 regularization. Then, we establish upper bounds on the number of isolated stationary points of these networks with the help of algebraic geometry. Using these upper bounds and utilizing a numerical algebraic geometry method, we find all stationary points of modest depth and matrix size. We show that in the presence of the non-zero regularization, deep linear networks indeed possess local minima which are not the global minima. Our computational results clarify certain aspects of the loss surfaces of deep linear networks and provide novel insights.",
    "Learning of high-dimensional simplices from uniformly-sampled observations, generally known as the \"unmixing problem\", is a long-studied task in computer science. More recently, a significant interest is focused on this problem from other areas, such as computational biology and remote sensing. In this paper, we have studied the Probably Approximately Correct (PAC)-learnability of simplices with a focus on sample complexity. Our analysis shows that a sufficient sample size for PAC-learning of K-simplices is only O(K2logK), yielding a huge improvement over the existing results, i.e. O(K22). Moreover, a novel continuously-relaxed optimization scheme is proposed which is guaranteed to achieve a PAC-approximation of the simplex, followed by a corresponding scalable algorithm whose performance is extensively tested on synthetic and real-world datasets. Experimental results show that not only being comparable to other existing strategies on noiseless samples, our method is superior to the state-of-the-art in noisy cases. The overall proposed framework is backed with solid theoretical guarantees and provides a rigorous framework for future research in this area.",
    "Active learning aims to reduce annotation cost by predicting which samples are useful for a human teacher to label. However it has become clear there is no best active learning algorithm. Inspired by various philosophies about what constitutes a good criteria, different algorithms perform well on different datasets. This has motivated research into ensembles of active learners that learn what constitutes a good criteria in a given scenario, typically via multi-armed bandit algorithms. Though algorithm ensembles can lead to better results, they overlook the fact that not only does algorithm efficacy vary across datasets, but also during a single active learning session. That is, the best criteria is non-stationary. This breaks existing algorithms' guarantees and hampers their performance in practice. In this paper, we propose dynamic ensemble active learning as a more general and promising research direction. We develop a dynamic ensemble active learner based on a non-stationary multi-armed bandit with expert advice algorithm. Our dynamic ensemble selects the right criteria at each step of active learning. It has theoretical guarantees, and shows encouraging results on 13 popular datasets.",
    "Topic models such as the latent Dirichlet allocation (LDA) have become a standard staple in the modeling toolbox of machine learning. They have been applied to a vast variety of data sets, contexts, and tasks to varying degrees of success. However, to date there is almost no formal theory explicating the LDA’s behavior, and despite its familiarity there is very little systematic analysis of and guidance on the properties of the data that affect the inferential performance of the model. This paper seeks to address this gap, by providing a systematic analysis of factors which characterize the LDA’s performance. We present theorems elucidating the posterior contraction rates of the topics as the amount of data increases, and a thorough supporting empirical study using synthetic and real data sets, including news and web-based articles and tweet messages. Based on these results we provide practical guidance on how to identify suitable data sets for topic models, and how to specify particular model parameters.",
    "This paper presents the Averaged CVB (ACVB) inference and offers convergence-guaranteed and practically useful fast Collapsed Variational Bayes (CVB) inferences. CVB inferences yield more precise inferences of Bayesian probabilistic models than Variational Bayes (VB) inferences. However, their convergence aspect is fairly unknown and has not been scrutinized. To make CVB more useful, we study their convergence behaviors in a empirical and practical approach. We develop a convergence- guaranteed algorithm for any CVB-based inference called ACVB, which enables automatic convergence detection and frees non- expert practitioners from the difficult and costly manual monitoring of inference processes. In experiments, ACVB inferences are comparable to or better than those of existing inference methods and deterministic, fast, and provide easier convergence detection. These features are especially convenient for practitioners who want precise Bayesian inference with assured convergence.",
    "We study the design of interactive clustering algorithms. The user supervision that we consider is in the form of cluster split/merge requests; such feedback is easy for users to provide because it only requires a high-level understanding of the clusters. Our algorithms start with any initial clustering and only make local changes in each step; both are desirable properties in many applications. Local changes are desirable because in practice edits of other parts of the clustering are considered churn - changes that are perceived as quality-neutral or quality-negative. We show that in this framework we can still design provably correct algorithms given that our data satisfies natural separability properties. We also show that our framework works well in practice.",
    "Many data sets can be viewed as a noisy sampling of an underlying space, and tools from topological data analysis can characterize this structure for the purpose of knowledge discovery. One such tool is persistent homology, which provides a multiscale description of the homological features within a data set. A useful representation of this homological information is a persistence diagram (PD). Efforts have been made to map PDs into spaces with additional structure valuable to machine learning tasks. We convert a PD to a finite- dimensional vector representation which we call a persistence image (PI), and prove the stability of this transformation with respect to small perturbations in the inputs. The discriminatory power of PIs is compared against existing methods, showing significant performance gains. We explore the use of PIs with vector-based machine learning tools, such as linear sparse support vector machines, which identify features containing discriminating topological information. Finally, high accuracy inference of parameter values from the dynamic output of a discrete dynamical system (the linked twist map) and a partial differential equation (the anisotropic Kuramoto-Sivashinsky equation) provide a novel application of the discriminatory power of PIs.",
    "We study a version of the proximal gradient algorithm for which the gradient is intractable and is approximated by Monte Carlo methods (and in particular Markov Chain Monte Carlo). We derive conditions on the step size and the Monte Carlo batch size under which convergence is guaranteed: both increasing batch size and constant batch size are considered. We also derive non- asymptotic bounds for an averaged version. Our results cover both the cases of biased and unbiased Monte Carlo approximation. To support our findings, we discuss the inference of a sparse generalized linear model with random effect and the problem of learning the edge structure and parameters of sparse undirected graphical models."
]