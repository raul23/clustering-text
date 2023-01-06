============================================
Experimenting with clustering text documents
============================================
.. contents:: **Contents**
   :depth: 4
   :local:
   :backlinks: top
   
I am basing my experimentation with clustering text on the very great scikit-learn's tutorial: `Clustering text documents using k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html>`_.

I am following along their tutorial but using my own two datasets: a bunch of ebooks with different formats (``pdf``, ``djvu``) and 
Wikipedia pages (html).

Clustering ebooks (``pdf``, ``djvu``)
=====================================
The dataset of ebooks that I used to test clustering consists of 129 ebooks (``pdf`` and ``djvu``) from 3 categories:

- ``computer_science`` with label 0 and 48 ebooks
- ``mathematics`` with label 1 and 50 ebooks
- ``physics`` with label 2 and 31 ebooks

.. code-block::

   Feature Extraction using TfidfVectorizer
   vectorization done in 1.043 s
   n_samples: 129, n_features: 7884
   Sparsity: 0.117

- Ignored terms: 

  - if they appear in more than 50% of the documents
  - if they are not present in at least 5 documents
- Around 11.7% of the entries of the ``X_tfidf`` matrix are non-zero

Script ``cluster_text_docs.py`` (part 1)
----------------------------------------
Dependencies
""""""""""""
This is the environment on which the script `cluster_text_docs.py <./scripts/cluster_text_docs.py>`_ was tested:

* **Platform:** macOS
* **Python**: version **3.7**
* `matplotlib <https://matplotlib.org/>`_: **v3.5.2** for generating graphs
* `numpy <https://numpy.org/>`_: **v1.21.5**, for "array processing for numbers, strings, records, and objects"
* `pandas <https://pandas.pydata.org/>`_: **v1.3.5**, "High-performance, easy-to-use data structures and data analysis tool" 
* `pycld2 <https://github.com/aboSamoor/pycld2>`_: **v0.41**, for detecting the language of a given ebook in order to keep 
  books based on chosen language
* `regex <https://pypi.org/project/regex/>`_: **v2022.7.9**, "this regex implementation is backwards-compatible with 
  the standard ``re`` module, but offers additional functionality."
* `scikit-learn <https://scikit-learn.org/>`_: **v1.0.2**, "a set of python modules for machine learning and data mining"

**Ref.:** https://docs.anaconda.com/anaconda/packages/py3.7_osx-64/

|

`:star:` **Other dependencies**

You also need recent versions of:

-  `poppler <https://poppler.freedesktop.org/>`_ and `DjVuLibre <http://djvu.sourceforge.net/>`_ can be installed 
   for conversion of ``.pdf`` and ``.djvu`` files respectively to ``.txt``.

Optionally:

- `diskcache <http://www.grantjenks.com/docs/diskcache/>`_: **v5.4.0** for caching persistently the converted files into ``txt``
- `Tesseract <https://github.com/tesseract-ocr/tesseract>`_ for running OCR on books - version 4 gives 
  better results. OCR is disabled by default since it is a slow resource-intensive process.

Script options for clustering ebooks
""""""""""""""""""""""""""""""""""""
To display the script's list of options and their descriptions::

 $ python cluster_text_docs.py -h
 usage: python cluster_text_docs.py [OPTIONS] {input_directory}

I won't list all options (too many) but here some of the important and interesting ones:

-s, --seed SEED                        Seed for numpy's and Python's random generators. (default: 123456)
-u, --use-cache                        Highly recommended to use cache to speed up **dataset re-creation**.
-t, --dataset-type DATASET_TYPE        Whether to cluster html pages or ebooks (``pdf`` and ``djvu``). By default, 
                                       only HTML pages are clustered from within the specified directory. (default: html)
-o, --ocr-enabled                      Whether to enable OCR for ``pdf``, ``djvu`` and image files. It is disabled by default. (default: false)

|

`:information_source:` Explaining some of these options/arguments

- ``input_directory`` is the path to the main directory containing the documents to cluster.
- By **dataset re-creation** I mean what happens when you delete the pickle dataset file and generate the dataset 
  again. If you are using cache, then the dataset generation should be quick since the text conversions were
  already computed and cached. Especially if you used OCR for some of the ebooks since this procedure is very
  resource intensive and can take awhile if many pages are OCRed.
- The choices for ``-o, --ocr-enabled`` are ``{always, true, false}``
  
  - 'always': always use OCR first when doing text conversion. If the converson fails, then use the other simpler conversion tools
    (``pdftotext`` and ``djvutxt``).
  - 'true': first simpler conversion tools (``pdftotext`` and ``djvutxt``) will be used first and then if that conversion
    failed to convert an ebook to ``txt`` or resulted in an empty file, the OCR method will be used.
  - 'false': never use OCR, only use the other simpler conversion tools (``pdftotext`` and ``djvutxt``)

Caching
"""""""
`:information_source:` About the caching option (``--use-cache``) supported by the script ``cluster_text_docs.py``

- Cache is used to save the converted ebook files into ``txt`` to
  avoid re-converting them which can be a time consuming process. 
  `DiskCache <http://www.grantjenks.com/docs/diskcache/>`_, a disk and file 
  backed cache library, is used by the ``cluster_text_docs.py`` script.
- The MD5 hashes of the ebook files are used as keys to the file-based cache.

`:warning:` Important things to keep in mind when using the caching option

* When enabling the cache with the flag ``--use-cache``, the ``cluster_text_docs.py`` 
  script has to cache the converted ebooks (``txt``) if they were
  not already saved in previous runs. Therefore, the speed up of some of the
  tasks (dataset re-creation and updating) will be seen in subsequent executions of the 
  script.
* Keep in mind that caching has its caveats. For instance if a given ebook
  is modified (e.g. a page is deleted) then the ``cluster_text_docs.py`` 
  script has to run the text conversion again since the keys in the cache are the MD5 hashes of
  the ebooks.
* There is no problem in the
  cache growing without bounds since its size is set to a maximum of 1 GB by
  default (check the ``--cache-size-limit`` option) and its eviction policy
  determines what items get to be evicted to make space for more items which
  by default it is the least-recently-stored eviction policy (check the
  ``--eviction-policy`` option).

Ebooks dataset structure
------------------------
`:warning:` In order to run the script `cluster_text_docs.py <./scripts/cluster_text_docs.py>`_, you need first to have a main directory (e.g. ``./ebooks/``) with all the ebooks (``pdf`` and ``djvu``) you want to test clustering on. Each ebook should be in a folder whose name should correspond to the category of said page.

For example:

- ../ebooks/**biology**/Cell theory.djvu
- ../ebooks/**philosophy**/History of Philosophy in Europe.pdf
- ../ebooks/**physics**/Electricity.pdf

Then, you need to give the path to the main directory to the script, like this::

 $ python cluster_text_docs.py ~/Data/ebooks/ -t ebooks --use-cache

`:warning:` When generating datasets from ebooks (instead of datasets from HTML pages like in the `second part <#clustering-wikipedia-pages>`_ 
of this document), always use the ``-t ebooks`` option which tells the script that the input directory given contains ebooks and therefore should be search for these kinds of documents (``pdf`` and ``djvu``). When generating datasets from HTML pages, you don't need to specify this option since by default the script treats the input directory as potentially
containing HTML pages.

|

`:information_source:` The first time the script is run, the dataset of text (from ebooks) will be generated. This dataset is a `Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html>`_ object (a dictionary-like object that allows you to access its values by keys or attributes) with the following structure:

- ``data``: list of shape (n_samples,)
- ``filenames``: list of shape (n_samples,)
- ``target_names``:  list of shape (n_classes,)
- ``target``: ndarray of shape (n_samples,)
- ``DESCR``: str, the full description of the dataset

It is the same structure as the one used by scikit-learn for their `datasets <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html>`_.

The label used by ``target`` is automatically generated by assigning integers (from the range ``[0, number of classes - 1]``) to each sample. 

The dataset is saved as a pickle file under the main directory that you provided to the script.

The next times the script is run, the dataset will be loaded from disk as long as you don't delete or move the pickle file saved directly under the main directory.

Results of clustering ebooks (``pdf`` and ``djvu``)
---------------------------------------------------
`:information_source:` A random model is also "trained" on this dataset and its performance is reported. This model
randomly generates the `labels <#clustering-ebooks-pdf-djvu>`_ (from 0 to 2) for the ebooks:

.. code-block:: python

   self.labels_ = np.random.randint(0, self.n_clusters, X.shape[0])

But keep in mind what they say about random labeling in scikit-learn's tutorial `Clustering text documents using k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering-evaluation-summary>`_:

 The homogeneity, completeness and hence v-measure metrics do not yield a baseline with regards to random labeling: 
 this means that depending on the number of samples, clusters and ground truth classes, a completely random labeling will 
 not always yield the same values.

|

+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
|                         | RandomModel    | KMeans on tf-idf vectors  | KMeans with LSA on tf-idf vectors  | MiniBatchKMeans with LSA on tf-idf vectors  | KMeans with LSA on hashed vectors  | MiniBatchKMeans with LSA on hashed vectors  |
+=========================+================+===========================+====================================+=============================================+====================================+=============================================+
| Time                    | 0.01 ± 0.00 s  | 0.11 ± 0.01 s             | 0.00 ± 0.00 s                      | 0.04 ± 0.02 s                               | 0.01 ± 0.00 s                      | 0.04 ± 0.00 s                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Homogeneity             | 0.018 ± 0.011  | 0.564 ± 0.085             | 0.486 ± 0.070                      | 0.449 ± 0.131                               | 0.531 ± 0.152                      | 0.491 ± 0.135                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Completeness            | 0.017 ± 0.011  | 0.598 ± 0.074             | 0.496 ± 0.084                      | 0.466 ± 0.116                               | 0.579 ± 0.144                      | 0.543 ± 0.116                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| V-measure               | 0.017 ± 0.011  | 0.580 ± 0.080             | 0.491 ± 0.077                      | 0.457 ± 0.124                               | 0.553 ± 0.147                      | 0.515 ± 0.126                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Adjusted Rand-Index     | 0.005 ± 0.014  | 0.523 ± 0.107             | 0.450 ± 0.060                      | 0.401 ± 0.177                               | 0.479 ± 0.185                      | 0.451 ± 0.171                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Silhouette Coefficient  | -0.004 ± 0.001 | 0.049 ± 0.003             | 0.048 ± 0.008                      | 0.051 ± 0.003                               | 0.051 ± 0.004                      | 0.051 ± 0.002                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+

.. raw:: html

   <p align="center"><img src="./images/results_clustering_ebooks.png">
   </p>

Top terms per cluster (ebooks)
------------------------------
TODO

Clustering Wikipedia pages
==========================
The dataset of HTML pages is small: 70 Wikipedia pages from 5 categories

- ``biology`` with label 0
- ``chemistry`` with label 1
- ``mathematics`` with label 2
- ``philosophy`` with label 3
- ``physics`` with label 4

I will eventually build a larger dataset but for now I just wanted to test out some of the clustering algorithms as soon as possible but even with
a small dataset, the clustering `results <#results-of-clustering-wikipedia-pages>`_ are not that bad.

The list of these Wikipedia pages can be found at `List of Wikipedia pages used for clustering <./list_wikipedia_pages.rst>`_.

The **size** for each category:

- Biology: 12
- Chemistry: 12
- Mathematics: 10
- Philosophy: 16
- Physics: 20

.. code-block::

   Feature Extraction using TfidfVectorizer
   vectorization done in 0.530 s
   n_samples: 70, n_features: 5474
   Sparsity: 0.166

- Ignored terms: 

  - if they appear in more than 50% of the documents
  - if they are not present in at least 5 documents
- Around 16.6% of the entries of the ``X_tfidf`` matrix are non-zero

Script ``cluster_text_docs.py`` (part 2)
----------------------------------------
This is the environment on which the script `cluster_text_docs.py <./scripts/cluster_text_docs.py>`_ was tested:

* **Platform:** macOS
* **Python**: version **3.7**
* `beautifulsoup4 <https://www.crummy.com/software/BeautifulSoup/>`_: **v4.11.1**, for retrieving the only the text from an HTML page
* `matplotlib <https://matplotlib.org/>`_: **v3.5.2** for generating graphs
* `numpy <https://numpy.org/>`_: **v1.21.5**, for "array processing for numbers, strings, records, and objects"
* `pandas <https://pandas.pydata.org/>`_: **v1.3.5**, "High-performance, easy-to-use data structures and data analysis tool" 
* `scikit-learn <https://scikit-learn.org/>`_: **v1.0.2**, "a set of python modules for machine learning and data mining"

**Ref.:** https://docs.anaconda.com/anaconda/packages/py3.7_osx-64/

Wikipedia dataset structure
---------------------------
`:warning:` In order to run the script `cluster_text_docs.py <./scripts/cluster_text_docs.py>`_, you need first to have a main directory (e.g. ``./wikipedia/``) with all the Wikipedia pages (``*.html``) you want to test clustering on. Each Wikipedia page should be in a folder whose name should correspond to the category of said page.

For example:

- ../wikipedia/**biology**/Cell theory.html
- ../wikipedia/**philosophy**/Cartesian doubt.html
- ../wikipedia/**physics**/Charge conservation.html

Then, you need to give the path to the main directory to the script, like this::

 $ python cluster_text_docs.py ~/Data/wikipedia/

`:information_source:` The first time the script is run, the dataset of HTML documents will be generated. This dataset is a `Bunch <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html>`_ object (a dictionary-like object that allows you to access its values by keys or attributes) with the following structure:

- ``data``: list of shape (n_samples,)
- ``filenames``: list of shape (n_samples,)
- ``target_names``:  list of shape (n_classes,)
- ``target``: ndarray of shape (n_samples,)
- ``DESCR``: str, the full description of the dataset

It is the same structure as the one used by scikit-learn for their `datasets <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html>`_.

The label used by ``target`` is automatically generated by assigning integers (from the range ``[0, number of classes - 1]``) to each sample. 

The dataset is saved as a pickle file under the main directory that you provided to the script.

The next times the script is run, the dataset will be loaded from disk as long as you don't delete or move the pickle file saved directly under the main directory.

Results of clustering Wikipedia pages
-------------------------------------
`:information_source:` A random model is also "trained" on this dataset and its performance is reported. This model
randomly generates the `labels <#clustering-wikipedia-pages>`_ (from 0 to 4) for the Wikipedia pages:

|

+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
|                         | RandomModel    | KMeans on tf-idf vectors  | KMeans with LSA on tf-idf vectors  | MiniBatchKMeans with LSA on tf-idf vectors  | KMeans with LSA on hashed vectors  | MiniBatchKMeans with LSA on hashed vectors  |
+=========================+================+===========================+====================================+=============================================+====================================+=============================================+
| Time                    | 0.00 ± 0.00 s  | 0.10 ± 0.00 s             | 0.00 ± 0.00 s                      | 0.05 ± 0.02 s                               | 0.00 ± 0.00 s                      | 0.03 ± 0.00 s                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Homogeneity             | 0.112 ± 0.035  | 0.591 ± 0.066             | 0.587 ± 0.063                      | 0.513 ± 0.073                               | 0.556 ± 0.093                      | 0.527 ± 0.114                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Completeness            | 0.111 ± 0.035  | 0.610 ± 0.050             | 0.605 ± 0.060                      | 0.591 ± 0.030                               | 0.578 ± 0.093                      | 0.597 ± 0.088                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| V-measure               | 0.112 ± 0.035  | 0.600 ± 0.057             | 0.596 ± 0.062                      | 0.548 ± 0.054                               | 0.566 ± 0.092                      | 0.559 ± 0.104                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Adjusted Rand-Index     | 0.019 ± 0.025  | 0.477 ± 0.082             | 0.450 ± 0.095                      | 0.394 ± 0.119                               | 0.429 ± 0.094                      | 0.382 ± 0.121                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+
| Silhouette Coefficient  | -0.012 ± 0.001 | 0.047 ± 0.007             | 0.043 ± 0.010                      | 0.040 ± 0.011                               | 0.034 ± 0.006                      | 0.028 ± 0.023                               |
+-------------------------+----------------+---------------------------+------------------------------------+---------------------------------------------+------------------------------------+---------------------------------------------+

.. raw:: html

   <p align="center"><img src="./images/results_clustering_html_pages_3.png">
   </p>

Top terms per cluster (Wikipedia pages)
---------------------------------------
The 10 most influential words for each cluster according to the KMean algorithm (with LSA on tf-idf vectors)::

   Cluster 0: probability language statistical reality realism events scale sample interpretation hypothesis 
   Cluster 1: cell dna biology cells genes gene organisms bacteria population genetic 
   Cluster 2: chemical chemistry equilibrium reaction bond gas atoms mathrm reactions compounds 
   Cluster 3: relativity motion speed mathbf spacetime wave frame conservation waves charge 
   Cluster 4: mathematics logic geometry algebra discrete reasoning mind numbers socratic descartes 

Recall the `true labels <#clustering-wikipedia-pages>`_: biology, chemistry, mathematics, philosophy, physics.

Thus we could infer the labels for each cluster found by KMeans:

- Cluster 0: philosophy
- Cluster 1: biology
- Cluster 2: chemistry
- Cluster 3: physics
- Cluster 4: mathematics

In general, the top terms for each cluster are well selected by the KMeans algorithm. Though KMeans has some difficulty with the 
philosophy and mathematics categories as some words are misplaced such as socratic which
should be in the philosophy category and probability & statistical should be in the mathematics category.

`:information_source:` From some of the Wikipedia pages forming the `dataset <./list_wikipedia_pages.rst>`_:

 - `Mathematics <https://en.wikipedia.org/wiki/Mathematics>`_: Socrates, Descartes and mind are mentioned zero, 
   twice and seven times, respectively.
 - `Philosophy <https://en.wikipedia.org/wiki/Philosophy>`_: Only once is the word statistical mentioned and
   probability is not mentioned at all.
 - `Socratic questioning <https://en.wikipedia.org/wiki/Socratic_questioning>`_: no mention of mathematics at all.

The top words for the other clusters 1 to 3 (in particular cluster 1 with the biology-related words) are well choosen by KMeans.
