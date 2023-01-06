============================================
Experimenting with clustering text documents
============================================
.. contents:: **Contents**
   :depth: 4
   :local:
   :backlinks: top
   
I am basing my experimentation with clustering text on the very great scikit-learn's tutorial: `Clustering text documents using k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html>`_.

I am following along their tutorial but using my own two datasets: a bunch of ebooks (``pdf``, ``djvu``) and Wikipedia pages (``html``).

1. Clustering ebooks (``pdf``, ``djvu``)
========================================
The dataset of ebooks that I used to test clustering consists of 129 ebooks (``pdf`` and ``djvu``) from 3 categories:

- ``computer_science`` with label 0 and 48 ebooks
- ``mathematics`` with label 1 and 50 ebooks
- ``physics`` with label 2 and 31 ebooks

.. code-block::

   Feature Extraction using TfidfVectorizer
   vectorization done in 1.406 s
   n_samples: 129, n_features: 7909
   Sparsity: 0.117

- Ignored terms: 

  - if they appear in more than 50% of the documents
  - if they are not present in at least 5 documents
- Around 11.7% of the entries of the ``X_tfidf`` matrix are non-zero

To quickly check the clustering results which are surprisingly good considering the small dataset of ebook text: 
`Results of clustering ebooks <#results-of-clustering-ebooks-pdf-and-djvu>`_

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
  books based on a chosen language
* `regex <https://pypi.org/project/regex/>`_: **v2022.7.9**, "this regex implementation is backwards-compatible with 
  the standard ``re`` module, but offers additional functionality"
* `scikit-learn <https://scikit-learn.org/>`_: **v1.0.2**, "a set of python modules for machine learning and data mining"

**Ref.:** https://docs.anaconda.com/anaconda/packages/py3.7_osx-64/

|

`:star:` **Other dependencies**

You also need recent versions of:

-  `poppler <https://poppler.freedesktop.org/>`_ (including ``pdftotext``) and `DjVuLibre <http://djvu.sourceforge.net/>`_ (including ``djvutxt``)
   can be installed for conversion of ``.pdf`` and ``.djvu`` files to ``.txt``, respectively.

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
--ud, --update-dataset                 Update dataset with text from more new ebooks found in the directory.

|

`:information_source:` Explaining some important and interesting options/arguments

- ``input_directory`` is the path to the main directory containing the documents to cluster.
- By **dataset re-creation** I mean the case when you delete the pickle dataset file and generate the dataset 
  again. If you are using cache, then the dataset generation should be quick since the text conversions were
  already computed and cached. Using the option ``-u`` is worthwhile especially if you used OCR for some of the ebooks since this procedure is very
  resource intensive and can take awhile if many pages are OCRed.
- The choices for ``-o, --ocr-enabled`` are ``{always, true, false}``
  
  - 'always': always use OCR first when doing text conversion. If the converson fails, then use the other simpler conversion tools
    (``pdftotext`` and ``djvutxt``).
  - 'true': first simpler conversion tools (``pdftotext`` and ``djvutxt``) will be used and then if a conversion
    failed to convert an ebook to ``txt`` or resulted in an empty file, the OCR method will be used.
  - 'false': never use OCR, only use the other simpler conversion tools (``pdftotext`` and ``djvutxt``).

Caching
"""""""
`:information_source:` About the caching option (``--use-cache``) supported by the script ``cluster_text_docs.py``

- Cache is used to save the converted ebook files into ``txt`` to
  avoid re-converting them which can be a time consuming process. 
  `DiskCache <http://www.grantjenks.com/docs/diskcache/>`_, a disk and file 
  backed cache library, is used by the ``cluster_text_docs.py`` script.
- Two default cache folders are used:

  - ``~/.cluster_html``: used when clustering HTML pages
  - ``~/.cluster_ebooks``: used when clustering ebooks
  
  You can also specify your own cache folder for 
  
  - HTML pages with the option ``--cfh PATH``
  - ebooks with the option ``--cfe PATH``
- The MD5 hashes of the ebook files are used as keys to the file-based cache.
- These hashes of ebooks (keys) are then mapped to a dictionary with the following structure:

  - key: ``convert_method+convert_only_percentage_ebook+ocr_only_random_pages``
  
    where 
    
    - ``convert_method`` is either ``djvutxt`` or ``pdftotext``
    - ``convert_only_percentage_ebook`` is the percentage of a given ebook that is converted to ``txt``
    - ``ocr_only_random_pages`` is the number of pages chosen randomly in the first 50% of a given ebook
      that will be OCRed
      
    e.g. djvutxt+15+3
    
  - value: the extracted text based on the options mentioned in the associated key
  
  Hence, you can have multiple extracted texts associated with a given ebook with each of the text
  extraction based on different values of the options mentioned in the key.

- In the `case for HTML pages <#2-clustering-wikipedia-pages>`_, the hashes of these pages are directly mapped to extracted text.

|

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

Show number of items in a given cache
'''''''''''''''''''''''''''''''''''''
To show the number of items (i.e. ebooks whose text was extracted) for a **given cache**, the options ``-n`` and ``--cfe`` are used::

 $ python cluster_text_docs.py -n --cfe ~/.cluster_ebooks_test/
 
 Cache: ~/.cluster_ebooks_test/
 There are 138 items in cache
 
`:information_source:`

 - ``-n, --number-items``: Shows number of items stored in cache.
 - ``--cfe, --cache-folder-ebooks PATH``: Cache folder for ebooks.

|

If you don't specify a specific cache folder, then the default cache folder used for HTML pages (see the second part of this document about `clustering Wikipedia pages <#2-clustering-wikipedia-pages>`_) will be selected::

 $ python cluster_text_docs.py -n
 
 Cache: ~/.cluster_html
 There are 71 items in cache

|

To show the number of items in the default cache used for ebooks (i.e. ``~/.cluster_ebooks/``), the option ``-t ebooks`` is used::

 $ python cluster_text_docs.py -n -t ebooks

 Cache: ~/.cluster_ebooks
 There are 153 items in cache

Remove items from a given cache
'''''''''''''''''''''''''''''''
To remove items (i.e. texts from ebooks) from a **given cache**, the options ``-r`` and ``--cfe`` are used along with the corresponding hashes
associated with the texts you want to remove since file hashes are used as keys mapping to texts in the cache::

 $ python cluster_text_docs.py -r 123 1234 --cfe ~/.cluster_ebooks_test/
 
 Removing keys from cache: ~/.cluster_ebooks_test/
 Key=123 was not found in cache
 Key=1234 was not found in cache
 
`:information_source:`

 - ``-r, --remove-keys KEY [KEY ...]``: Keys (MD5 hashes of ebooks) to be removed from the cache along with the 
   texts associated with them. Thus be careful before deleting them.
 - ``--cfe, --cache-folder-ebooks PATH``: Cache folder for ebooks.

Clear a given cache
'''''''''''''''''''
To clear a given cache, the option ``-c`` is used::

 $ python cluster_text_docs.py -c ~/.cluster_ebooks_test/ 
 
 Clearing cache: ~/.cluster_ebooks_test/
 Cache was already empty!
 
`:information_source:`

 - ``-c, --clear-cache PATH``: Path to the cache folder to be cleared. Be careful before using this option since everything
   in cache will be deleted including the text conversions.
 - ``--cfe, --cache-folder-ebooks PATH``: Cache folder for ebooks.

Ebooks directory
""""""""""""""""
`:warning:` In order to run the script `cluster_text_docs.py <./scripts/cluster_text_docs.py>`_, you need first to have a main directory (e.g. ``./ebooks/``) with all the ebooks (``pdf`` and ``djvu``) you want to test clustering on. Each ebook should be in a folder whose name should correspond to the category of said ebook.

For example:

- ../ebooks/**biology**/Cell theory.djvu
- ../ebooks/**philosophy**/History of Philosophy in Europe.pdf
- ../ebooks/**physics**/Electricity.pdf

Then, you need to give the path to the main directory to the script, like this::

 $ python cluster_text_docs.py -t ebooks ~/Data/ebooks/
 
The next section explains in details the generation of a dataset containing text from these ebooks.

Dataset generation
""""""""""""""""""
To start generating a dataset containing texts from ebooks after you have setup your directory of ebooks, the option ``-t ebooks`` and the input directory are necessary::

 $ python cluster_text_docs.py -t ebooks ~/Data/ebooks_test/
 
`:information_source:` Explaining the text conversion procedure

- It is necessary to specify the type of dataset (``-t ebooks``) you want to generate because the script can also be used to `generate datasets
  from HTMl pages <#2-clustering-wikipedia-pages>`_.
- The script will try to convert each ebook to text by using ``pdftotext`` or ``djvutxt`` depending on the type of file.
- By default, OCR is not used (``--ocr-enabled`` is set to 'false') since it is a very resource intensive procedure. The other
  simpler conversion methods (``pdftotext`` or ``djvutxt``) are used instead which are very quick and reliable in their text conversion of ebooks.
- By default, only 10% of a given ebook is converted to text. The option ``--cope, --convert-only-percentage-ebook PAGES`` controls
  this percentage.
- If the text conversion fails with the simpler tools ((``pdftotext`` or ``djvutxt``) because an ebook is composed of images 
  for example, then a warning message is printed suggesting you to use OCR which should be able to fix the problem but if too many ebooks
  are images then it might not be practicable to use OCR if updating the dataset afterward.
- The hash of each ebook is computed so as to avoid adding duplicates in the dataset. Also the hashes are used as keys in the cache if
  caching is used (i.e. the option ``-u, --use-cache`` is enabled).

|

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

|

Generating the ebooks dataset using cache (``-u`` option) without OCR support (i.e. the ``-o true`` option is not used)::

 $ python cluster_text_docs.py -t ebooks -u ~/Data/ebooks_test/

First time running the script with a cleared cache:

.. raw:: html

   <p align="left"><img src="./images/dataset_generation_first_time_used_cache.png">
   </p>

|

Second time running the script with some of the text conversions already cached:


.. raw:: html

   <p align="left"><img src="./images/dataset_generation_second_time_used_cache.png">
   </p>

|

Warning message shown when a text conversion fails (e.g. the ebook is made up of images):

.. raw:: html

   <p align="left"><img src="./images/dataset_generation_conversion_failed_use_ocr.png">
   </p>
   
`:information_source:` The dataset generation can be re-run again after with the ``-o true --ud`` options which enable the use of OCR for those
problematic ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``).

|

When a duplicate is found (based on MD5 hashes), the correponding ebook is not processed further:

.. raw:: html

   <p align="left"><img src="./images/dataset_generation_found_duplicate.png">
   </p>

|

At the end of the dataset generation, some results are shown about the number of texts
added to the dataset and cache, books rejected and duplicates found

.. raw:: html

   <p align="left"><img src="./images/dataset_generation_end_results2.png">
   </p>

OCR
"""
For those ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``), 
you can run the dataset generation using the ``-o true`` and ``--ud`` options::

 $ python cluster_text_docs.py -t ebooks -u -o true --ud ~/Data/ebooks_test/

`:information_source:` 

 - The ``--ud`` flag refers to the action of updating the dataset pickle file that was already saved within the main ebooks directory
   (e.g. ``~/Data/ebooks_test/``)
 - The OCR procedure is resource intensive, thus the conversion for those problematic ebooks might take longer than usual.

|

Loading a dataset and applying OCR to those ebooks that couldn't be converted to ``txt`` with simpler methods (``pdftotext`` and ``djvutxt``):

 .. raw:: html

   <p align="left"><img src="./images/updating_dataset_ocr.png">
   </p>

|

Results at the end of applying OCR to all problematic ebooks (maded of images):

.. raw:: html

   <p align="left"><img src="./images/updating_dataset_ocr_end_results.png">
   </p>
   
`:information_source:` All 14 problematic ebooks (made up of images) were successfully converted to ``txt`` and added to the dataset and cache.

Updating a dataset
""""""""""""""""""
After a dataset is generated and saved, you can update it with new texts from more ebooks by using the ``--ud`` option::

 $ python cluster_text_docs.py -t ebooks -u -o true --ud ~/Data/ebooks_test/

.. raw:: html

   <p align="left"><img src="./images/updating_dataset_ocr.png">
   </p>
   
`:information_source:`

 - ``--ud``: tells the script to update the dataset pickle file saved within the main ebooks directory (e.g. ``~/Data/ebooks_test/``).
 - ``-o true``: apply OCR on those ebooks that couldn't be converted with simpler methods (``pdftotext`` and ``djvutxt``).
 - ``-u``: use cache to avoid re-computing the text conversion for those ebooks that were already processed previously.
 - ``-t ebooks``: tells the script that the input directory (e.g. ``~/Data/ebooks_test/``) should be search for
   ``pdf`` and ``djvu`` ebooks to be added to the dataset.

Filtering a dataset: select texts only in English and from valid categories
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
After the dataset containing texts from ebooks is generated, the resulting dataset is filtered by removing text that is not English
and not part of the specified categories (i.e. ``computer_science``, ``mathematics``, ``physics``).

Here are some samples of output from the script ``cluster_text_docs.py``::

 python cluster_text_docs.py --cfe ~/.cluster_ebooks_test -t ebooks -u ~/Data/ebooks_test/ --verbose
 
`:information_source:` Since the option ``--verbose`` is used, you will see more information printed in the terminal such as
if the text is in English or its category.

| 
 
Showing the categories that will be kept:

.. raw:: html

   <p align="left"><img src="./images/filtering_keeping_categories.png">
   </p>

|

Texts rejected for not being in English:

.. raw:: html

   <p align="left"><img src="./images/filtering_rejected_french_spanish.png">
   </p>
   
|

Texts rejected for not being part of the specified categories (``computer_science``, ``mathematics``, ``physics``):

.. raw:: html

   <p align="left"><img src="./images/filtering_rejected_politics.png">
   </p>

|

What it looks like in the terminal if the option ``--verbose`` is not used: only the list of rejected texts is shown after the
filtering is completed

.. raw:: html

   <p align="left"><img src="./images/filtering_no_verbose.png">
   </p>

`:information_source:` You will see in my list of ebooks that the text from the ebook ``abstract algebra.pdf`` was rejected even though it
is from an English mathematics ebook. ``pycld2`` detected the text as not being in English because the text conversion (``pdftotext``) didn't 100% succeeded and introduced too many odd characters (e.g. ``0ß Å ÞBð``) mixed with english words. It seems that it is the only ebook over 153 converted documents that has this problem.

Results of clustering ebooks (``pdf`` and ``djvu``) ⭐
------------------------------------------------------
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
The 10 most influential words for each cluster according to the KMean algorithm (with LSA on tf-idf vectors)::

   Cluster 0: geometry riemann euclidean universe triangle euclid lines angle ab earth 
   Cluster 1: quantum riemann zeta physics vector particle wave zeros hypothesis particles 
   Cluster 2: algorithm algorithms programming code gcd input python programs integer mod

Recall the `true labels <#clustering-ebooks-pdf-djvu>`_: computer_science, mathematics, physics.

Thus we could infer the labels for each cluster found by KMeans:

- Cluster 0: mathematics
- Cluster 1: physics
- Cluster 2: computer_science

In general, the top terms for each cluster are well selected by the KMeans algorithm. Though KMeans has some difficulty with the 
mathematics and physics categories as some words are misplaced such as universe & earth which
should be in the physics category and riemann & zeta should be in the mathematics category.

The top words for the other cluster (2) with the computer science related words are well choosen by KMeans.

2. Clustering Wikipedia pages
=============================
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

Wikipedia directory 
-------------------
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

Results of clustering Wikipedia pages ⭐
----------------------------------------
`:information_source:` A random model is also "trained" on this dataset and its performance is reported. This model
randomly generates the `labels <#2-clustering-wikipedia-pages>`_ (from 0 to 4) for the Wikipedia pages:

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

Recall the `true labels <#2-clustering-wikipedia-pages>`_: biology, chemistry, mathematics, philosophy, physics.

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
