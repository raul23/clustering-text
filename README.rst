============================================
Experimenting with clustering text documents
============================================
.. contents:: **Contents**
   :depth: 4
   :local:
   :backlinks: top
   
I am basing my experimentation with clustering text on the very great scikit-learn's tutorial: `Clustering text documents using k-means <https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html>`_

I am following along their tutorial but using my own two datasets: Wikipedia pages and a bunch of ebooks with different formats (pdf, djvu, epub and txt).

Clustering HTML pages
=====================
The dataset of HTML pages is small: 70 Wikipedia pages from 5 different categories (biology, chemistry, mathematics, philosophy, physics).

I will eventually build a larger dataset but for now I just wanted to test out some of the clustering algorithms as soon as possible but even with
a small dataset, the clustering results are not that bad.

The list of these Wikipedia pages can be found at `list_wikipedia_pages.rst <./list_wikipedia_pages.rst>`_.

The size for each category:

- Biology: 12
- Chemistry: 12
- Mathematics: 10
- Philosophy: 16
- Physics: 20

This is the environment on which the `script <./scripts/cluster_text_docs.py >`_ was tested:

* **Platform:** macOS
* **Python**: version **3.7**
* `beautifulsoup4 <https://www.crummy.com/software/BeautifulSoup/>`_: **v4.11.1**, for screen-scraping

`:warning:` In order to run the script ``cluster_text_docs.py ``, you need first to have a directory with all the Wikipedia pages (\*.html) you want to parse. Each Wikipedia page should be in a folder whose name should correspond to the category of the page.

For example:

- ./wikipedia/biology/Cell theory.html
- ./wikipedia/philosophy/Cartesian doubt.html
- ./wikipedia/physics/Charge conservation.html

Then, you need to give the path to this directory to the script, like this::

 $ python extract_from_infobox.py ~/Data/wikipedia/

Clustering ebooks (PDF, djvu, epub, txt)
========================================
TODO
