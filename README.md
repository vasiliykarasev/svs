slic video segmentation
=======================

What this is
------------

interactive foreground/background "video" segmentation. 
uses SLIC superpixels (1) (with temporal extension) and a variant of Grabcut (2) 
to perform the segmentation. The optimization itself (binary segmentation with TV
regularization) is solved by Chambolle-Pock's primal-dual method (3).


(1) R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, *SLIC Superpixels Compared to State-of-the-art Superpixel Methods*, TPAMI, 2012

(2) C. Rother, V. Kolmogorov, and A. Blake. *Grabcut: Interactive foreground extraction using iterated graph cuts*, ACM Transactions on Graphics, 2004.

(3)  A. Chambolle, T. Pock, *A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging*, J.Math.Imaging Vis., 2011


How to compile 
==============

compile it with eclipse


about
==============
karasev00@gmail.com
