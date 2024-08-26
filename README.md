.. class:: center

    |sai| |itmo|


# ProtoLLM

**Proto-LLM** is an open-source framework for fast protyping of LLM-based applications


Proto LLM features
==================
- Rapid prototyping of information retrieval systems based on BNM using RAG:
Implementations of architectural patterns for interacting with different databases and web service interfaces;
Methods for optimising RAG pipelines to eliminate redundancy.
- Development and integration of applications with BNM with connection of external services and models through plugin system:
Integration with AutoML solutions for predictive tasks;
Providing structured output generation and validation;
- Implementation of ensemble methods and multi-agent approaches to improve the efficiency of BNMs:
Possibility of combining arbitrary BNMs into ensembles to improve generation quality, automatic selection of ensemble composition;
Work with model-agents and ensemble pipelines;
- Generation of complex synthetic data for further training and improvement of BNM: Generating examples from existing models and data sets;
Evolutionary optimisation to increase the diversity of examples; Integration with Label Studio; 
- Providing interoperability with various LLM providers:
Support for native models (GigaChat, YandexGPT, vsegpt, etc.). 
Interaction with open-source models deployed locally.


Analogs
=======

LLMware, LLMStach, LangChain

However, they are not direct competitors to the framework being created, as it is a higher-level tool that uses existing LLMOps solutions where possible and necessary, and provides compatibility with them for most tasks.

Project Structure
=================

The latest stable release of ProtoLLM is in the `master branch <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master>`__.

The repository includes the following directories:

* Package `core <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/protollm>`__  contains the main modules. It is the *core* of the ProtoLLM framework
* Package `examples <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/examples>`__ includes several *how-to-use-cases* where you can start to discover how ProtoLLM works
* All *unit and integration tests* can be observed in the `test <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/test>`__ directory
* The sources of the documentation are in the `docs <https://github.com/ITMO-NSS-team/ProtoLLM/tree/master/docs>`__ directory

Contribution Guide
==================

- The contribution guide is available in this `repository <https://github.com/ITMO-NSS-team/ProtoLLM/blob/master/docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and
workshops for their valuable advice and suggestions.

Supported by
============

The study is supported by the Research `Center Strong Artificial Intelligence in Industry <https://sai.itmo.ru/>`_
of `ITMO University <https://itmo.ru/>`_ as part of the plan of the center's program.


Contacts
========
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, Team leader (anna.kalyuzhnaya@itmo.ru)
- `Newsfeed <https://t.me/NSS_group>`_
- `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_


.. |ITMO| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/ITMO_badge.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |SAI| image:: https://raw.githubusercontent.com/aimclub/open-source-ops/43bb283758b43d75ec1df0a6bb4ae3eb20066323/badges/SAI_badge.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/
