safeNLP - Safety-critical NLP benchmark for VNNCOMP
========

Structure
------------
```
.
├── onnx
│   ├── medical                                      - folder containing the onnx network for the Medical dataset
│   └── ruarobot                                     - folder containing the onnx network for the R-U-A-Robot dataset
│   
├── vnnlib
│   ├── medical                                      - folder containing the queries in vnnlib format for the Medical dataset
│   └── ruarobot                                     - folder containing the queries in vnnlib format for the R-U-A-Robot dataset
│
└── instances.csv                                    - list of network-query-timeout instances
```

Details
------------

* Datasets:
<center>
<table style="width:90%">
  <tr>
    <th>Name</th>
    <th>Number of instances</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><a href='https://aclanthology.org/2022.aacl-short.30/'>Medical</a></td>
    <td>2886</td>
    <td> The Medical safety dataset is a written English dataset consisting of 2,917 risk-graded medical and non medical queries (1,417 and 1,500 examples respectively). The dataset was constructed via collecting questions posted on reddit, such as r/AskDocs. The medical queries have been labelled by experts and crowd annotators for both relevance and levels of risk (i.e. non-serious, serious to critical) following established World Economic Forum (WEF) risk levels designated for chatbots in healthcare. We merge the medical queries of different risk-levels into one class, given the high scarcity of the latter 2 labels to create an in-domain/out-of-domain classification task for medical queries. Additionally, we consider only the medical queries that were labelled as such by expert medical practitioners. Thus this dataset will facilitate discussion on how to guarantee a system recognises medical queries, in order to avoid generating medical output.</td>
  </tr>
  <tr>
    <td><a href='https://aclanthology.org/2021.acl-long.544/'>R-U-A-Robot</a></td>
    <td>7926</td>
    <td>The R-U-A-Robot dataset is a written English dataset consisting of 6800 variations on queries relating to the intent of ‘Are you a robot?’, such as ‘I’m a man, what about you?’. The dataset was created via a context-free grammar template, crowd-sourcing and pre-existing data sources. It consists of 2,720 positive examples (where given the query, it is appropriate for the system to state its non-human identity), 3,400 negative/adversarial examples and 680‘ambiguous-if-clarify’ examples (where it is unclear whether the system is required to state its identity). The dataset was created to promote transparency which may be required when the user receives unsolicited phone calls from artificial systems. Given systems like Google Duplex, and the criticism it received for human-sounding outputs, it is also highly plausible for the user to be deceived regarding the outputs generated by other NLP-based systems. Thus we choose this dataset to understand how to enforce such disclosure requirements. We collapse the positive and ambiguous examples into one label, following the principle of ‘better be safe than sorry’, i.e. prioritising a high recall system.</td>
  </tr>
</table>
</center>

* Networks:
<center>
<table style="width:90%">
  <tr>
    <th>Input dimension</th>
    <th>Layers type</th>
    <th>Layers number</th>
    <th>Layers size</th>
    <th>Activation functions</th>
  </tr>
  <tr>
    <td>30</td>
    <td>Fully connected</td>
    <td>2</td>
    <td>(128, 2)</td>
    <td>ReLU</td>
  </tr>
</table>
</center>