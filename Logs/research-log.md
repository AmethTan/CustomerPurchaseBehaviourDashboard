The following is the research logs for the FYP and the dashboard development journal.

Research Planning Weeks 

Week 1 (9/28/25-10/4/2025)
Research planning and workflow deciding - Initial plan draft and schedule (daily & weekly) making
Research idea gathering -Search, gather and review elements related to the research
FYP 01 decided topic - Development of an E-commerce Customer Purchase Behaviour Analytic Dashboard using Machine Learning
Domain: E-commerce Customer Purchase Behaviour
Method: Analytic dashboard, Machine Learning techniques

Week 2 (10/5/25-10/11/2025)
Research idea understanding, determining research type and direction - review back to research FYP 01 works, research ideas gathered, current progress, and task listing
Task list: 
 Determine research type: Quantitative
 Defining key components of intro and LR - factors influencing e-commerce customer purchase behaviour, methods implemented for customer analysis: (the key components initially identified are shown below)
- Definition of customer purchase behaviour 
- Importance of customer purchase behaviour knowledge in e-commerce
- Factors affecting customer purchase behaviour and their indicators 
- Past efforts and research to understand and define customer purchase behaviour- Results and research gap identified. 
- Definition of analytical dashboard 
- Implications of analytical dashboard in e-commerce and related studies.
- Definition of machine learning 
- Implications of machine learning in e-commerce and related studies. 
- Past efforts and research to implement machine learning and analytic dashboards - Methodologies, results, and potential research gap established
- Hypotheses formulating and research gaps identifying
Search for research papers according to the key components, both in theoretical components and methodological components
Read, make short summaries, and organise the collected papers
- Steps to read the papers: Title, abstract, brief intro and conclusion in the introduction,  results' subheadings, conclusion, look back again to gather details, highlighting interesting references 
- put the gathered info into a table matrix
Write intro, LR, and methodology - Intro required only minor changes from FYP 01, but LR required large modifications with the research direction changes made last semester in FYP 01 - no longer attempt integrating Python with Tableau and shifting from Machine Learning and data visualising to machine learning assisted by data visuals from Python libraries
Find data - Kaggle / Open sources / Use API / web-scraping?
Confirm data choice 
Learn Python and any required libraries with online resources from W3Schools, DataCamp, etc.
Draft model design 
Check with SV the research direction - research ideas, data, and model design draft
Update research direction and model design accordingly
Try implementing the model design and building the app based on the following order:
1. Data preprocessing 
2. Model development 
3. Dashboard integration
4. Others, including debugging and additional aspects discovered later 
Intro & LR Planning 

Literature Review (LR) Formulating & Dashboard Design Preparing Weeks

Week 3 (10/12/25-10/18/2025)
Literature Summary Excel file preparation
Try run on recording key info of articles/research papers

Week 4 (10/19/25-10/25/2025)
Keywords and abstract extraction
Try out the AI method on the key findings extract
Finish extracting all the key findings/prints of research papers
Finish reorganising key findings/points 
Finish reorganising key findings/points 

Week 5 (10/26/25-11/1/2025)
Formulate LR
Reconsider LR flow as including all previously defined components would be too lengthy and some of them may be irrelevant to the main goal of the research.
New LR flow: 
1. Quote the two most famous and common versions of customer purchase behaviour definition, namely the Kotler and Keller version in the marketing textbook, and Solomon's version. Defining customer purchase behaviour as a whole from all discovered versions. In the context of e-commerce, define customer purchase behaviours.
2. Factors, indicators, and variables affecting e-commerce customer purchase behaviour
3. Types of customer purchase behaviour 
4. Stages/Phases of customer purchase behaviour 
5. Analytical techniques applied to customer purchase behaviour - Traditional and statistical
6. Analytical techniques applied to customer purchase behaviour - Modern, machine learning based and dashboard-related
7. Implementations of techniques and technology stacks on e-commerce customer purchase behaviour machine learning application  
8. Research gaps identified and to be tackled in this research. 
Fully defined customer purchase behaviour 

Week 6 (11/2/25-11/8/2025)
Defined the 5 dimensions of influence/factors on customer purchase behaviour. 
Listed down and grouped all the identified factors from the notes according to the dimensions
Narrate the four main types of customer purchase behaviour identified 
Narrate the identified steps/phases of customer purchase 

Week 7 (11/9/25-11/15/2025)
Roughly finished parts of LR, aim at finish drafting LR and research idea to meet SV before mid-semester break:
Narrate the traditional statistical methods implemented in the past to analyse customer purchase behaviour 
Narrate the machine learning techniques utilised to analyse customer purchase behaviour 
Narrate the methods used to monitor and improve machine learning results
Narrate the technology stacks utilised to develop an analytic dashboard, particularly in integrating machine learning to customer purchase behaviour 
Concluding research gaps that are going to be tackled in the research 
Decided and drew out clearly the research flow - consider the research to be a hybrid research with qualitative part being a systematic literature review to frame the e-commerce customer purchase behaviour and determine best machine techniques to analyse it while the quantitative part design, develop and analyse the machine learning analytic dashboard.
Came up with a dashboard design with KPIs derived from factors of e-commerce customer purchase behaviour
Selecting datasets 
Book first appointment with SV, Dr. Aamir Adeeb
First appointment with Dr. Aamir Adeeb: 
Discussed the research direction, key focus/questions/dashboard design elements suggested for the project: 
- What do the boss / key stakeholders want from the dashboard?
- Product purchase prediction - Will the customer buy the product?
- Which products do customers buy instead, if they do not buy the currently viewed one?
- What can lead to purchase conversion/sales?
Suggested to focus on developing the dashboard based on one united aim instead of a multifaceted KPI-based dashboard, because the latter would take a long time, while only being suitable for monitoring and management purposes
Suggested Streamlit for faster dashboard development 
Suggested using regression modelling for purchase prediction. Based on the new dashboard design direction, determine the best machine learning techniques.  
Suggested to use HuggingFace instead for dataset finding because it offers larger, more advanced, and big datasets.
This week: dataset finding. Next week: draft model. To leave time for model update and debugging while completing the model will at least put something on the table, even if dashboard development faces problems. 
Devised a plan for the changes in research direction: 
1. Decide on the dashboard design.
2. Find out the required machine learning techniques.
3. List down crucial variables and search for datasets accordingly using Hugging Face.
4. Develop the selected model/models. 
Decided on a purchase prediction designed dashboard.

Dashboard Developing Weeks 

Week 8 (11/23/25-11/29/2025)
Looking back at the collected past literature and online resources, defining use cases, and reorganising machine learning techniques implemented according to the use cases. Compare each model based on performance and scalability/applicability. 
Decided on designing and implementing a dashboard which predict purchase possibility of products, suggesting product bundles to improve average order value, and suggest products where customer will click next and with good purchase potential. 
Decided on implementing XGBoost, FP-growth and SASRec where XGBoost is considered the most common and superior direct competitor to regression; FP-growth is superior than Apriori for basket analysis and product bundling; and SASRec is standard method for next-click prediction with consistently great performance and implement complexity.   
Began to search for suitable datasets for the developing the models, through Hugging Face.
Based on the search, only a handful dataset fulfil the requirements of having clickstream data especially event type data which records the transition of pages of e-commerce customer from viewing to purchasing. 
Found "kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019", but the dataset is tool large with high chances of crashing to train the model. From the same page, found the user's another dataset's link to Kaggle, namely the dataset "eCommerce events history in electronics store" which is significant smaller in file size but still contain good amount of data to work with. After continue searching and comparing to other datasets, decided to use  the "eCommerce events history in electronics store" dataset. 

Week 9 (11/30/25-12/6/2025)
Search and learn from online tutorials on creating machine learning implemented Streamlit dashboard.
Design, draw and finalising the dashboard sketch.
Designed three sections of the dashboard: Purchase prediction, product bundles and next buy suggestions, and a clickstream simulator. 
Reviewing past literature, online tutorials, and notes on data preprocessing.
Begin discovering, mark down imperfections, and cleaning the dataset using Microsoft Excel.
Deal with duplicated rows, and null values.
Fill in unknown category codes uniquely with unique numbers after "unknown".
Search and learn from online tutorials on developing the the three models.
Install and import all neccessary Python libraries.
Prepare code for extra preprocessing.

Week 10 (12/7/25-12/13/2025)
Code the three models in "model.py".
Try run "model.py" and export the models into pickle format.
Error encounter for SASRec model when trying to use TensorFlow.
Try PyTorch code for SASRec but fail again. 
Search and find the error code appeared online, and try troubleshooting using various sources' suggestions. 
Error persist even if updates C++ compiler. 

Week 11 (12/14/25-12/20/2025)
Continues troubleshooting the Tensorflow incompatibility.
Repair and update the dashboard's virtual environment.
Tried specifically use Tensorflow CPU only but error persist, high possibility GPU incompatible.
Switch to PyTorch CPU only. 
Troubleshoot library incompatibilites.
Successfully run "model.py" and trained all three models.
Prepare "model_report.py" and check model performance. 
Minor tweaks in model parameters achieve model performance with satisfactory results. 
Design and implement the Streamlit dashboard code. 
Try ran "dashboard.py".
Perform minor debugs. 
Try ran again with success. 
Test each sections and note down possible changes and additions to be made.
Critical problems discovered: XGBoost customer-related inputs are insufficient to provide better insights on output; FP-growth has identified no rules; SASRec shows the "Echo" effect where it mostly predicts that the most probable next-click is the currently viewed product of the customer because of the high repeated click instants in the dataset; previous model performance report do not thoroughly examine every aspect of the model performance.
Identified the need of feature engineer variables like recency, frequency, monetary, average order vale, page viewed and several others.
Identified the need of tweaking the model parameters of FP-Growth, especially min_support.
Studied and noted down all related performance metrics for the three model.
Improve the model report with rule listing, key performance metrics calculations and sketching other related performance graphs.

Week 12 (12/21/25-12/27/2025)
Code and run "xgboost_retrain.py" and "fpgrowth_retrain.py".
FP-growth model stabilise at min_support = 0.00008 with several thousands of rules and L-shaped support vs, confidence graph.
Meanwhile, XGBoost achieve only mediocre result (medium PR-ROC score) due to high level of data imbalance regarding purchase vs. non-purchase event.
Built and coded "xgboost_tuning.py" to perform hyperparameter tuning on the XGBoost model using Bayesian Optimisation. The code also compare SMOTE-ENN, SMOTE-ENN with scale_pos_weight, and scale_pos_weight only.
scale_pos_weight with Bayesian Optimisation obtain the best PR-ROC score, so the exported parameter is implemented into "xgboost_retrain.py".
Finish updated the dashboard design with other features like advanced product search, product detail checker, and etc.
Functionalities check for the dashboard. 
Dashboard fully functional with core functions performing well.
Planning on report content.

Report Formulating and Symposium Preparing Weeks

Week 13 (12/28/25-1/3/2026)
Touched up the previous remaining drafted parts of LR.
Prepare results and discussion part with the content decided as below: 
Summaries of the systematic literature review: 
- Factors of customer purchase behaviour
- Types of customer purchase behaviour 
- Phases of customer purchase behaviour
- Use cases of machine learning analytics in analysing e-commerce customer purchase behaviour
- Model comparisons to select the best machine learning model baseline for each use case
Results and notable discussion on the development of the finished dashboard, "E-commerce CusPurBeAD v1.0":
- Data preprocessing
- Model built
- Dashboard built
Prepared the summaries of the systematic literature review in the form of diagrams and tables.
Thought of the peculiar transitions from the summaries of SLR to the dashboard developed. 
Realised the hybrid research direction is aimed at understanding customer purchase behaviour and taking the best possible action to developed a machine learning analytic dashboard to improve e-commerce sales. 
Decided on proposing a framework for developing an e-commerce customer purchase behaviour analytic dashboard using machine learning techniques. 
Developed the ADFCBv framework to assist quick and procedural development of e-commerce customer purchase behaviour machine learning application. 
Narrated the ADFCBv framework in the report.
Explained the results and notable discussion on the development of "E-commerce CusPurBeAD v1.0" in the flow of the ADFCBv framework. 

Week 14 (1/4/26-1/10/2026)
Drafted the methodology part and the conclusion part of the report. 
Updated the intro part of the report (from FYP01).
Prepared the poster for presentation purpose.
Attended the SQS symposium and presented the result and findings of the research in front of two evaluators, namely Dr. Ch'ng Chee Keong and Dr. Mazlan Mohd Bin Sappri.

Research Report and E-Portfolio Finalisation Weeks

Finalised and touched up the drafted intro, methodology and conclusion part of the report. 
Formulated the abstract of the study. 
Touched up report with cover page, table of content, and appendix.
Printed and handed-in the report to SV, Dr. Aamir Adeeb 
Prepared the research e-portfolio. 

