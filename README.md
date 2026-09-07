# RBM-Based Course Recommendation System

A backend machine learning system that recommends relevant courses to learners based on their previous course-enrollment interactions using a **Restricted Boltzmann Machine (RBM)**.

The system learns patterns from historical learner-course interactions and uses those learned patterns to generate personalized course recommendations.

---

## 1. Project Overview

Online learning platforms may contain thousands of courses, making it difficult for learners to identify courses relevant to their interests and previous learning activity.

This project aims to build a **Python backend recommendation system** that learns from historical course-enrollment data and recommends courses that a learner may be interested in.

### Core idea

```text
Historical Learner-Course Interactions
                │
                ▼
       Data Preprocessing
                │
                ▼
      Learner-Course Matrix
                │
                ▼
         Train RBM Model
                │
                ▼
      Generate Recommendations
                │
                ▼
        Python Backend API
                │
                ▼
            JSON Response
```

---

# 2. Problem Statement

The system recommends suitable courses to learners based on their previous course-enrollment interactions.

Given a learner and their historical course interactions, the backend should identify courses that the learner has not previously enrolled in and rank them according to the model's predicted relevance.

---

# 3. Project Goals

The primary goals are:

* Understand and preprocess course-enrollment data.
* Identify learners, courses, and enrollment interactions.
* Convert historical interactions into a learner-course interaction matrix.
* Train an RBM on the interaction matrix.
* Learn latent patterns in learner-course behavior.
* Generate course recommendations for a learner.
* Expose the recommendation functionality through a Python backend API.
* Evaluate the recommendation system using appropriate recommendation metrics.

---

# 4. Scope

## In Scope

* Course-enrollment dataset analysis
* Data preprocessing
* Learner-course interaction matrix construction
* RBM-based recommendation model
* Model training
* Recommendation generation
* Recommendation evaluation
* Python backend API
* Model persistence/loading
* Basic automated tests
* Documentation

## Out of Scope

* Frontend UI
* Web/mobile application interface
* User authentication
* Payment functionality
* Course creation/management
* Course content delivery
* Real-time user tracking
* Production-scale distributed infrastructure

This repository focuses specifically on the **backend recommendation engine and its supporting ML/data pipeline**.

---

# 5. System Definition

| Component             | Definition                                    |
| --------------------- | --------------------------------------------- |
| Target User           | Learner / Student                             |
| Recommendation Item   | Course                                        |
| User-Item Interaction | Course Enrollment                             |
| Input                 | Learner's previous course interactions        |
| Processing            | Data preprocessing → interaction matrix → RBM |
| Output                | Ranked recommended courses                    |
| ML Model              | Restricted Boltzmann Machine                  |
| Programming Language  | Python                                        |
| Application Type      | Backend-only                                  |
| API Framework         | FastAPI                                       |
| Interface             | REST API                                      |

---

# 6. Example

Suppose learner `U001` has previously enrolled in:

```text
Python
Statistics
Machine Learning
Deep Learning
```

The backend processes these interactions through the trained RBM.

The model may return:

```json
{
  "user_id": "U001",
  "recommendations": [
    {
      "course_id": "C021",
      "score": 0.91
    },
    {
      "course_id": "C034",
      "score": 0.87
    },
    {
      "course_id": "C047",
      "score": 0.82
    }
  ]
}
```

The exact recommendation scores and courses will depend on the trained model and dataset.

### Python representation

The recommendation request can be represented using Python:

```python
user_id = "U001"

previous_courses = [
    "Python",
    "Statistics",
    "Machine Learning",
    "Deep Learning"
]
```

The recommendation service could eventually return:

```python
recommendations = [
    {
        "course_id": "C021",
        "score": 0.91
    },
    {
        "course_id": "C034",
        "score": 0.87
    },
    {
        "course_id": "C047",
        "score": 0.82
    }
]
```

---

# 7. Data Definition

The project is based on a course-enrollment dataset.

The required conceptual structure is:

```text
Learner ─────── Enrollment ─────── Course
```

At minimum, the dataset should allow us to identify:

### User

The learner/student who interacts with courses.

Example:

```python
user_id = "U001"
```

### Item

The course being interacted with.

Example:

```python
course_id = "C001"
```

### Interaction

The learner's interaction with the course.

For this project:

```python
interaction = "enrollment"
```

A simplified dataset may look like:

| user_id | course_id | interaction |
| ------- | --------- | ----------- |
| U001    | C001      | 1           |
| U001    | C005      | 1           |
| U002    | C001      | 1           |
| U002    | C010      | 1           |
| U003    | C005      | 1           |

Where:

```text
1 = learner enrolled in the course
```

---

# 8. Interaction Matrix

The raw enrollment records will be transformed into a learner-course interaction matrix.

Example:

| Learner | Python | ML | Statistics | Deep Learning | NLP |
| ------- | -----: | -: | ---------: | ------------: | --: |
| U001    |      1 |  1 |          1 |             1 |   0 |
| U002    |      1 |  1 |          0 |             0 |   1 |
| U003    |      0 |  1 |          1 |             0 |   0 |

For the initial RBM approach:

```text
1 → observed enrollment
0 → no observed enrollment
```

This binary representation is subject to validation against the actual dataset.

The matrix can be represented in Python using NumPy:

```python
import numpy as np

interaction_matrix = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0]
])
```

---

# 9. Why RBM?

A Restricted Boltzmann Machine is a generative neural network consisting of:

* Visible units
* Hidden units

The visible layer represents observed data.

For this project:

```text
Visible units → Courses
```

A learner's interaction vector becomes the visible input.

Example:

```text
Python        → 1
Machine ML    → 1
Statistics    → 1
Deep Learning → 0
NLP           → 0
```

The hidden layer learns latent patterns in course-enrollment behavior.

Conceptually:

```text
                 Hidden Units
              ┌────┬────┬────┐
              │ H1 │ H2 │ H3 │
              └─┬──┴─┬──┴─┬──┘
                │    │    │
        ┌───────┼────┼────┼───────┐
        ▼       ▼    ▼    ▼       ▼
     Python    ML  Stats  DL     NLP
        Visible / Course Units
```

The learned relationships can then be used to estimate which courses a learner may be interested in.

---

# 10. Recommendation Flow

The recommendation process will follow approximately this flow:

```text
                 API Request
                     │
                     ▼
                learner_id
                     │
                     ▼
       Retrieve learner history
                     │
                     ▼
       Convert to interaction vector
                     │
                     ▼
              Trained RBM
                     │
                     ▼
       Reconstruct / score courses
                     │
                     ▼
     Remove already-enrolled courses
                     │
                     ▼
          Rank recommendations
                     │
                     ▼
                Return top-K
```

---

# 11. Backend Architecture

The project is backend-only.

A high-level architecture:

```text
Client / API Consumer
         │
         ▼
    FastAPI REST API
         │
         ▼
 Recommendation Service
         │
         ├──────────────┐
         ▼              ▼
   Learner Data      RBM Model
         │              │
         └───────┬──────┘
                 ▼
       Recommendation Engine
                 │
                 ▼
        Ranked Course List
                 │
                 ▼
             JSON Response
```

There will be **no frontend/UI component** in this repository.

---

# 12. Proposed Python API

The backend will use **FastAPI** to expose the recommendation service.

## Recommendation Endpoint

```http
GET /recommendations/{user_id}
```

Example:

```http
GET /recommendations/U001
```

### FastAPI implementation concept

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    recommendations = recommendation_service(user_id)

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }
```

The `recommendation_service()` function will contain the recommendation logic rather than putting the ML logic directly inside the API route.

For example:

```python
def recommendation_service(user_id: str):
    # 1. Retrieve learner history
    # 2. Create interaction vector
    # 3. Pass vector to trained RBM
    # 4. Generate course scores
    # 5. Remove already-enrolled courses
    # 6. Sort courses
    # 7. Return top-K

    return [
        {
            "course_id": "C021",
            "score": 0.91
        },
        {
            "course_id": "C034",
            "score": 0.87
        }
    ]
```

The above is an architectural example. The actual implementation will be developed after the dataset and RBM design are finalized.

---

# 13. Example API Response

Request:

```http
GET /recommendations/U001
```

Response:

```json
{
  "user_id": "U001",
  "recommendations": [
    {
      "course_id": "C021",
      "score": 0.91
    },
    {
      "course_id": "C034",
      "score": 0.87
    },
    {
      "course_id": "C047",
      "score": 0.82
    }
  ]
}
```

The API returns JSON and does not require a frontend application.

---

# 14. ML Pipeline

The planned ML pipeline is:

```text
Raw Dataset
    │
    ▼
Data Validation
    │
    ▼
Data Cleaning
    │
    ▼
Identify Users & Courses
    │
    ▼
Create Interaction Matrix
    │
    ▼
Train/Test Split
    │
    ▼
RBM Training
    │
    ▼
Model Evaluation
    │
    ▼
Save Trained Model
    │
    ▼
Recommendation Service
```

Python will be used throughout the data and ML pipeline.

Example preprocessing:

```python
import pandas as pd

df = pd.read_csv("data/raw/course_enrollments.csv")

df = df.dropna(subset=["user_id", "course_id"])

interaction_matrix = (
    df.assign(interaction=1)
      .pivot_table(
          index="user_id",
          columns="course_id",
          values="interaction",
          fill_value=0
      )
)
```

The exact column names will depend on the actual dataset.

---

# 15. Data Engineering Responsibilities

The data pipeline should:

1. Load the course-enrollment dataset.
2. Inspect available fields.
3. Identify the learner/user identifier.
4. Identify the course/item identifier.
5. Identify the enrollment interaction.
6. Remove or handle invalid records.
7. Handle duplicate interactions where necessary.
8. Handle missing user/course identifiers.
9. Encode users and courses.
10. Construct the learner-course interaction matrix.
11. Prepare training and evaluation data.
12. Provide reusable preprocessing logic for inference.

---

# 16. ML Engineering Responsibilities

The ML pipeline should:

1. Research RBMs and recommendation use cases.
2. Define the RBM input representation.
3. Determine the number of visible units.
4. Determine an appropriate hidden-layer size.
5. Train the RBM.
6. Experiment with training parameters.
7. Generate course scores/reconstructions.
8. Filter courses already enrolled in by the learner.
9. Rank candidate courses.
10. Return top-K recommendations.
11. Evaluate the recommendation quality.
12. Save the trained model and required preprocessing artifacts.

---

# 17. Backend Engineering Responsibilities

The Python backend should:

1. Load the trained RBM model.
2. Load required preprocessing artifacts.
3. Accept a learner identifier.
4. Retrieve/construct the learner's interaction vector.
5. Pass the vector to the recommendation engine.
6. Generate course scores.
7. Filter previously enrolled courses.
8. Rank recommendations.
9. Return a structured JSON response.
10. Handle invalid learner IDs and other API errors.

---

# 18. QA Responsibilities

Quality assurance will focus on:

### Data validation

Verify:

* User IDs are valid.
* Course IDs are valid.
* Enrollment records are correctly interpreted.
* Missing values are handled.
* Duplicate records are handled appropriately.

### ML validation

Verify:

* Interaction matrix has the expected dimensions.
* Model receives correctly formatted input.
* Recommendations are generated.
* Previously enrolled courses are excluded.
* Top-K behavior works correctly.

### API validation

Verify:

```text
Valid learner
      ↓
FastAPI endpoint
      ↓
200 response
      ↓
Recommendations
```

And:

```text
Invalid learner
      ↓
FastAPI endpoint
      ↓
Appropriate HTTP error
```

### Documentation validation

Verify that:

* Project requirements are documented.
* Dataset assumptions are documented.
* Technical decisions are recorded.
* API behavior is documented.
* Known limitations are documented.

---

# 19. Acceptance Criteria

The initial system will be considered successful when:

* [ ] A learner can be uniquely identified from the dataset.
* [ ] A course can be uniquely identified from the dataset.
* [ ] Enrollment can be represented as a learner-course interaction.
* [ ] A learner-course interaction matrix can be constructed.
* [ ] The dataset contains enough information for the proposed RBM approach.
* [ ] The RBM can be trained successfully on the prepared data.
* [ ] A trained model can generate course scores/recommendations.
* [ ] Previously enrolled courses are excluded from recommendations.
* [ ] Recommendations can be ranked.
* [ ] The Python backend can return top-K recommendations for a valid learner.
* [ ] Invalid learner requests are handled appropriately.
* [ ] Basic automated tests pass.
* [ ] The complete data → model → recommendation pipeline is reproducible.

---

# 20. Evaluation

The recommendation system should not be evaluated only by checking whether the model trains successfully.

We need to evaluate recommendation quality.

Potential metrics include:

* Precision@K
* Recall@K
* Hit Rate@K
* Mean Average Precision (MAP)
* NDCG@K

The final evaluation metric(s) will be selected after understanding the dataset and defining an appropriate train/test methodology.

A key consideration is avoiding data leakage when splitting historical interactions.

---

# 21. Important Technical Questions

The following questions must be resolved during implementation.

### Dataset

* Does the dataset contain enough learner-course interactions?
* Are interactions strictly enrollments?
* Are there duplicate enrollments?
* How sparse is the interaction matrix?
* Are there enough interactions per learner?
* Are there enough interactions per course?

### RBM

* How many hidden units should be used?
* Which RBM implementation/library should be used?
* Which training algorithm should be used?
* How many epochs are required?
* What learning rate should be used?
* How should sparsity be handled?
* How should model convergence be evaluated?

### Recommendation

* How should reconstructed course probabilities be converted into recommendations?
* How should already-enrolled courses be filtered?
* What value of K should be used?
* How should learners with very little history be handled?

### Evaluation

* What train/test split is appropriate?
* Which recommendation metrics are most appropriate?
* How do we establish a baseline?
* Does RBM outperform a simpler recommendation approach?

### Backend

* How should the trained model be loaded?
* Where should preprocessing mappings be stored?
* How should unknown learners be handled?
* What API response format should be standardized?
* Should the API use synchronous or asynchronous endpoints?
* How should model errors be handled?

---

# 22. Repository Structure

The proposed repository structure is:

```text
rbm-course-recommender/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── rbm.py
│   │   ├── train.py
│   │   ├── recommend.py
│   │   └── evaluate.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   │
│   └── config.py
│
├── tests/
│   ├── test_data.py
│   ├── test_rbm.py
│   ├── test_recommendations.py
│   └── test_api.py
│
├── docs/
│   └── problem-definition.md
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 23. Directory Responsibilities

### `data/`

Dataset files and processed data.

```text
raw/        → Original dataset
processed/  → Cleaned/transformed data
```

Raw datasets should not be modified directly.

---

### `models/`

Stores trained model artifacts.

Example:

```text
models/
├── rbm_model
├── user_mapping
└── course_mapping
```

Large model files should be managed appropriately rather than blindly committed to Git.

---

### `notebooks/`

Used for:

* Dataset exploration
* Visualization
* Experiments
* Initial model investigation

Production logic should eventually move into `src/`.

---

### `src/data/`

Responsible for:

```text
Load → Validate → Clean → Transform
```

---

### `src/ml/`

Responsible for:

```text
Train → Predict → Recommend → Evaluate
```

---

### `src/api/`

Responsible for the **FastAPI backend**.

---

### `tests/`

Automated tests for:

* Data processing
* RBM behavior
* Recommendation generation
* FastAPI endpoints

---

### `docs/`

Project documentation and requirements.

---

# 24. Development Workflow

The project should be developed in stages.

### Phase 1 — Problem Definition

```text
Define user
Define item
Define interaction
Define input
Define output
Select dataset
```

### Phase 2 — Data Understanding

```text
Load dataset
      ↓
Explore fields
      ↓
Validate learner/course/interaction
      ↓
Check data quality
```

### Phase 3 — Data Preparation

```text
Clean data
   ↓
Encode users/courses
   ↓
Create interaction matrix
   ↓
Prepare training/evaluation data
```

### Phase 4 — RBM Development

```text
Define RBM
   ↓
Train
   ↓
Tune
   ↓
Evaluate
```

### Phase 5 — Recommendation Engine

```text
Learner
   ↓
Interaction vector
   ↓
RBM
   ↓
Course scores
   ↓
Filter existing courses
   ↓
Top-K recommendations
```

### Phase 6 — Python Backend

```text
FastAPI
   ↓
Recommendation Service
   ↓
RBM Model
   ↓
JSON Response
```

### Phase 7 — Testing

```text
Unit Tests
    +
Integration Tests
    +
API Tests
    +
Recommendation Evaluation
```

---

# 25. Example End-to-End Python Workflow

A learner requests recommendations from the FastAPI backend.

### Step 1 — API request

```http
GET /recommendations/U001
```

### Step 2 — FastAPI receives the request

```python
@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    return recommendation_service(user_id)
```

### Step 3 — Recommendation service retrieves history

```python
def recommendation_service(user_id: str):

    history = get_user_history(user_id)

    interaction_vector = create_interaction_vector(history)

    scores = rbm_recommend(interaction_vector)

    recommendations = filter_and_rank(
        user_id=user_id,
        scores=scores
    )

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }
```

### Step 4 — RBM generates course scores

Conceptually:

```python
scores = rbm.predict(interaction_vector)
```

The exact implementation depends on the RBM framework and model design selected by the ML Engineer.

### Step 5 — Filter already enrolled courses

```python
recommendations = [
    course
    for course in ranked_courses
    if course not in previously_enrolled_courses
]
```

### Step 6 — Return top-K courses

```python
top_k = recommendations[:5]
```

### Step 7 — FastAPI returns JSON

```json
{
  "user_id": "U001",
  "recommendations": [
    {
      "course_id": "C021",
      "score": 0.91
    },
    {
      "course_id": "C034",
      "score": 0.87
    },
    {
      "course_id": "C047",
      "score": 0.82
    }
  ]
}
```

The complete flow is therefore:

```text
Python Client / API Consumer
          │
          ▼
       FastAPI
          │
          ▼
Recommendation Service
          │
          ▼
 Learner History
          │
          ▼
Interaction Vector
          │
          ▼
      Trained RBM
          │
          ▼
   Course Scores
          │
          ▼
 Filter Existing Courses
          │
          ▼
       Top-K
          │
          ▼
     JSON Response
```

---

# 26. Engineering Principles

The project should follow these principles:

### Separation of concerns

Data processing, ML logic, recommendation logic and API logic should remain separate.

### Reproducibility

The same preprocessing and model artifacts should produce consistent recommendations.

### Testability

Core components should be independently testable.

### No hardcoded learner/course mappings

Mappings should be generated and persisted as model artifacts.

### No training inside API requests

The API should use a previously trained model.

Training and inference should be separate processes.

```text
Training:

Dataset → Preprocessing → RBM Training → Saved Model


Inference:

API Request → Preprocessing → Saved Model → Recommendation
```

---

# 27. Security & Data Considerations

The project should avoid committing sensitive or private learner information to the repository.

The `.gitignore` should exclude:

```text
.env
data/raw/*
models/*
__pycache__/
.pytest_cache/
```

A small sample dataset may be included for development/testing if it does not contain sensitive information.

---

# 28. Current Limitations

The initial version has several expected limitations:

* Course enrollment does not necessarily represent course preference.
* Binary interactions lose information about interaction frequency or intensity.
* New learners may suffer from the cold-start problem.
* New courses may suffer from the item cold-start problem.
* Sparse interaction matrices may make learning difficult.
* RBM performance depends heavily on data quality and representation.
* Recommendation quality must be compared against simpler baselines.

These limitations should be evaluated during development.

---

# 29. Future Improvements

Possible future improvements include:

* Incorporating course metadata.
* Using additional interaction types such as completion or ratings if available.
* Hybrid recommendation approaches.
* Better cold-start handling.
* Candidate generation followed by ranking.
* Model versioning.
* Experiment tracking.
* Monitoring recommendation quality.
* Dockerized deployment.
* Cloud deployment.
* Database integration.
* Authentication and authorization.

These are outside the initial project scope.

---

# 30. Project Success Criteria

The project will be considered successful if we can demonstrate the complete backend pipeline:

```text
Course Enrollment Dataset
          ↓
Data Processing
          ↓
Learner-Course Matrix
          ↓
RBM Training
          ↓
Trained Model
          ↓
Learner API Request
          ↓
Recommendation Engine
          ↓
Top-K Course Recommendations
          ↓
FastAPI JSON Response
```

The final system should be able to take a valid learner identifier and return a ranked list of relevant courses that the learner has not already enrolled in.

---

# 31. Team Responsibilities

### Data Engineer

Owns:

* Dataset investigation
* Data validation
* Data cleaning
* Interaction matrix creation
* Data pipeline

### ML Engineer

Owns:

* RBM research
* Model design
* Model training
* Recommendation algorithm
* Model evaluation

### Software Engineer

Owns:

* Python backend architecture
* FastAPI implementation
* Recommendation service integration
* Model serving
* Application structure

### QA & Documentation

Owns:

* Requirements
* Acceptance criteria
* Test planning
* Documentation
* Requirement consistency
* Recording team decisions
* Validation of data/ML/API behavior

### Team Lead

Owns:

* Project coordination
* Technical decisions
* Task allocation
* Milestones
* Integration across roles
* Final project direction

---

# 32. Current Project Status

| Area                  | Status                     |
| --------------------- | -------------------------- |
| Problem Definition    | In Progress                |
| Dataset Selection     | Course Enrollments Dataset |
| Data Analysis         | Pending                    |
| Interaction Matrix    | Pending                    |
| RBM Research          | Pending                    |
| RBM Implementation    | Pending                    |
| Recommendation Engine | Pending                    |
| FastAPI Backend       | Pending                    |
| Testing               | Pending                    |
| Evaluation            | Pending                    |
| Deployment            | Future                     |

---

# 33. Final Architecture

The intended backend architecture is:

```text
                    ┌──────────────────────┐
                    │    API Consumer      │
                    └──────────┬───────────┘
                               │
                               │ GET /recommendations/{user_id}
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │     API Layer        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Recommendation       │
                    │ Service              │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌──────────────────┐       ┌──────────────────┐
        │ Learner History  │       │   Trained RBM    │
        │ / Interaction    │       │      Model       │
        │ Data             │       └────────┬─────────┘
        └────────┬─────────┘                │
                 │                          │
                 └────────────┬─────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Recommendation       │
                    │ Generation           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Ranked Top-K Courses │
                    └──────────┬───────────┘
                               │
                               ▼
                         JSON Response
```

---

# 34. Key Project Principle

The project should follow one central definition:

```text
USER        → Learner
ITEM        → Course
INTERACTION → Enrollment
INPUT       → Previous learner-course interactions
MODEL       → RBM
OUTPUT      → Recommended courses
LANGUAGE    → Python
BACKEND     → FastAPI
```

Everything in the data pipeline, ML pipeline and backend should remain consistent with these definitions.
