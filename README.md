# support-ticket-ai-agent
Multi-tier AI helpdesk engine combining automated text classification, natural language querying (LLM), statistical anomaly detection, and telemetry tracking.



# AI-Powered Support Ticket Analyzer & Telemetry Dashboard
An end-to-end intelligence system built to automate helpdesk workflows, track performance metrics, and handle conversational analytics. This production-ready prototype bridges core data mechanics with Large Language Models (LLMs) using a decoupled, multi-tier microservice architecture.



## 🏗️ System Architecture

The application is structured into two main microservices containerized independently and bridged via a network bridge:

* **FastAPI Backend (`/app`):** Handles environment configurations, parses the core datasets, executes text classification analytics, runs anomaly heuristics, and hosts RESTful API routing endpoints.
* **Streamlit Frontend (`/ui`):** A web dashboard offering reactive metric visualization grids (Total Leads, Transmission Quantities, Funnel Open/Conversion Rates) and an ad-hoc Natural Language query bar.



## 🛠️ Tech Stack & Prerequisites

* **Language:** Python 3.10+
* **Frameworks:** FastAPI, Streamlit, Pandas, SQLite
* **AI Engine:** OpenAI API (GPT-based processing)
* **Containerization:** Docker & Docker Compose



## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/SatutiKesar/support-ticket-ai-agent).git]
cd YOUR_REPO_NAME
