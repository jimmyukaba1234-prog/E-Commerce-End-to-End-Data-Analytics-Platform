E-Commerce End-to-End Data Analytics Platform

This repository contains a full end-to-end e-commerce data analytics platform designed to provide deep visibility into customer behavior, operational performance, and revenue drivers. The project addresses a common business challenge: while transactions occur consistently, there is limited clarity on why customers churn, which products and regions drive profit, and where operational inefficiencies impact growth.

The solution is built on large-scale historical e-commerce data spanning multiple domains, including customers (50,500 rows × 24 columns), orders (101,000 rows × 24 columns), products (10,300 rows × 30 columns), and reviews (50,400 rows × 20 columns). A relational database was first designed and populated, after which an automated ETL pipeline was developed to extract, clean, transform, and integrate the data into analytics-ready tables.

Sample Database (Upload via Web App)

A pre-built SQLite database is provided so users can immediately test the full functionality of the web application without running any ETL scripts locally.

Download the database here:
👉 https://drive.google.com/file/d/1BlVZUbcUUVuYTZXYHfy0kBaN8Y83VGN8/view?usp=sharing

How to use the database

Download the Ecommerce.db file from the link above

Launch the web application

Navigate to the “Upload Database” section

Upload the downloaded Ecommerce.db file

The dashboards, analytics, models, and reports will load automatically

⚠️ Note: The database is not required to be placed in the project directory. All interactions are handled through the web application’s database upload interface.

Following data preparation, comprehensive exploratory analysis, business intelligence dashboards, and visualizations were created to answer key business questions around customer retention, churn patterns, regional performance, product profitability, fulfillment efficiency, payment failures, and long-term financial trends. Insights reveal a “leaky bucket” retention pattern, declining sales and profit over time, operational bottlenecks in delivery and payments, and a heavy concentration of revenue among a small group of high-value customers.

To move beyond descriptive analytics, predictive models were built for customer churn and customer lifetime value (LTV). The churn model leverages RFM features—especially recency—to identify high-risk customers early, enabling targeted re-engagement strategies. The LTV model highlights how future revenue is driven by a small number of high-value customers with infrequent but high-ticket purchases, supporting smarter prioritization and resource allocation.

The project also includes automated reporting, generating stakeholder-ready PDF reports, and a web application that allows users to interact with insights in real time. Overall, this repository demonstrates a complete analytics lifecycle from database design and data engineering to modeling, reporting, and deployment aimed at driving data-informed decision-making, improving retention, and supporting sustainable e-commerce growth.


roject Role & Contributions

Role: Team Lead | Data Engineer | Data Analyst

This project was completed by a five-person team, where I served as the team lead. Beyond coordinating the team and ensuring deadlines were met, I was deeply involved in the core technical implementation of the platform.

My Key Contributions

Led project planning, provided technical direction, and ensured timely delivery under a tight deadline.

Designed and implemented the end-to-end ETL pipeline, handling data extraction, cleaning, transformation, and integration into analytics-ready tables.

Built the web application that enables users to upload the database, interact with dashboards, and explore insights in real time.

Developed automation workflows, including automated data processing and generation of stakeholder-ready PDF reports.

Created and integrated interactive dashboards and visualizations to surface insights around churn, revenue, operational performance, and customer behavior.

Supported model development by preparing features and analytics outputs used for churn and LTV prediction.

Team Collaboration

Collaborated with team members on exploratory analysis, business insights, and model interpretation.

Incorporated team feedback during development while making firm technical decisions when alignment was required to meet deadlines.
